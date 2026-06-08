import math
import threading
from math import atan2, radians, degrees

import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import sys
sys.path.append('../')
sys.path.append('/home/paar-core0/python_tracker/tools/python_tracking/')
import time 
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Imu
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import SetMode, CommandBool

from jetson_pysot_tracker import (
    set_tracking_point,
    get_current_bbox,
    run_pysot_tracker,
    enable_tracking_and_load_model,
    push_frame,
)

# ─────────────────────────────────────────────────────────────────────────────
# Startup sequence state machine
# ─────────────────────────────────────────────────────────────────────────────
#
#   IDLE
#    │  (heartbeat has been streaming >2 s)
#    ▼
#   ARMING  ──► arm service call ──► wait for State.armed == True
#    │
#    ▼
#   TAKEOFF ──► ramp thrust from hover to climb for TAKEOFF_DURATION seconds
#    │
#    ▼
#   TRACKING ──► normal tracker_ang loop
#
# ─────────────────────────────────────────────────────────────────────────────

MODEL_INIT = "MODEL_INIT"   # wait for tracker model to load before anything else
IDLE       = "IDLE"
ARMING     = "ARMING"
TAKEOFF    = "TAKEOFF"
TRACKING   = "TRACKING"

# How long to wait (seconds) for the tracker model to initialise before arming
MODEL_INIT_WAIT = 5.0

# Tuning constants
HOVER_THRUST    = 0.55   # thrust that ~holds altitude (tune per platform)
CLIMB_THRUST    = 0.70   # thrust used while climbing during takeoff
TRACK_THRUST    = 0.65   # thrust during tracking
TAKEOFF_DURATION = 4.0   # seconds to climb before switching to TRACKING


def quaternion_from_euler(roll_rad, pitch_rad, yaw_rad):
    """ZYX convention → quaternion [x, y, z, w]."""
    cy = math.cos(yaw_rad   * 0.5)
    sy = math.sin(yaw_rad   * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad  * 0.5)
    sr = math.sin(roll_rad  * 0.5)

    return [
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    ]


class PIDController:
    """
    Simple discrete PID that outputs a *rate* command (rad/s).

    Args:
        kp, ki, kd : PID gains
        max_rate   : output clamp ±  (rad/s)
        max_integral: anti-windup clamp on the integral term
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 max_rate: float = 0.5, max_integral: float = 0.3):
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
        self.max_rate     = max_rate
        self.max_integral = max_integral

        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = None          # seconds (float), set on first call

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def compute(self, error: float, now: float) -> float:
        """
        Compute PID output given current error and timestamp.

        Parameters
        ----------
        error : signed error in radians  (setpoint − measurement)
        now   : current time in seconds  (monotonic)

        Returns
        -------
        rate command in rad/s, clamped to ±max_rate
        """
        if self._prev_time is None:
            dt = 0.0
        else:
            dt = now - self._prev_time
            dt = max(dt, 1e-6)           # guard against zero / negative dt

        # Proportional
        p_term = self.kp * error

        # Integral  (with anti-windup clamp)
        self._integral += error * dt
        self._integral  = max(-self.max_integral,
                               min(self.max_integral, self._integral))
        i_term = self.ki * self._integral

        # Derivative  (on measurement change, i.e. −kd * Δerror/dt)
        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0

        self._prev_error = error
        self._prev_time  = now

        output = p_term + i_term + d_term
        return max(-self.max_rate, min(self.max_rate, output))


class AutoCannyTracker(Node):

    def __init__(self):
        super().__init__("auto_canny_tracker")

        # ── image / tracker state ─────────────────────────────────────────────
        self.current_image       = None
        self.width               = 0
        self.height              = 0
        self.track_on            = False
        self.tracker_initialized = False
        self.free_float          = True

        # Angular errors stored in DEGREES for logging / limiting
        self.angle_error_x_deg = 0.0
        self.angle_error_y_deg = 0.0

        self.Track_centerx = None
        self.Track_centery = None

        # Camera FOV
        self.hfov = 90.0
        self.vfov = 58.1

        # ── IMU state (radians, updated by imu_callback) ──────────────────────
        self.current_roll  = 0.0
        self.current_pitch = 0.0
        self.current_yaw   = 0.0

        # ── PID controllers for roll-rate and pitch-rate ───────────────────────
        # Gains are in rad/s per radian of pixel error.
        # Tune kp / ki / kd and max_rate for your platform.
        self.pid_roll  = PIDController(kp=1.4, ki=0.08, kd=0.11,
                                       max_rate=0.5, max_integral=0.3)
        self.pid_pitch = PIDController(kp=1.4, ki=0.08, kd=0.11,
                                       max_rate=0.5, max_integral=0.3)
        self._last_tracker_time: float | None = None   # for dt computation

        # ── FCU state ─────────────────────────────────────────────────────────
        self.fcu_armed  = False
        self.fcu_mode   = ""
        self._seq_state = MODEL_INIT     # startup: wait for model before arming
        self._takeoff_start_time  = None
        self._model_init_start    = time.monotonic()   # wall-clock start of MODEL_INIT
        self._model_ready         = False              # set True once model loaded
        
        # ── ROS publishers ────────────────────────────────────────────────────
        self.att_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10,
        )

        # ── ROS service clients ───────────────────────────────────────────────
        self.set_mode_cli = self.create_client(SetMode,      '/mavros/set_mode')
        self.arming_cli   = self.create_client(CommandBool,  '/mavros/cmd/arming')

        # ── ROS subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage,
            "/rgb_compressed",
            self.listener_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self.imu_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            State,
            "/mavros/state",
            self.state_callback,
            10,
        )

        # ── Timers ────────────────────────────────────────────────────────────
        # 10 Hz heartbeat — keeps OFFBOARD stream alive at all times
        self.create_timer(0.15, self.heartbeat_cb)

        # 3.3 Hz tracker update
        self.create_timer(0.15, self.tracker_ang)

        # 2 Hz startup sequencer (arm → OFFBOARD → takeoff)
        self.create_timer(0.5, self.startup_sequencer)

        self.get_logger().info(
            f"AutoCannyTracker node started — waiting {MODEL_INIT_WAIT:.0f}s for model init..."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FCU state
    # ══════════════════════════════════════════════════════════════════════════

    def state_callback(self, msg: State):
        self.fcu_armed = msg.armed
        self.fcu_mode  = msg.mode

    # ══════════════════════════════════════════════════════════════════════════
    # Startup sequencer  (IDLE → ARMING → TAKEOFF → TRACKING)
    # ══════════════════════════════════════════════════════════════════════════

    def startup_sequencer(self):
        """
        Called at 2 Hz.  Drives the model-init → arm → OFFBOARD → takeoff state machine.
        """
        if self._seq_state == MODEL_INIT:
            elapsed = time.monotonic() - self._model_init_start
            if not self._model_ready:
                # Load the model once (blocks briefly — called in timer context,
                # not in __init__, so ROS is already spinning).
                self.get_logger().info(
                    f"[SEQ] MODEL_INIT: loading tracker model... ({elapsed:.1f}s)"
                )
                enable_tracking_and_load_model()
                self._model_ready = True
                self.get_logger().info("[SEQ] MODEL_INIT: model loaded.")

            if elapsed >= MODEL_INIT_WAIT:
                self.get_logger().info(
                    f"[SEQ] MODEL_INIT → IDLE: model ready after {elapsed:.1f}s, "
                    "proceeding to arm sequence"
                )
                self._seq_state = IDLE
            return

        if self._seq_state == IDLE:
            # Pre-condition: setpoint stream must be running (heartbeat handles
            # that from t=0).  Just wait one cycle then try to go OFFBOARD+arm.
            self._seq_state = ARMING
            self.get_logger().info("[SEQ] IDLE → ARMING: requesting OFFBOARD + arm")
            self._send_offboard_request()
            self._send_arm_request(True)

        elif self._seq_state == ARMING:
            if self.fcu_armed and self.fcu_mode == "OFFBOARD":
                self.get_logger().info(
                    "[SEQ] ARMING → TAKEOFF: drone armed in OFFBOARD, climbing..."
                )
                self._seq_state = TAKEOFF
                self._takeoff_start_time = self.get_clock().now()
            else:
                # Keep hammering until FCU accepts
                if not self.fcu_armed:
                    self._send_arm_request(True)
                if self.fcu_mode != "OFFBOARD":
                    self._send_offboard_request()

        elif self._seq_state == TAKEOFF:
            elapsed = (
                self.get_clock().now() - self._takeoff_start_time
            ).nanoseconds * 1e-9

            if elapsed >= TAKEOFF_DURATION:
                self.get_logger().info(
                    f"[SEQ] TAKEOFF → TRACKING: {elapsed:.1f}s elapsed"
                )
                self._seq_state = TRACKING
                self.track_on   = True

        # TRACKING — nothing to do in sequencer

    def _send_offboard_request(self):
        if not self.set_mode_cli.service_is_ready():
            self.get_logger().warn("[SEQ] SetMode service not ready")
            return
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.set_mode_cli.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f"[SEQ] SetMode response: mode_sent={f.result().mode_sent}"
            ) if f.result() else None
        )

    def _send_arm_request(self, arm: bool):
        if not self.arming_cli.service_is_ready():
            self.get_logger().warn("[SEQ] Arming service not ready")
            return
        req = CommandBool.Request()
        req.value = arm
        future = self.arming_cli.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f"[SEQ] Arm response: success={f.result().success}"
            ) if f.result() else None
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Heartbeat  — always streams at 10 Hz so OFFBOARD never times out
    # ══════════════════════════════════════════════════════════════════════════

    def heartbeat_cb(self):
        if self._seq_state == MODEL_INIT:
            # Stream a neutral attitude hold so OFFBOARD pre-conditions are met
            # once we eventually switch — no arming yet.
            self._publish_attitude(
                roll_rad  = 0.0,
                pitch_rad = 0.0,
                yaw_rad   = self.current_yaw,
                thrust    = HOVER_THRUST,
            )
            return

        if self._seq_state == TAKEOFF:
            # During takeoff: level attitude, climb thrust
            self._publish_attitude(
                roll_rad  = 0.0,
                pitch_rad = 0.0,
                yaw_rad   = self.current_yaw,
                thrust    = CLIMB_THRUST,
            )
        elif self._seq_state == TRACKING:
            # tracker_ang owns the setpoint during TRACKING — heartbeat just
            # keeps OFFBOARD alive with zero rates so it doesn't stomp the PID
            self._publish_body_rates(
                roll_rate  = 0.0,
                pitch_rate = 0.0,
                yaw_rate   = 0.0,
                thrust     = TRACK_THRUST,
            )
        else:
            # IDLE / ARMING: hold current attitude
            self._publish_attitude(
                roll_rad  = self.current_roll,
                pitch_rad = self.current_pitch,
                yaw_rad   = self.current_yaw,
                thrust     = HOVER_THRUST,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Tracker update
    # ══════════════════════════════════════════════════════════════════════════

    def tracker_ang(self):
        """
        Runs at ~3.3 Hz.
        Converts pixel error → angular error (rad) → PID → body rate command
        → publishes AttitudeTarget with roll_rate / pitch_rate.
        """
        if self._seq_state != TRACKING:
            return

        bbox = get_current_bbox()

        if self.current_image is None:
            return

        # Monotonic timestamp for PID dt
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.track_on and bbox is not None:

            self.free_float = False
            x, y, w, h = bbox

            self.Track_centerx = x + w / 2.0
            self.Track_centery = y + h / 2.0

            target_x = self.width  / 2.0
            target_y = self.height / 2.0

            error_x = self.Track_centerx - target_x   # pixels (+ve = target right)
            error_y = self.Track_centery - target_y   # pixels (+ve = target below)

            # ── pixel error → angular error (radians) ────────────────────────
            # Negate x so a rightward pixel error commands positive roll-rate
            err_x_rad = radians(self.hfov * error_x * -1.0 / self.width)
            err_y_rad = radians(self.vfov * error_y          / self.height)

            self.angle_error_x_deg = degrees(err_x_rad)
            self.angle_error_y_deg = degrees(err_y_rad)

            # ── PID → body rate commands (rad/s) ─────────────────────────────
            roll_rate  = self.pid_roll.compute(err_x_rad,  now)
            pitch_rate = self.pid_pitch.compute(err_y_rad, now)

            self.get_logger().info(
                f"bbox=({int(x)},{int(y)},{int(w)},{int(h)}) "
                f"err_x={error_x:.1f}px err_y={error_y:.1f}px  "
                f"err_x={degrees(err_x_rad):.2f}° err_y={degrees(err_y_rad):.2f}°  "
                f"→ roll_rate={degrees(roll_rate):.2f}°/s  "
                f"pitch_rate={degrees(pitch_rate):.2f}°/s"
            )

            self._publish_body_rates(
                roll_rate  = roll_rate,
                pitch_rate = pitch_rate,
                yaw_rate   = 0.0,
                thrust     = TRACK_THRUST,
            )

        else:
            self.free_float        = True
            self.angle_error_x_deg = 0.0
            self.angle_error_y_deg = 0.0
            # Reset PID state so there's no windup when target re-appears
            self.pid_roll.reset()
            self.pid_pitch.reset()
            # heartbeat_cb already publishing hold — nothing extra needed

    # ══════════════════════════════════════════════════════════════════════════
    # Attitude publisher helper
    # ══════════════════════════════════════════════════════════════════════════

    def _publish_attitude(self, roll_rad, pitch_rad, yaw_rad, thrust):
        msg = AttitudeTarget()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE  |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )
        q = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust = float(thrust)
        self.att_pub.publish(msg)

    def _publish_body_rates(self, roll_rate: float, pitch_rate: float,
                            yaw_rate: float, thrust: float):
        """
        Publish an AttitudeTarget message using body-rate control.

        The FCU ignores the orientation quaternion and executes the requested
        angular rates directly.  This is the correct interface when a PID loop
        running on-board (here) is responsible for driving the error to zero.

        type_mask bits set  →  IGNORE_ATTITUDE (0x80) | IGNORE_YAW_RATE* (optional)
        *yaw_rate = 0 so the drone holds its current heading.
        """
        msg = AttitudeTarget()
        # Tell the FCU to use rates, ignore the attitude quaternion
        msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE   # 0x80
        msg.body_rate.x = float(roll_rate)
        msg.body_rate.y = float(pitch_rate)
        msg.body_rate.z = float(yaw_rate)
        msg.thrust      = float(thrust)
        self.att_pub.publish(msg)

    # ══════════════════════════════════════════════════════════════════════════
    # IMU callback
    # ══════════════════════════════════════════════════════════════════════════

    def imu_callback(self, msg: Imu):
        qx, qy, qz, qw = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        # Roll
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.current_roll = atan2(sinr, cosr)

        # Pitch
        sinp = 2.0 * (qw * qy - qz * qx)
        self.current_pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))

        # Yaw
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.current_yaw = atan2(siny, cosy)

    # ══════════════════════════════════════════════════════════════════════════
    # Vision callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def detect_largest_contour_center(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy, largest, edges

    def listener_callback(self, msg: CompressedImage):
        try:
            np_arr   = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            self.current_image = cv_image.copy()
            self.height = self.current_image.shape[0]
            self.width  = self.current_image.shape[1]

            push_frame(self.current_image)

            detection = self.detect_largest_contour_center(self.current_image)
            if detection is not None and self._model_ready:
                cx, cy, _, _ = detection
                if not self.tracker_initialized:
                    self.get_logger().info(
                        f"[TRACKER] Initializing at ({cx}, {cy})"
                    )
                    
                    set_tracking_point(int(cx), int(cy))
                    self.tracker_initialized = True
                    # track_on is set by startup_sequencer when TRACKING state is entered

        except Exception as e:
            self.get_logger().error(f"listener_callback error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main2(args=None):
    rclpy.init(args=args)
    node     = AutoCannyTracker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    tracker_thread = threading.Thread(target=run_pysot_tracker, daemon=True)
    tracker_thread.start()

    main_thread = threading.Thread(target=main2, daemon=True)
    main_thread.start()

    tracker_thread.join()
    main_thread.join()
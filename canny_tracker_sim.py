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

IDLE     = "IDLE"
ARMING   = "ARMING"
TAKEOFF  = "TAKEOFF"
TRACKING = "TRACKING"

# Tuning constants
HOVER_THRUST    = 0.55   # thrust that ~holds altitude (tune per platform)
CLIMB_THRUST    = 0.70   # thrust used while climbing during takeoff
TRACK_THRUST    = 0.65   # thrust during tracking
TAKEOFF_DURATION = 4.0   # seconds to climb before switching to TRACKING

# Angular correction limits (degrees) — prevents huge attitude commands
MAX_CORRECTION_DEG = 20.0


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

        # ── FCU state ─────────────────────────────────────────────────────────
        self.fcu_armed  = False
        self.fcu_mode   = ""
        self._seq_state = IDLE           # startup state machine
        self._takeoff_start_time = None

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
        self.create_timer(0.1, self.heartbeat_cb)

        # 3.3 Hz tracker update
        self.create_timer(0.15, self.tracker_ang)

        # 2 Hz startup sequencer (arm → OFFBOARD → takeoff)
        self.create_timer(0.5, self.startup_sequencer)

        self.get_logger().info("AutoCannyTracker node started — waiting for FCU...")

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
        Called at 2 Hz.  Drives the arm → OFFBOARD → takeoff state machine.
        """
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
        if self._seq_state == TAKEOFF:
            # During takeoff: level attitude, climb thrust
            self._publish_attitude(
                roll_rad  = 0.0,
                pitch_rad = 0.0,
                yaw_rad   = self.current_yaw,
                thrust    = CLIMB_THRUST,
            )
        else:
            # IDLE / ARMING / TRACKING: hold current attitude
            self._publish_attitude(
                roll_rad  = self.current_roll,
                pitch_rad = self.current_pitch,
                yaw_rad   = self.current_yaw,
                thrust    = HOVER_THRUST,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Tracker update
    # ══════════════════════════════════════════════════════════════════════════

    def tracker_ang(self):
        """
        Runs at ~3.3 Hz.
        Converts pixel error → degrees → clamps → adds to current IMU attitude
        → publishes AttitudeTarget.
        """
        if self._seq_state != TRACKING:
            return

        bbox = get_current_bbox()

        if self.current_image is None:
            return

        if self.track_on and bbox is not None:

            self.free_float = False
            x, y, w, h = bbox

            self.Track_centerx = x + w / 2.0
            self.Track_centery = y + h / 2.0

            target_x = self.width  / 2.0
            target_y = self.height / 2.0

            error_x = self.Track_centerx - target_x   # pixels
            error_y = self.Track_centery - target_y

            # ── pixel error → degrees ────────────────────────────────────────
            # Negative sign on x so positive pixel error → positive roll
            err_x_deg = self.hfov * error_x * -1.0 / self.width
            err_y_deg = self.vfov * error_y          / self.height

            # ── clamp to safe limits ─────────────────────────────────────────
            err_x_deg = max(-MAX_CORRECTION_DEG, min(MAX_CORRECTION_DEG, err_x_deg))
            err_y_deg = max(-MAX_CORRECTION_DEG, min(MAX_CORRECTION_DEG, err_y_deg))

            self.angle_error_x_deg = err_x_deg
            self.angle_error_y_deg = err_y_deg

            # ── desired attitude = current IMU + correction (convert to rad) ─
            desired_roll  = self.current_roll  + radians(err_x_deg)
            desired_pitch = self.current_pitch + radians(err_y_deg)

            self.get_logger().info(
                f"bbox=({int(x)},{int(y)},{int(w)},{int(h)}) "
                f"err_x={error_x:.1f}px err_y={error_y:.1f}px  "
                f"corr_roll={err_x_deg:.2f}° corr_pitch={err_y_deg:.2f}°  "
                f"→ roll={degrees(desired_roll):.2f}° pitch={degrees(desired_pitch):.2f}°"
            )

            self._publish_attitude(
                roll_rad  = desired_roll,
                pitch_rad = desired_pitch,
                yaw_rad   = self.current_yaw,
                thrust    = TRACK_THRUST,
            )

        else:
            self.free_float        = True
            self.angle_error_x_deg = 0.0
            self.angle_error_y_deg = 0.0
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
            if detection is not None:
                cx, cy, _, _ = detection
                if not self.tracker_initialized:
                    self.get_logger().info(
                        f"[TRACKER] Initializing at ({cx}, {cy})"
                    )
                    enable_tracking_and_load_model()
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
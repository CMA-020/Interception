import math
import threading
from math import atan2, radians, degrees

import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import sys
# sys.path.append('../')
# sys.path.append('/home/paar-core0/python_tracker/tools/python_tracking/')
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Float32MultiArray
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from vs_engine_interfaces.msg import SOTResult   # /VSE/tracker_result message type

# from jetson_pysot_tracker import (
#     run_pysot_tracker,
#     push_frame,
# )

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────
HOVER_THRUST = 0.1   # neutral hold thrust (tune per platform)
TRACK_THRUST = 0.1   # thrust while actively tracking


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

        self.counter = 0
        self.offboard_requested = False
        self.arm_requested = False
        self.current_state = State()


        # ── image / tracker state ─────────────────────────────────────────────
        self.current_image       = None
        self.width               = 0
        self.height              = 0
        self.tracker_initialized = False   # True once bbox published to /track_coords
        self.model_ready         = False   # True once load-model sentinel has been sent

        # Angular errors in RADIANS — written by tracking_status_callback,
        # consumed by tracker_ang PID loop
        self.angle_error_x = 0.0
        self.angle_error_y = 0.0

        # Angular errors in DEGREES — for logging only
        self.angle_error_x_deg = 0.0
        self.angle_error_y_deg = 0.0

        # Camera FOV
        self.hfov = 42.0
        self.vfov =23.0 

        # Latest data from /VSE/tracker_result
        self.latest_tracker_data = {
            "x":               0.0,
            "y":               0.0,
            "width":           0.0,
            "height":          0.0,
            "occlusion_status": False,
            "x_pred":          0.0,
            "y_pred":          0.0,
            "width_pred":      0.0,
            "height_pred":     0.0,
            "is_tracking":     False,
            "last_update":     None,
        }

        # ── IMU state ────────────────────────────────────────────────────────
        self.current_roll  = 0.0
        self.current_pitch = 0.0
        self.current_yaw   = 0.0

        # ── PID controllers ───────────────────────────────────────────────────
        self.pid_roll  = PIDController(kp=1.4, ki=0.08, kd=0.11,
                                       max_rate=0.5, max_integral=0.3)
        self.pid_pitch = PIDController(kp=1.4, ki=0.08, kd=0.11,
                                       max_rate=0.5, max_integral=0.3)

        # ── ROS publishers ────────────────────────────────────────────────────
        self.att_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10,
        )
        self.track_coords_publisher = self.create_publisher(
            Float32MultiArray,
            '/track_coords',
            10,
        )
        self.track_coords_msg = Float32MultiArray()

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.arm_client.wait_for_service()
        self.mode_client.wait_for_service()


        # ── ROS subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage,
            "/VSE/vse_frames/compressed",
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
            self.state_cb,
            10,
        )

        self.tracking_sub = self.create_subscription(
            SOTResult,
            "/VSE/tracker_result",
            self.tracking_status_callback,
            10,
        )

        # ── Heartbeat timer — keeps OFFBOARD stream alive ─────────────────────
        # Publishes a neutral attitude hold at 10 Hz.  The FCU requires a
        # continuous setpoint stream to stay in OFFBOARD mode; once the tracker
        # fires, tracking_status_callback overwrites this with rate commands.
        # self.create_timer(0.1, self.heartbeat_cb)
        self.create_timer(0.05, self.offboard_manager_cb)

        # ── Send load-model sentinel immediately ──────────────────────────────
        # Use a one-shot timer so the publisher has time to be discovered before
        # the first message is sent (avoids dropped first publish).
        self.create_timer(0.5, self._send_model_init_sentinel)

        self.get_logger().info("AutoCannyTracker started — sending model-init sentinel...")

    # ══════════════════════════════════════════════════════════════════════════
    # Model-init sentinel (fires once, ~0.5 s after node start)
    # ══════════════════════════════════════════════════════════════════════════

    def _send_model_init_sentinel(self):
        """One-shot: publish (-20,-20,-20,-20) to tell the tracker to load its model."""
        if self.model_ready:
            return   # already sent — timer keeps firing but we no-op after first call
        self.publish_track_coords(-20.0, -20.0, -20.0, -20.0)
        self.model_ready = True
        self.get_logger().info("[INIT] Load-model sentinel sent on /track_coords.")


    def state_cb(self, msg):
        self.current_state = msg

    def offboard_manager_cb(self):

        # continuously stream setpoints first
        if self.counter < 50:
            self.counter += 1
            return

        if self.current_state.mode != "OFFBOARD" and not self.offboard_requested:
            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"
            self.mode_client.call_async(req)
            self.offboard_requested = True
            self.get_logger().warn("REQUESTED OFFBOARD")
            return

        if not self.current_state.armed and not self.arm_requested:
            req = CommandBool.Request()
            req.value = True
            self.arm_client.call_async(req)
            self.arm_requested = True
            self.get_logger().warn("REQUESTED ARM")
            return


    # ══════════════════════════════════════════════════════════════════════════
    # Heartbeat — keeps OFFBOARD stream alive
    # ══════════════════════════════════════════════════════════════════════════

    def heartbeat_cb(self):
        """
        Publishes a neutral attitude-hold at 10 Hz so the FCU stays in OFFBOARD.
        tracking_status_callback overwrites the setpoint with PID rate commands
        whenever the tracker is active; this just fills the gaps.
        """
        self._publish_attitude(
            roll_rad  = 0.0,
            pitch_rad = 0.0,
            yaw_rad   = self.current_yaw,
            thrust    = HOVER_THRUST,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tracker result callback — computes angular errors from SOTResult
    # ══════════════════════════════════════════════════════════════════════════

    def tracking_status_callback(self, msg: SOTResult):
        """
        Fires on every /VSE/tracker_result message.
        Stores tracker fields, computes angular errors, then runs PID immediately.
        No stage gating — as soon as the tracker publishes, we output commands.
        """
        # ── store raw tracker fields ──────────────────────────────────────────
        self.latest_tracker_data["x"]                = msg.x
        self.latest_tracker_data["y"]                = msg.y
        self.latest_tracker_data["width"]             = msg.width
        self.latest_tracker_data["height"]            = msg.height
        self.latest_tracker_data["occlusion_status"]  = msg.occlusion_status
        self.latest_tracker_data["x_pred"]            = msg.x_pred
        self.latest_tracker_data["y_pred"]            = msg.y_pred
        self.latest_tracker_data["width_pred"]        = msg.width_pred
        self.latest_tracker_data["height_pred"]       = msg.height_pred
        # is_tracking == False → target visible (mirrors C++ convention)
        self.latest_tracker_data["is_tracking"]       = msg.occlusion_status
        self.latest_tracker_data["last_update"]       = self.get_clock().now()

        # ── compute angular errors when target is visible ─────────────────────
        if not self.latest_tracker_data["is_tracking"]:
            track_center_x = msg.x + msg.width  / 2.0
            track_center_y = msg.y + msg.height / 2.0

            error_x = track_center_x - (self.width  / 2.0)
            error_y = track_center_y - (self.height / 2.0)

            self.angle_error_x = radians(self.hfov * error_x * -1.0 / self.width)
            self.angle_error_y = radians(self.vfov * error_y          / self.height)

            self.angle_error_x_deg = degrees(self.angle_error_x)
            self.angle_error_y_deg = degrees(self.angle_error_y)
        else:
            # Target occluded — zero errors, PID resets in tracker_ang
            self.angle_error_x     = 0.0
            self.angle_error_y     = 0.0
            self.angle_error_x_deg = 0.0
            self.angle_error_y_deg = 0.0

        self.tracker_ang()

    # ══════════════════════════════════════════════════════════════════════════
    # Tracker PID — driven directly by tracking_status_callback
    # ══════════════════════════════════════════════════════════════════════════

    def tracker_ang(self):
        """
        Reads pre-computed angular errors, runs PIDs, publishes body-rate commands.
        Called from tracking_status_callback — no timer, no stage gate.
        """
        if self.current_image is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        is_occluded = self.latest_tracker_data["is_tracking"]

        if not is_occluded:
            # Target visible — compute and publish PID rate commands
            roll_rate  = self.pid_roll.compute(self.angle_error_x, now)
            pitch_rate = self.pid_pitch.compute(self.angle_error_y, now)

            self.get_logger().info(
                f"err_x={self.angle_error_x_deg:.2f}°  "
                f"err_y={self.angle_error_y_deg:.2f}°  "
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
            # Target lost / occluded — reset PID, heartbeat holds attitude
            self.pid_roll.reset()
            self.pid_pitch.reset()

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
    # Track coords publisher
    # ══════════════════════════════════════════════════════════════════════════

    def publish_track_coords(self, x: float, y: float, w: float, h: float):
        """
        Publish [x, y, w, h] to /track_coords.

        Special sentinel: (-20, -20, -20, -20) signals the tracker process
        to load its model.  Any other valid bbox triggers tracker init.
        """
        self.track_coords_msg.data = [x, y, w, h]
        self.track_coords_publisher.publish(self.track_coords_msg)
        self.get_logger().info(
            f'Published /track_coords: [{x}, {y}, {w}, {h}]'
        )

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

            

            # Once the model sentinel has been sent, init the tracker on the
            # first Canny detection by publishing the contour bbox.
            if self.model_ready and not self.tracker_initialized:
                detection = self.detect_largest_contour_center(self.current_image)
                if detection is not None:
                    cx, cy, largest, _ = detection
                    bx, by, bw, bh = cv2.boundingRect(largest)
                    self.get_logger().info(
                        f"[TRACKER] Initializing at ({cx}, {cy}) "
                        f"bbox=({bx},{by},{bw},{bh})"
                    )
                    self.publish_track_coords(float(bx), float(by),
                                             float(bw), float(bh))
                    self.tracker_initialized = True

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
    # tracker_thread = threading.Thread(target=run_pysot_tracker, daemon=True)
    # tracker_thread.start()

    main_thread = threading.Thread(target=main2, daemon=True)
    main_thread.start()

    # tracker_thread.join()
    main_thread.join()
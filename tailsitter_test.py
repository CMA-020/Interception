#!/usr/bin/env python3
"""
Tailsitter lift test — MOTOR-ONLY control (no control surfaces).

Strategy
--------
The airframe stays in MULTICOPTER mode the entire flight.
PX4's MC attitude controller uses differential thrust to track
attitude setpoints throughout — including during winged cruise.

Phase sequence
--------------
CLIMB      : vertical ascent, 0 deg pitch, thrust 0.6
TRANSITION : gradually pitch nose forward over N seconds
CRUISE     : ~70 deg nose-forward; motors thrust horizontally,
             wings generate lift passively.  MC mixer stays active.

NOTE: MAV_CMD_DO_VTOL_TRANSITION is intentionally NOT sent.
Switching to FW mode on a no-surface airframe hands control to
the FW mixer which expects elevon/rudder outputs that don't exist.
"""

import math

import rclpy
from rclpy.node import Node

from mavros_msgs.msg import AttitudeTarget, Altitude, State
from mavros_msgs.srv  import CommandBool, SetMode
from sensor_msgs.msg  import Imu

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ==========================================================
# FLIGHT PHASES
# ==========================================================

PHASE_CLIMB       = "CLIMB"       # vertical, motors pointing up
PHASE_TRANSITION  = "TRANSITION"  # ramp pitch forward over time
PHASE_CRUISE      = "CRUISE"      # motors horizontal, wings lifting


# ==========================================================
# QUATERNION  (ZYX / roll-pitch-yaw convention)
# ==========================================================

def quaternion_from_euler(roll, pitch, yaw):

    cy = math.cos(yaw   * 0.5);  sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5);  sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5);  sr = math.sin(roll  * 0.5)

    return [
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    ]


def euler_from_quaternion(x, y, z, w):
    """
    Returns (roll_deg, pitch_deg, yaw_deg) from a unit quaternion.
    Angles follow the ZYX / aerospace convention (same as PX4 / MAVROS).
    """

    # roll  (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll_rad  = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)  — clamp for numerical safety at ±90 deg
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch_rad = math.asin(sinp)

    # yaw   (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad   = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll_rad),
        math.degrees(pitch_rad),
        math.degrees(yaw_rad),
    )


# ==========================================================
# NODE
# ==========================================================

class TailsitterLiftTest(Node):

    def __init__(self):

        super().__init__("tailsitter_lift_test")

        # --------------------------------------------------
        # FCU STATE
        # --------------------------------------------------

        self.current_state = State()
        self.current_yaw   = 0.0
        self.current_alt_m = 0.0

        # Euler angles extracted from IMU for logging
        self.imu_roll_deg  = 0.0
        self.imu_pitch_deg = 0.0
        self.imu_yaw_deg   = 0.0

        # Raw quaternion from IMU for logging
        self.imu_qx = 0.0
        self.imu_qy = 0.0
        self.imu_qz = 0.0
        self.imu_qw = 1.0

        # Throttle IMU orientation logs to 5 Hz (every 4th tick at 20 Hz)
        self._imu_log_counter = 0
        self._IMU_LOG_EVERY_N = 4

        # --------------------------------------------------
        # PHASE
        # --------------------------------------------------

        self.phase = PHASE_CLIMB

        # Altitude that triggers transition
        self.climb_target_m = 100.0

        # --------------------------------------------------
        # CLIMB params
        # --------------------------------------------------

        self.climb_thrust = 0.6
        self.climb_pitch  = 0.0    # deg — vertical hover
        self.climb_roll   = 0.0

        # --------------------------------------------------
        # TRANSITION params
        # --------------------------------------------------

        self.transition_duration_s  = 5.0
        self._trans_start_time: float | None = None
        self._trans_start_pitch: float = 0.0

        self.trans_thrust_start = 0.65
        self.trans_thrust_end   = 0.75

        # --------------------------------------------------
        # CRUISE params
        # --------------------------------------------------

        self.cruise_thrust = 0.95
        self.cruise_pitch  = 85.0
        self.cruise_roll   = 0.0

        # --------------------------------------------------
        # PUB / SUBS / SERVICES / TIMER
        # --------------------------------------------------

        self.att_pub = self.create_publisher(
            AttitudeTarget,
            "/mavros/setpoint_raw/attitude",
            10
        )

        qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 10
        )

        self.create_subscription(State,    "/mavros/state",     self.state_cb, 10)
        self.create_subscription(Imu,      "/mavros/imu/data",  self.imu_cb,   qos)
        self.create_subscription(Altitude, "/mavros/altitude",  self.alt_cb,   qos)

        # FALLBACK altitude source (uncomment if /mavros/altitude stays 0):
        # from std_msgs.msg import Float64
        # self.create_subscription(
        #     Float64, "/mavros/global_position/rel_alt",
        #     lambda m: setattr(self, "current_alt_m", m.data), qos
        # )

        self.arm_client  = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode,     "/mavros/set_mode")

        self.timer = self.create_timer(0.05, self.run)   # 20 Hz

        self.get_logger().info("TAILSITTER LIFT TEST — motor-only control")
        self.get_logger().info(
            f"CLIMB      pitch={self.climb_pitch} deg  "
            f"thrust={self.climb_thrust}  target={self.climb_target_m} m"
        )
        self.get_logger().info(
            f"TRANSITION pitch ramp 0 → {self.cruise_pitch} deg  "
            f"over {self.transition_duration_s} s"
        )
        self.get_logger().info(
            f"CRUISE     pitch={self.cruise_pitch} deg  "
            f"thrust={self.cruise_thrust}  (MC mode, wings lifting)"
        )

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def state_cb(self, msg):
        self.current_state = msg

    # ----------------------------------------------------------

    def imu_cb(self, msg):

        q = msg.orientation

        # Cache raw quaternion
        self.imu_qx = q.x
        self.imu_qy = q.y
        self.imu_qz = q.z
        self.imu_qw = q.w

        # Derive Euler angles
        self.imu_roll_deg, self.imu_pitch_deg, self.imu_yaw_deg = \
            euler_from_quaternion(q.x, q.y, q.z, q.w)

        # Compass yaw (used for attitude setpoint heading-hold)
        yaw_rad = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.current_yaw = (90.0 - math.degrees(yaw_rad)) % 360.0

    # ----------------------------------------------------------

    def alt_cb(self, msg):
        self.current_alt_m = msg.relative

    # ==========================================================
    # SEND ATTITUDE SETPOINT
    # ==========================================================

    def send_attitude(self, roll_deg, pitch_deg, compass_yaw_deg, thrust):

        msg = AttitudeTarget()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE  |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        # Compass (NED, 0 = North, CW) → PX4 ENU yaw
        px4_yaw = 90.0 - compass_yaw_deg
        while px4_yaw >  180.0: px4_yaw -= 360.0
        while px4_yaw < -180.0: px4_yaw += 360.0

        q = quaternion_from_euler(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(px4_yaw)
        )

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust = thrust

        self.att_pub.publish(msg)

    # ==========================================================
    # PHASE MANAGER
    # ==========================================================

    def update_phase(self):

        now = self.get_clock().now().nanoseconds * 1e-9

        # ---- CLIMB → TRANSITION -----------------------------------------
        if (
            self.phase == PHASE_CLIMB and
            self.current_alt_m >= self.climb_target_m
        ):
            self.phase = PHASE_TRANSITION
            self._trans_start_time  = now
            self._trans_start_pitch = self.climb_pitch

            self.get_logger().info(
                f"Alt {self.current_alt_m:.1f} m — starting pitch-over transition "
                f"({self.transition_duration_s} s, "
                f"target pitch = {self.cruise_pitch} deg)"
            )

        # ---- TRANSITION → CRUISE ----------------------------------------
        elif (
            self.phase == PHASE_TRANSITION and
            self._trans_start_time is not None
        ):
            elapsed = now - self._trans_start_time

            if elapsed >= self.transition_duration_s:
                self.phase = PHASE_CRUISE

                self.get_logger().info(
                    "Pitch-over complete — CRUISE  "
                    f"(pitch={self.cruise_pitch} deg, "
                    f"thrust={self.cruise_thrust}, "
                    "MC mode, wings generating lift)"
                )

    # ==========================================================
    # INTERPOLATED TRANSITION SETPOINT
    # ==========================================================

    def transition_setpoint(self):

        now     = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._trans_start_time
        alpha   = min(elapsed / self.transition_duration_s, 1.0)

        pitch  = self._trans_start_pitch + alpha * (self.cruise_pitch - self._trans_start_pitch)
        thrust = self.trans_thrust_start  + alpha * (self.trans_thrust_end - self.trans_thrust_start)

        return pitch, thrust

    # ==========================================================
    # IMU ORIENTATION LOG  (called from run() at 5 Hz)
    # ==========================================================

    def log_imu_orientation(self):

        self.get_logger().info(
            f"[IMU]  "
            f"quat=({self.imu_qx:.4f}, {self.imu_qy:.4f}, "
            f"{self.imu_qz:.4f}, {self.imu_qw:.4f})  |  "
            f"roll={self.imu_roll_deg:+7.2f} deg  "
            f"pitch={self.imu_pitch_deg:+7.2f} deg  "
            f"yaw={self.imu_yaw_deg:+7.2f} deg"
        )

    # ==========================================================
    # MAIN LOOP  (20 Hz)
    # ==========================================================

    def run(self):

        if not self.current_state.connected:
            self.get_logger().info("Waiting for FCU...")
            return

        # --------------------------------------------------
        # RESOLVE SETPOINT
        # --------------------------------------------------

        self.update_phase()

        if self.phase == PHASE_CLIMB:

            roll   = self.climb_roll
            pitch  = self.climb_pitch
            thrust = self.climb_thrust

        elif self.phase == PHASE_TRANSITION:

            pitch, thrust = self.transition_setpoint()
            roll = self.cruise_roll

        else:   # PHASE_CRUISE

            roll   = self.cruise_roll
            pitch  = self.cruise_pitch
            thrust = self.cruise_thrust

        # --------------------------------------------------
        # STREAM SETPOINTS
        # --------------------------------------------------

        self.send_attitude(
            roll_deg        = roll,
            pitch_deg       = pitch,
            compass_yaw_deg = self.current_yaw,
            thrust          = thrust
        )

        # --------------------------------------------------
        # OFFBOARD MODE
        # --------------------------------------------------

        if self.current_state.mode != "OFFBOARD":

            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"
            self.mode_client.call_async(req)
            self.get_logger().info("Trying OFFBOARD...")
            return

        # --------------------------------------------------
        # ARM
        # --------------------------------------------------

        if not self.current_state.armed:

            req = CommandBool.Request()
            req.value = True
            self.arm_client.call_async(req)
            self.get_logger().info("Trying ARM...")
            return

        # --------------------------------------------------
        # IMU ORIENTATION LOG  (5 Hz — every 4th tick)
        # --------------------------------------------------

        self._imu_log_counter += 1

        if self._imu_log_counter >= self._IMU_LOG_EVERY_N:
            self._imu_log_counter = 0
            self.log_imu_orientation()

        # --------------------------------------------------
        # FLIGHT STATUS LOG
        # --------------------------------------------------

        self.get_logger().info(
            f"[{self.phase}]  "
            f"alt={self.current_alt_m:.1f} m  "
            f"pitch_sp={pitch:.1f} deg  "
            f"thrust={thrust:.2f}  "
            f"yaw={self.current_yaw:.1f} deg"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():
    rclpy.init()
    node = TailsitterLiftTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
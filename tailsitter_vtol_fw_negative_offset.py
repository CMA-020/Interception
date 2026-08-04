#!/usr/bin/env python3
"""
Modified from the user's working AttitudeTarget script.

Key changes:
- Keeps AttitudeTarget OFFBOARD control (same mechanism that already flies)
- Adds VTOL transition service
- Keeps tailsitter pitch convention:
    0 deg   = vertical hover
   -85 deg  = nearly horizontal flight
- Transitions:
    CLIMB -> TRANSITION -> FW_TRANSITION -> FW_CRUISE
"""

import math
import rclpy

from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget, Altitude, State
from mavros_msgs.srv import CommandBool, SetMode, CommandVtolTransition
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


PHASE_CLIMB = "CLIMB"
PHASE_TRANSITION = "TRANSITION"
PHASE_FW_TRANSITION = "FW_TRANSITION"
PHASE_FW_CRUISE = "FW_CRUISE"


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


class TailsitterVTOL(Node):

    def __init__(self):

        super().__init__("tailsitter_vtol")

        self.current_state = State()
        self.current_alt_m = 0.0
        self.current_yaw = 0.0

        self.phase = PHASE_CLIMB

        # =========================
        # USER TUNABLE PARAMETERS
        # =========================

        self.climb_target_m = 100.0

        self.climb_pitch = 0.0
        self.climb_thrust = 0.60

        self.transition_duration_s = 5.0

        # For YOUR tailsitter:
        # 0 = hover
        # -85 ~= horizontal
        self.cruise_pitch = 90.0

        self.cruise_thrust = 0.70

        self.fw_transition_wait_s = 0.0

        # Added because you suspect PX4 switches pitch convention
        # after VTOL transition.
        self.fw_pitch_offset_deg = -90.0

        # =========================

        self._trans_start_time = None

        self.vtol_sent = False
        self.fw_transition_start = None

        self.att_pub = self.create_publisher(
            AttitudeTarget,
            "/mavros/setpoint_raw/attitude",
            10
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            State,
            "/mavros/state",
            self.state_cb,
            10
        )

        self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self.imu_cb,
            qos
        )

        self.create_subscription(
            Altitude,
            "/mavros/altitude",
            self.alt_cb,
            qos
        )

        self.arm_client = self.create_client(
            CommandBool,
            "/mavros/cmd/arming"
        )

        self.mode_client = self.create_client(
            SetMode,
            "/mavros/set_mode"
        )

        self.vtol_client = self.create_client(
            CommandVtolTransition,
            "/mavros/cmd/vtol_transition"
        )

        self.timer = self.create_timer(
            0.05,
            self.run
        )

    def state_cb(self, msg):
        self.current_state = msg

    def alt_cb(self, msg):
        self.current_alt_m = msg.relative

    def imu_cb(self, msg):

        q = msg.orientation

        # -----------------------------
        # Roll
        # -----------------------------
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)

        roll_rad = math.atan2(
            sinr_cosp,
            cosr_cosp
        )

        # -----------------------------
        # Pitch
        # -----------------------------
        sinp = 2.0 * (q.w * q.y - q.z * q.x)

        sinp = max(-1.0, min(1.0, sinp))

        pitch_rad = math.asin(sinp)

        # -----------------------------
        # Yaw
        # -----------------------------
        yaw_rad = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # -----------------------------
        # Store Euler angles
        # -----------------------------
        self.current_roll = math.degrees(roll_rad)

        self.current_pitch = math.degrees(pitch_rad)

        self.current_yaw = (
            90.0 - math.degrees(yaw_rad)
        ) % 360.0

    def send_attitude(self, roll_deg, pitch_deg, compass_yaw_deg, thrust):

        msg = AttitudeTarget()

        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        px4_yaw = 90.0 - compass_yaw_deg

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

    def request_fw_transition(self):

        req = CommandVtolTransition.Request()

        req.state = 4  # STATE_FW

        self.vtol_client.call_async(req)

        self.get_logger().info(
            "VTOL TRANSITION COMMAND SENT"
        )

    def run(self):

        if not self.current_state.connected:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # =========================
        # PHASE LOGIC
        # =========================

        if self.phase == PHASE_CLIMB:

            roll = 0.0
            pitch = self.climb_pitch
            thrust = self.climb_thrust

            if self.current_alt_m >= self.climb_target_m:

                self.phase = PHASE_TRANSITION
                self._trans_start_time = now

                self.get_logger().info(
                    "STARTING PITCH TRANSITION"
                )

        elif self.phase == PHASE_TRANSITION:

            alpha = min(
                (now - self._trans_start_time)
                / self.transition_duration_s,
                1.0
            )

            roll = 0.0
            pitch = alpha * self.cruise_pitch
            thrust = 0.65 + alpha * 0.15

            if alpha >= 1.0:

                self.phase = PHASE_FW_TRANSITION

                self.get_logger().info(
                    "REQUESTING FIXED WING MODE"
                )

        elif self.phase == PHASE_FW_TRANSITION:

            roll = 0.0
            pitch = -(self.cruise_pitch + self.fw_pitch_offset_deg)
            thrust = self.cruise_thrust

            if not self.vtol_sent:

                self.request_fw_transition()

                self.vtol_sent = True
                self.fw_transition_start = now

            if (
                now - self.fw_transition_start
                > self.fw_transition_wait_s
            ):

                self.phase = PHASE_FW_CRUISE

                self.get_logger().info(
                    "ENTERING FW CRUISE"
                )

        else:

            roll = 0.0

            # keep same tailsitter orientation
            pitch = -(self.cruise_pitch + self.fw_pitch_offset_deg)

            thrust = self.cruise_thrust

        self.send_attitude(
            roll,
            pitch,
            self.current_yaw,
            thrust
        )

        if self.current_state.mode != "OFFBOARD":

            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"

            self.mode_client.call_async(req)

            return

        if not self.current_state.armed:

            req = CommandBool.Request()
            req.value = True

            self.arm_client.call_async(req)

            return

        self.get_logger().info(
            f"[{self.phase}] "
            f"alt={self.current_alt_m:.1f} "
            f"pitch={self.current_pitch:.1f} "
            f"thrust={thrust:.2f}"
        )


def main():

    rclpy.init()

    node = TailsitterVTOL()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()

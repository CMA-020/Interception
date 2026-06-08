#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from mavros_msgs.msg import (
    AttitudeTarget,
    State
)

from mavros_msgs.srv import (
    CommandBool,
    SetMode
)

from sensor_msgs.msg import Imu

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)


# ==========================================================
# QUATERNION
# ==========================================================

def quaternion_from_euler(
    roll,
    pitch,
    yaw
):

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

        cr * cp * cy + sr * sp * sy
    ]


# ==========================================================
# NODE
# ==========================================================

class FullThrottleClimb(Node):

    def __init__(self):

        super().__init__(
            "full_throttle_climb"
        )

        # ==================================================
        # FCU STATE
        # ==================================================

        self.current_state = State()

        self.current_yaw = 0.0

        # ==================================================
        # PUB
        # ==================================================

        self.pub = self.create_publisher(

            AttitudeTarget,

            "/mavros/setpoint_raw/attitude",

            10
        )

        # ==================================================
        # QoS
        # ==================================================

        qos = QoSProfile(

            reliability=ReliabilityPolicy.BEST_EFFORT,

            history=HistoryPolicy.KEEP_LAST,

            depth=10
        )

        # ==================================================
        # SUBS
        # ==================================================

        self.state_sub = self.create_subscription(

            State,

            "/mavros/state",

            self.state_cb,

            10
        )

        self.imu_sub = self.create_subscription(

            Imu,

            "/mavros/imu/data",

            self.imu_cb,

            qos
        )

        # ==================================================
        # SERVICES
        # ==================================================

        self.arm_client = self.create_client(

            CommandBool,

            "/mavros/cmd/arming"
        )

        self.mode_client = self.create_client(

            SetMode,

            "/mavros/set_mode"
        )

        # ==================================================
        # TIMER
        # ==================================================

        self.timer = self.create_timer(

            0.05,

            self.run
        )

        self.get_logger().info(
            "FULL THROTTLE CLIMB NODE STARTED"
        )

    # ==========================================================
    # STATE
    # ==========================================================

    def state_cb(
        self,
        msg
    ):

        self.current_state = msg

    # ==========================================================
    # IMU
    # ==========================================================

    def imu_cb(
        self,
        msg
    ):

        q = msg.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        yaw_rad = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        yaw_deg_enu = math.degrees(
            yaw_rad
        )

        self.current_yaw = (
            90.0 - yaw_deg_enu
        ) % 360.0

    # ==========================================================
    # SEND ATTITUDE
    # ==========================================================

    def send_attitude(

        self,

        roll_deg,

        pitch_deg,

        compass_yaw_deg,

        thrust
    ):

        msg = AttitudeTarget()

        msg.type_mask = (

            AttitudeTarget.IGNORE_ROLL_RATE |

            AttitudeTarget.IGNORE_PITCH_RATE |

            AttitudeTarget.IGNORE_YAW_RATE
        )

        px4_yaw_deg = (
            90.0 - compass_yaw_deg
        )

        while px4_yaw_deg > 180.0:
            px4_yaw_deg -= 360.0

        while px4_yaw_deg < -180.0:
            px4_yaw_deg += 360.0

        roll = math.radians(
            roll_deg
        )

        pitch = math.radians(
            pitch_deg
        )

        yaw = math.radians(
            px4_yaw_deg
        )

        q = quaternion_from_euler(

            roll,

            pitch,

            yaw
        )

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        msg.thrust = thrust

        self.pub.publish(msg)

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    def run(self):

        if not self.current_state.connected:

            self.get_logger().info(
                "Waiting for FCU..."
            )

            return

        # --------------------------------------------------
        # ALWAYS STREAM SETPOINTS
        # --------------------------------------------------

        self.send_attitude(

            roll_deg=0.0,

            pitch_deg=0.0,

            compass_yaw_deg=self.current_yaw,

            thrust=0.8
        )

        # --------------------------------------------------
        # OFFBOARD
        # --------------------------------------------------

        if self.current_state.mode != "OFFBOARD":

            req = SetMode.Request()

            req.custom_mode = "OFFBOARD"

            self.mode_client.call_async(
                req
            )

            self.get_logger().info(
                "Trying OFFBOARD..."
            )

            return

        # --------------------------------------------------
        # ARM
        # --------------------------------------------------

        if not self.current_state.armed:

            req = CommandBool.Request()

            req.value = True

            self.arm_client.call_async(
                req
            )

            self.get_logger().info(
                "Trying ARM..."
            )

            return

        self.get_logger().info(
            "FULL THRUST CLIMB"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    node = FullThrottleClimb()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()
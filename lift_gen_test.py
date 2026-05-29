import rclpy
from rclpy.node import Node

from mavros_msgs.msg import (
    AttitudeTarget,
    Altitude,
    State
)

from mavros_msgs.srv import (
    CommandBool,
    SetMode
)

from sensor_msgs.msg import Imu

import math

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
# CLASS
# ==========================================================

class VerticalLaunch(Node):

    def __init__(self):

        super().__init__(
            'vertical_launch'
        )

        # ======================================================
        # QOS
        # ======================================================

        sensor_qos = QoSProfile(

            reliability=ReliabilityPolicy.BEST_EFFORT,

            history=HistoryPolicy.KEEP_LAST,

            depth=10
        )

        state_qos = QoSProfile(

            reliability=ReliabilityPolicy.RELIABLE,

            history=HistoryPolicy.KEEP_LAST,

            depth=10
        )

        # ======================================================
        # PUBLISHER
        # ======================================================

        self.pub = self.create_publisher(

            AttitudeTarget,

            '/mavros/setpoint_raw/attitude',

            10
        )

        # ======================================================
        # SUBSCRIBERS
        # ======================================================

        self.alt_sub = self.create_subscription(

            Altitude,

            '/mavros/altitude',

            self.alt_cb,

            sensor_qos
        )

        self.imu_sub = self.create_subscription(

            Imu,

            '/mavros/imu/data',

            self.imu_cb,

            sensor_qos
        )

        self.state_sub = self.create_subscription(

            State,

            '/mavros/state',

            self.state_cb,

            state_qos
        )

        # ======================================================
        # SERVICES
        # ======================================================

        self.arm_client = self.create_client(

            CommandBool,

            '/mavros/cmd/arming'
        )

        self.mode_client = self.create_client(

            SetMode,

            '/mavros/set_mode'
        )

        # ======================================================
        # TIMER
        # ======================================================

        self.timer = self.create_timer(

            0.05,

            self.run
        )

        # ======================================================
        # STATE
        # ======================================================

        self.current_state = State()

        self.current_alt = 0.0

        self.base_altitude = None

        self.current_yaw = 0.0

        self.locked_yaw = None

        self.phase_2 = False

        self.offboard_sent = False

        # ======================================================
        # LOG
        # ======================================================

        self.get_logger().info(
            "Vertical Launch Initialized"
        )

    # ==========================================================
    # STATE CALLBACK
    # ==========================================================

    def state_cb(
        self,
        msg
    ):

        self.current_state = msg

    # ==========================================================
    # ALT CALLBACK
    # ==========================================================

    def alt_cb(
        self,
        msg
    ):

        self.current_alt = msg.amsl

        if self.base_altitude is None:

            self.base_altitude = (
                self.current_alt
            )

            self.get_logger().info(

                f"Base Altitude Locked: "

                f"{self.base_altitude:.2f}"
            )

    # ==========================================================
    # IMU CALLBACK
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

        if self.locked_yaw is None:

            self.locked_yaw = (
                self.current_yaw
            )

            self.get_logger().info(

                f"Yaw Locked: "

                f"{self.locked_yaw:.2f}"
            )

    # ==========================================================
    # SEND ATTITUDE
    # ==========================================================

    def send_attitude(

        self,

        pitch_deg,

        yaw_deg,

        thrust
    ):

        msg = AttitudeTarget()

        msg.type_mask = (

            AttitudeTarget.IGNORE_ROLL_RATE |

            AttitudeTarget.IGNORE_PITCH_RATE |

            AttitudeTarget.IGNORE_YAW_RATE
        )

        # ======================================================
        # COMPASS -> PX4 YAW
        # ======================================================

        px4_yaw_deg = (

            90.0 - yaw_deg
        )

        while px4_yaw_deg > 180.0:

            px4_yaw_deg -= 360.0

        while px4_yaw_deg < -180.0:

            px4_yaw_deg += 360.0

        roll = 0.0

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

        # ======================================================
        # WAIT FOR FCU
        # ======================================================

        if not self.current_state.connected:

            self.get_logger().info(
                "Waiting FCU..."
            )

            return

        # ======================================================
        # WAIT FOR YAW + ALT
        # ======================================================

        if self.locked_yaw is None:

            self.get_logger().info(
                "Waiting IMU..."
            )

            return

        if self.base_altitude is None:

            self.get_logger().info(
                "Waiting Altitude..."
            )

            return

        # ======================================================
        # ALWAYS STREAM SETPOINTS
        # ======================================================

        if not self.phase_2:

            self.send_attitude(

                0.0,

                self.locked_yaw,

                0.5
            )

        else:

            self.send_attitude(

                90.0,

                self.locked_yaw,

                1.0
            )

        # ======================================================
        # GIVE PX4/ARDUPILOT TIME
        # ======================================================

        if not self.offboard_sent:

            self.offboard_sent = True

            self.get_logger().info(
                "Initial setpoints sent"
            )

            return

        # ======================================================
        # OFFBOARD
        # ======================================================

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

        # ======================================================
        # ARM
        # ======================================================

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

        # ======================================================
        # RELATIVE ALTITUDE
        # ======================================================

        relative_alt = (

            self.current_alt -

            self.base_altitude
        )

        # ======================================================
        # PHASE 1
        # ======================================================

        if not self.phase_2:

            if relative_alt < 100.0:

                self.send_attitude(

                    0.0,

                    self.locked_yaw,

                    1.0
                )

                self.get_logger().info(

                    f"ASCENDING | "

                    f"Altitude: "

                    f"{relative_alt:.2f} m"
                )

            else:

                self.phase_2 = True

                self.get_logger().info(
                    "SWITCHING TO +90 PITCH"
                )

        # ======================================================
        # PHASE 2
        # ======================================================

        else:

            self.send_attitude(

                90.0,

                self.locked_yaw,

                1.0
            )

            self.get_logger().info(

                f"PITCH +90 | "

                f"Altitude: "

                f"{relative_alt:.2f} m"
            )


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    node = VerticalLaunch()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
import rclpy
from rclpy.node import Node

from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Imu

import math

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


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
        cr * cp * cy + sr * sp * sy
    ]


class PitchTest(Node):
    def __init__(self):
        super().__init__('pitch_test')

        self.pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )

        # ✅ CORRECT QoS (THIS FIXES YOUR ISSUE)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_cb,
            qos
        )

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.timer = self.create_timer(0.05, self.run)

        self.counter = 0
        self.offboard = False
        self.armed = False

        self.current_pitch = 0.0

    def imu_cb(self, msg):
        q = msg.orientation

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        self.current_pitch = math.degrees(pitch)

    def send_attitude(self):
        msg = AttitudeTarget()

        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        roll = 0.0
        pitch = math.radians(45.0)
        yaw = 0.0

        q = quaternion_from_euler(roll, pitch, yaw)

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        msg.thrust = 0.1

        self.pub.publish(msg)

    def run(self):
        # Prestream
        if self.counter < 50:
            self.send_attitude()
            self.counter += 1
            return

        # OFFBOARD
        if not self.offboard:
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            self.mode_client.call_async(req)
            self.offboard = True
            return

        # ARM
        if not self.armed:
            req = CommandBool.Request()
            req.value = True
            self.arm_client.call_async(req)
            self.armed = True
            return

        self.send_attitude()

        # 🔥 NOW THIS WILL WORK
        self.get_logger().info(
            f"PitchCmd: 45.0 | ActualPitch: {self.current_pitch:.2f}"
        )


def main():
    rclpy.init()
    node = PitchTest()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu

import time
import math

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ParabolicAttack(Node):
    def __init__(self):
        super().__init__('parabolic_attack')

        # Attitude publisher
        self.pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_cb,
            qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_cb,
            qos
        )

        self.current_z = 0.0
        self.got_pose = False
        self.pitch = 0.0

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.timer = self.create_timer(0.05, self.run)

        self.start_time = None
        self.counter = 0
        self.offboard = False
        self.armed = False

        self.T = 20.0

    def pose_cb(self, msg):
        self.current_z = msg.pose.position.z
        self.got_pose = True

    def imu_cb(self, msg):
        q = msg.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        self.pitch = math.degrees(pitch)

    # ✅ Euler → Quaternion
    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0.0] * 4
        q[0] = cr * cp * cy + sr * sp * sy  # w
        q[1] = sr * cp * cy - cr * sp * sy  # x
        q[2] = cr * sp * cy + sr * cp * sy  # y
        q[3] = cr * cp * sy - sr * sp * cy  # z

        return q

    def send_attitude(self, pitch_deg):
        msg = AttitudeTarget()

        # Ignore body rates → use orientation only
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                        AttitudeTarget.IGNORE_PITCH_RATE | \
                        AttitudeTarget.IGNORE_YAW_RATE

        roll = 0.0
        pitch = math.radians(pitch_deg)
        yaw = 0.0

        q = self.euler_to_quaternion(roll, pitch, yaw)

        msg.orientation.w = q[0]
        msg.orientation.x = q[1]
        msg.orientation.y = q[2]
        msg.orientation.z = q[3]

        # Thrust (0–1)
        msg.thrust = 0.7  # tune this

        self.pub.publish(msg)

    def run(self):
        if not self.got_pose:
            self.get_logger().info("Waiting for pose...")
            return

        if self.start_time is None:
            self.start_time = time.time()

        t = time.time() - self.start_time
        s = min(t / self.T, 1.0)

        # ✅ Gradually increase pitch to 45°
        target_pitch = 45.0 * s

        # Prestream
        if self.counter < 40:
            self.send_attitude(0.0)
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

        # Send attitude
        self.send_attitude(target_pitch)

        self.get_logger().info(
            f"s={s:.2f}, target_pitch={target_pitch:.2f}, actual_pitch={self.pitch:.2f}"
        )


def main():
    rclpy.init()
    node = ParabolicAttack()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
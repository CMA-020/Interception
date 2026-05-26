import rclpy
from rclpy.node import Node
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
import time
import math

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ParabolicAttack(Node):
    def __init__(self):
        super().__init__('parabolic_attack')

        self.pub = self.create_publisher(
            PositionTarget,
            '/mavros/setpoint_raw/local',
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
        self.z0 = None

        self.counter = 0
        self.offboard = False
        self.armed = False

        # Trajectory params
        self.X = 400.0
        self.H = 200.0
        self.T = 20.0

    def pose_cb(self, msg):
        self.current_z = msg.pose.position.z
        self.got_pose = True

    def imu_cb(self, msg):
        q = msg.orientation

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        self.pitch = math.degrees(pitch)

    def send_velocity(self, vx, vy, vz):
        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        # ✅ IGNORE POSITION → PURE VELOCITY CONTROL
        msg.type_mask = (
            PositionTarget.IGNORE_PX |
            PositionTarget.IGNORE_PY |
            PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = vz

        msg.yaw = 0.0

        self.pub.publish(msg)

    def run(self):
        if not self.got_pose:
            self.get_logger().info("Waiting for pose...")
            return

        if self.z0 is None:
            self.z0 = self.current_z

        if self.start_time is None:
            self.start_time = time.time()

        t = time.time() - self.start_time
        s = min(t / self.T, 1.0)

        # ✅ Parabolic velocity profile
        vx = self.X / self.T
        vz = (2 * self.H * s) / self.T

        # Prestream
        if self.counter < 40:
            self.send_velocity(0.0, 0.0, 0.0)
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

        # Send velocity only
        self.send_velocity(vx, 0.0, vz)

        slope = vz / vx if vx != 0 else 0.0

        self.get_logger().info(
            f"s={s:.2f}, slope={slope:.2f}, pitch={self.pitch:.2f} deg"
        )


def main():
    rclpy.init()
    node = ParabolicAttack()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
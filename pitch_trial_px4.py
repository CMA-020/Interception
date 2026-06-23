import rclpy
from rclpy.node import Node

from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode

from sensor_msgs.msg import Imu

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

import math
import threading
import sys
import termios
import tty
import time


# ==========================================================
# USER SETTINGS
# ==========================================================

ROLL_DEG = 0.0
PITCH_DEG = 0.0
YAW_DEG = 10.0
THRUST = 0.1

STREAM_RATE_HZ = 20.0

PRESTREAM_COUNT = 50

OFFBOARD_MODE = 'OFFBOARD'
SAFE_MODE = 'MANUAL'


# ==========================================================
# GLOBAL FLAGS
# ==========================================================

stop_requested = False


# ==========================================================
# KEYBOARD LISTENER
# ==========================================================

def keyboard_listener():

    global stop_requested

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:

        tty.setcbreak(fd)

        while True:

            ch = sys.stdin.read(1)

            if ch.lower() == 'd':

                stop_requested = True

                print("\n[D] pressed -> DISARM + SAFE MODE\n")

                break

    finally:

        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            old_settings
        )


# ==========================================================
# QUATERNION FROM EULER
# ==========================================================

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


# ==========================================================
# EULER FROM QUATERNION
# ==========================================================

def euler_from_quaternion(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)

    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)

    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)

    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


# ==========================================================
# NODE
# ==========================================================

class OffboardAttitudeControl(Node):

    def __init__(self):

        super().__init__('offboard_attitude_control')

        # ==================================================
        # PUBLISHER
        # ==================================================

        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )

        # ==================================================
        # QOS PROFILE FOR MAVROS IMU
        # ==================================================

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==================================================
        # IMU SUBSCRIBER
        # ==================================================

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile
        )

        # ==================================================
        # SERVICES
        # ==================================================

        self.arm_client = self.create_client(
            CommandBool,
            '/mavros/cmd/arming'
        )

        self.mode_client = self.create_client(
            SetMode,
            '/mavros/set_mode'
        )

        # ==================================================
        # WAIT FOR SERVICES
        # ==================================================

        self.get_logger().info(
            "Waiting for MAVROS services..."
        )

        self.arm_client.wait_for_service()
        self.mode_client.wait_for_service()

        self.get_logger().info(
            "MAVROS services connected"
        )

        # ==================================================
        # IMU STATE
        # ==================================================

        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0

        # ==================================================
        # STATE
        # ==================================================

        self.counter = 0

        self.offboard_requested = False
        self.arm_requested = False

        self.shutdown_executed = False

        # ==================================================
        # TIMER
        # ==================================================

        self.timer = self.create_timer(
            1.0 / STREAM_RATE_HZ,
            self.run
        )

    # ==========================================================
    # IMU CALLBACK
    # ==========================================================

    def imu_callback(self, msg):

        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        roll, pitch, yaw = euler_from_quaternion(
            qx,
            qy,
            qz,
            qw
        )

        self.current_roll = math.degrees(roll)
        self.current_pitch = math.degrees(pitch)
        self.current_yaw = math.degrees(yaw)

        print(
            f"IMU | "
            f"Roll: {self.current_roll:.2f} | "
            f"Pitch: {self.current_pitch:.2f} | "
            f"Yaw: {self.current_yaw:.2f}"
        )

    # ==========================================================
    # SEND ATTITUDE
    # ==========================================================

    def send_attitude(self):

        msg = AttitudeTarget()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        roll = math.radians(ROLL_DEG)
        pitch = math.radians(PITCH_DEG)
        yaw = math.radians(YAW_DEG)

        q = quaternion_from_euler(
            roll,
            pitch,
            yaw
        )

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        msg.body_rate.x = 0.0
        msg.body_rate.y = 0.0
        msg.body_rate.z = 0.0

        msg.thrust = THRUST

        self.attitude_pub.publish(msg)

    # ==========================================================
    # SAFE SHUTDOWN
    # ==========================================================

    def execute_shutdown(self):

        if self.shutdown_executed:
            return

        self.shutdown_executed = True

        self.get_logger().warn(
            "EXECUTING SAFE SHUTDOWN"
        )

        # ======================================================
        # CHANGE MODE
        # ======================================================

        mode_req = SetMode.Request()

        mode_req.custom_mode = SAFE_MODE

        self.mode_client.call_async(mode_req)

        self.get_logger().warn(
            f"Requested mode: {SAFE_MODE}"
        )

        time.sleep(1.0)

        # ======================================================
        # DISARM
        # ======================================================

        arm_req = CommandBool.Request()

        arm_req.value = False

        self.arm_client.call_async(arm_req)

        self.get_logger().warn(
            "Requested DISARM"
        )

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    def run(self):

        global stop_requested

        # ======================================================
        # D KEY PRESSED
        # ======================================================

        if stop_requested:

            self.execute_shutdown()

            return

        # ======================================================
        # ALWAYS STREAM SETPOINTS
        # ======================================================

        self.send_attitude()

        # ======================================================
        # PRESTREAM
        # ======================================================

        if self.counter < PRESTREAM_COUNT:

            self.counter += 1

            self.get_logger().info(
                f"Prestream {self.counter}/{PRESTREAM_COUNT}"
            )

            return

        # ======================================================
        # SET OFFBOARD MODE
        # ======================================================

        if not self.offboard_requested:

            req = SetMode.Request()

            req.custom_mode = OFFBOARD_MODE

            self.mode_client.call_async(req)

            self.offboard_requested = True

            self.get_logger().info(
                f"Requested mode: {OFFBOARD_MODE}"
            )

            return

        # ======================================================
        # ARM
        # ======================================================

        if not self.arm_requested:

            req = CommandBool.Request()

            req.value = True

            self.arm_client.call_async(req)

            self.arm_requested = True

            self.get_logger().info(
                "Requested ARM"
            )

            return

        # ======================================================
        # STATUS LOGGING
        # ======================================================

        self.get_logger().info(

            f"OFFBOARD ACTIVE | "
            f"Target Roll={ROLL_DEG} | "
            f"Target Pitch={PITCH_DEG} | "
            f"Target Yaw={YAW_DEG} | "
            f"Thrust={THRUST}"

        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    keyboard_thread = threading.Thread(
        target=keyboard_listener,
        daemon=True
    )

    keyboard_thread.start()

    rclpy.init()

    node = OffboardAttitudeControl()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
 
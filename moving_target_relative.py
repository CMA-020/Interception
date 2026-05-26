import rclpy
from rclpy.node import Node

from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import Altitude

from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode

from sensor_msgs.msg import Imu

from geometry_msgs.msg import Quaternion

from tf2_msgs.msg import TFMessage

import math

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy


# ==========================================================
# QUATERNION
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
# NODE
# ==========================================================

class PitchNavigation(Node):

    def __init__(self):

        super().__init__('pitch_navigation')

        # ==================================================
        # ATTITUDE PUBLISHER
        # ==================================================

        self.pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )

        # ==================================================
        # ORIENTATION PUBLISHER
        # ==================================================

        self.orientation_pub = self.create_publisher(
            Quaternion,
            'paar_drone_orientation',
            10
        )

        # ==================================================
        # QOS
        # ==================================================

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==================================================
        # SUBSCRIBERS
        # ==================================================

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_cb,
            qos
        )

        self.alt_sub = self.create_subscription(
            Altitude,
            '/mavros/altitude',
            self.alt_cb,
            qos
        )

        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_cb,
            qos
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
        # TIMER
        # ==================================================

        self.timer = self.create_timer(
            0.05,
            self.run
        )

        # ==================================================
        # STATE
        # ==================================================

        self.counter = 0

        self.offboard = False
        self.armed = False

        self.takeoff_complete = False

        # ==================================================
        # DRONE STATE
        # ==================================================

        self.current_pitch = 0.0
        self.current_yaw = 0.0

        self.current_alt = 0.0

        self.base_altitude = None

        # ==================================================
        # TF ORIGIN
        # ==================================================

        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

        # ==================================================
        # TARGET TRACKING
        # ==================================================

        self.target_history = []

        self.learning_start_time = None

        self.learning_complete = False

        # ==================================================
        # VELOCITY
        # ==================================================

        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0

        # ==================================================
        # LAST REAL TARGET
        # ==================================================

        self.last_target_x = 0.0
        self.last_target_y = 0.0
        self.last_target_z = 0.0

        # ==================================================
        # PROPAGATED TARGET
        # ==================================================

        self.propagated_x = 0.0
        self.propagated_y = 0.0
        self.propagated_z = 0.0

        # ==================================================
        # PROPAGATION CLOCK
        # ==================================================

        self.propagation_time = (
            self.get_clock().now().nanoseconds
            / 1e9
        )

    # ==========================================================
    # IMU CALLBACK
    # ==========================================================

    def imu_cb(self, msg):

        q = msg.orientation

        orientation_msg = Quaternion()

        orientation_msg.x = q.x
        orientation_msg.y = q.y
        orientation_msg.z = q.z
        orientation_msg.w = q.w

        self.orientation_pub.publish(
            orientation_msg
        )

        # ======================================================
        # PITCH
        # ======================================================

        sinp = 2.0 * (
            q.w * q.y -
            q.z * q.x
        )

        pitch = math.asin(
            max(
                -1.0,
                min(1.0, sinp)
            )
        )

        self.current_pitch = math.degrees(
            pitch
        )

        # ======================================================
        # YAW
        # ======================================================

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

        # ======================================================
        # ENU -> COMPASS
        # ======================================================

        self.current_yaw = (
            90.0 - yaw_deg_enu
        ) % 360.0

    # ==========================================================
    # ALTITUDE CALLBACK
    # ==========================================================

    def alt_cb(self, msg):

        self.current_alt = msg.amsl

        if self.base_altitude is None:

            self.base_altitude = self.current_alt

            self.get_logger().info(

                f"Base altitude set: "
                f"{self.base_altitude:.2f}"
            )

    # ==========================================================
    # TF CALLBACK
    # ==========================================================

    def tf_cb(self, msg):

        current_time = (
            self.get_clock().now().nanoseconds
            / 1e9
        )

        for t in msg.transforms:

            # ==================================================
            # DRONE ORIGIN
            # ==================================================

            if t.child_frame_id == "base_link":

                if self.origin_x is None:

                    self.origin_x = (
                        t.transform.translation.x
                    )

                    self.origin_y = (
                        t.transform.translation.y
                    )

                    self.origin_z = (
                        t.transform.translation.z
                    )

                    self.get_logger().info(

                        f"TF Origin Set -> "

                        f"X: {self.origin_x:.2f}, "

                        f"Y: {self.origin_y:.2f}, "

                        f"Z: {self.origin_z:.2f}"
                    )

            # ==================================================
            # MOVING TARGET
            # ==================================================

            if t.child_frame_id == "moving_cube":

                if self.origin_x is None:
                    return

                cube_x = (
                    t.transform.translation.x
                )

                cube_y = (
                    t.transform.translation.y
                )

                cube_z = (
                    t.transform.translation.z
                )

                relative_x = (
                    cube_x -
                    self.origin_x
                )

                relative_y = (
                    cube_y -
                    self.origin_y
                )

                relative_z = (
                    cube_z -
                    self.origin_z
                )

                # ==============================================
                # START LEARNING
                # ==============================================

                if self.learning_start_time is None:

                    self.learning_start_time = (
                        current_time
                    )

                # ==============================================
                # STORE HISTORY
                # ==============================================

                self.target_history.append({

                    'time': current_time,

                    'x': relative_x,

                    'y': relative_y,

                    'z': relative_z
                })

                # ==============================================
                # KEEP LAST 5 SEC
                # ==============================================

                self.target_history = [

                    h for h in self.target_history

                    if current_time - h['time'] <= 5.0
                ]

                # ==============================================
                # LEARN VELOCITY
                # ==============================================

                elapsed = (
                    current_time -
                    self.learning_start_time
                )

                if elapsed >= 5.0:

                    self.learning_complete = True

                    oldest = self.target_history[0]
                    newest = self.target_history[-1]

                    dt = (
                        newest['time'] -
                        oldest['time']
                    )

                    if dt > 0.01:

                        self.velocity_x = (
                            newest['x'] -
                            oldest['x']
                        ) / dt

                        self.velocity_y = (
                            newest['y'] -
                            oldest['y']
                        ) / dt

                        self.velocity_z = (
                            newest['z'] -
                            oldest['z']
                        ) / dt

                # ==============================================
                # STORE REAL TARGET
                # ==============================================

                self.last_target_x = relative_x
                self.last_target_y = relative_y
                self.last_target_z = relative_z

                # ==============================================
                # BEFORE LEARNING
                # ==============================================

                if not self.learning_complete:

                    self.propagated_x = relative_x
                    self.propagated_y = relative_y
                    self.propagated_z = relative_z

    # ==========================================================
    # SEND ATTITUDE
    # ==========================================================

    def send_attitude(
        self,
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

        # ======================================================
        # COMPASS -> PX4 YAW
        # ======================================================

        px4_yaw_deg = (
            90.0 - compass_yaw_deg
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
        # WAIT FOR ALTITUDE INIT
        # ======================================================

        if self.base_altitude is None:
            return

        # ======================================================
        # PRESTREAM
        # ======================================================

        if self.counter < 50:

            self.send_attitude(
                0.0,
                self.current_yaw,
                0.5
            )

            self.counter += 1

            return

        # ======================================================
        # OFFBOARD
        # ======================================================

        if not self.offboard:

            req = SetMode.Request()

            req.custom_mode = 'OFFBOARD'

            self.mode_client.call_async(
                req
            )

            self.offboard = True

            self.get_logger().info(
                "OFFBOARD enabled"
            )

            return

        # ======================================================
        # ARM
        # ======================================================

        if not self.armed:

            req = CommandBool.Request()

            req.value = True

            self.arm_client.call_async(
                req
            )

            self.armed = True

            self.get_logger().info(
                "Vehicle armed"
            )

            return

        # ======================================================
        # TAKEOFF
        # ======================================================

        relative_alt = (
            self.current_alt -
            self.base_altitude
        )

        if not self.takeoff_complete:

            if relative_alt < 10.0:

                self.send_attitude(
                    0.0,
                    self.current_yaw,
                    0.75
                )

                self.get_logger().info(
                    f"Taking off | Relative Alt: "
                    f"{relative_alt:.2f}"
                )

                return

            else:

                self.takeoff_complete = True

                self.get_logger().info(
                    "Takeoff complete"
                )

        # ======================================================
        # CONTINUOUS PROPAGATION
        # ======================================================

        current_time = (
            self.get_clock().now().nanoseconds
            / 1e9
        )

        dt = (
            current_time -
            self.propagation_time
        )

        self.propagation_time = (
            current_time
        )

        if self.learning_complete:

            self.propagated_x += (
                self.velocity_x * dt
            )

            self.propagated_y += (
                self.velocity_y * dt
            )

            self.propagated_z += (
                self.velocity_z * dt
            )

        else:

            self.propagated_x = (
                self.last_target_x
            )

            self.propagated_y = (
                self.last_target_y
            )

            self.propagated_z = (
                self.last_target_z
            )

        # ======================================================
        # TARGET VECTOR
        # ======================================================

        target_x = self.propagated_x
        target_y = self.propagated_y
        target_z = self.propagated_z

        # ======================================================
        # DISTANCES
        # ======================================================

        horizontal_distance = math.sqrt(
            target_x**2 +
            target_y**2
        )

        total_distance = math.sqrt(
            target_x**2 +
            target_y**2 +
            target_z**2
        )

        # ======================================================
        # TARGET YAW
        # ======================================================

        yaw_rad = math.atan2(
            target_y,
            target_x
        )

        target_yaw_deg = (
            math.degrees(yaw_rad)
            + 360.0
        ) % 360.0

        # ======================================================
        # YAW ERROR
        # ======================================================

        yaw_error = (
            target_yaw_deg -
            self.current_yaw
        )

        while yaw_error > 180.0:
            yaw_error -= 360.0

        while yaw_error < -180.0:
            yaw_error += 360.0

        # ======================================================
        # ALIGN YAW FIRST
        # ======================================================

        if abs(yaw_error) > 10.0:

            pitch_deg = 0.0

            thrust = 0.55

        else:

            # ==================================================
            # INTERCEPT PITCH
            # ==================================================

            safe_distance = max(
                horizontal_distance,
                0.01
            )

            pitch_rad = math.atan2(
                safe_distance,
                target_z
            )

            pitch_deg = math.degrees(
                pitch_rad
            )

            # ==================================================
            # PX4 FORWARD PITCH
            # ==================================================

            pitch_deg = -pitch_deg

            # ==================================================
            # LIMIT
            # ==================================================

            pitch_deg = max(
                -45.0,
                min(45.0, pitch_deg)
            )

            # ==================================================
            # THRUST
            # ==================================================

            thrust = 1.0

        # ======================================================
        # SEND COMMAND
        # ======================================================

        self.send_attitude(
            pitch_deg,
            target_yaw_deg,
            thrust
        )

        # ======================================================
        # LOGGING
        # ======================================================

        self.get_logger().info(

            f"Prediction: "
            f"{self.learning_complete} | "

            f"TargetX: {target_x:.2f} | "
            f"TargetY: {target_y:.2f} | "
            f"TargetZ: {target_z:.2f} | "

            f"YawCmd: {target_yaw_deg:.2f} | "

            f"YawError: {yaw_error:.2f} | "

            f"PitchCmd: {pitch_deg:.2f} | "

            f"Distance: {total_distance:.2f} | "

            f"VelX: {self.velocity_x:.2f} | "

            f"VelY: {self.velocity_y:.2f} | "

            f"VelZ: {self.velocity_z:.2f}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    node = PitchNavigation()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
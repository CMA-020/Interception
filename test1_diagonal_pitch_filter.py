import rclpy
from rclpy.node import Node

from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import Altitude

from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode

from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix

from haversine import haversine
from haversine import Unit

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
        # TARGET GPS
        # ==================================================

        # self.target_lat = 40.59277689801748
        # self.target_lon = -79.88906507129052

        # self.target_lat = 40.585125579371095
        # self.target_lon = -79.88772541690355
        
        self.target_lat = 40.59190553758221 
        self.target_lon = -79.8862427481675    ######3rd target for jig 


        # RELATIVE ALTITUDE ABOVE TAKEOFF
        self.target_alt = 750.0

        # ==================================================
        # PUBLISHER
        # ==================================================

        self.pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
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
        # IMU
        # ==================================================

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_cb,
            qos
        )

        # ==================================================
        # GPS
        # ==================================================

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_cb,
            qos
        )

        # ==================================================
        # ALTITUDE
        # ==================================================

        self.alt_sub = self.create_subscription(
            Altitude,
            '/mavros/altitude',
            self.alt_cb,
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
        # CURRENT STATE
        # ==================================================

        self.current_pitch = 0.0

        # COMPASS HEADING
        # 0 = NORTH
        # 90 = EAST

        self.current_yaw = 0.0

        self.current_lat = None
        self.current_lon = None

        # AMSL ALTITUDE

        self.current_alt = 0.0

        self.base_altitude = None

        self.absolute_target_alt = None

    # ==========================================================
    # IMU CALLBACK
    # ==========================================================

    def imu_cb(self, msg):

        q = msg.orientation

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
    # GPS CALLBACK
    # ==========================================================

    def gps_cb(self, msg):

        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    # ==========================================================
    # ALTITUDE CALLBACK
    # ==========================================================

    def alt_cb(self, msg):

        self.current_alt = msg.amsl

        # SAVE TAKEOFF ALTITUDE ONCE

        if self.base_altitude is None:

            self.base_altitude = self.current_alt

            self.absolute_target_alt = (
                self.base_altitude +
                self.target_alt
            )

            self.get_logger().info(

                f"BaseAltitude: "
                f"{self.base_altitude:.2f} m | "

                f"AbsoluteTargetAlt: "
                f"{self.absolute_target_alt:.2f} m"
            )

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
        # WAIT FOR GPS
        # ======================================================

        if self.current_lat is None:
            return

        # ======================================================
        # RELATIVE ALTITUDE
        # ======================================================

        relative_alt = (
            self.current_alt -
            self.base_altitude
        )

        # ======================================================
        # TAKEOFF TO 10m
        # ======================================================

        if not self.takeoff_complete:

            if relative_alt < 10.0:

                self.send_attitude(
                    0.0,
                    self.current_yaw,
                    1.0
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
        # GPS POINTS
        # ======================================================

        current_gps = (
            self.current_lat,
            self.current_lon
        )

        target_gps = (
            self.target_lat,
            self.target_lon
        )

        # ======================================================
        # DISTANCE
        # ======================================================

        horizontal_distance = haversine(
            current_gps,
            target_gps,
            unit=Unit.METERS
        )

        # ======================================================
        # TRUE GPS BEARING
        # ======================================================

        lat1 = math.radians(
            self.current_lat
        )

        lon1 = math.radians(
            self.current_lon
        )

        lat2 = math.radians(
            self.target_lat
        )

        lon2 = math.radians(
            self.target_lon
        )

        dlon = lon2 - lon1

        x = (
            math.sin(dlon)
            *
            math.cos(lat2)
        )

        y = (
            math.cos(lat1)
            *
            math.sin(lat2)
            -
            math.sin(lat1)
            *
            math.cos(lat2)
            *
            math.cos(dlon)
        )

        bearing_rad = math.atan2(
            x,
            y
        )

        target_bearing_deg = (
            math.degrees(
                bearing_rad
            )
            + 360.0
        ) % 360.0

        # ======================================================
        # BEARING ERROR
        # ======================================================

        bearing_error = (
            target_bearing_deg -
            self.current_yaw
        )

        while bearing_error > 180.0:
            bearing_error -= 360.0

        while bearing_error < -180.0:
            bearing_error += 360.0

        # ======================================================
        # ALTITUDE REMAINING
        # ======================================================

        height_remaining = (
            self.absolute_target_alt -
            self.current_alt
        )

        # ======================================================
        # ALIGN YAW FIRST
        # ======================================================

        if abs(bearing_error) > 10.0:

            pitch_deg = 0.0

            thrust = 0.5

        else:

            # ==================================================
            # GEOMETRIC PITCH
            # ==================================================

            safe_distance = max(
                horizontal_distance,
                0.01
            )

            # pitch_rad = math.atan2(
            #     height_remaining,
            #     safe_distance
            # )
            pitch_rad = math.atan2(
                safe_distance,
                height_remaining
                
            )

            pitch_deg = math.degrees(
                pitch_rad
            )

            # ==================================================
            # AGGRESSIVE FINAL CLIMB
            # ==================================================

            if (
                height_remaining > 0.0 and
                height_remaining < 300.0
            ):

                pitch_deg *= 0.2  #0.25

            # ==================================================
            # ALWAYS FULL THRUST
            # ==================================================

            thrust = 1.0

        # ======================================================
        # SEND COMMAND
        # ======================================================

        self.send_attitude(
            pitch_deg,
            target_bearing_deg,
            thrust
        )

        # ======================================================
        # LOGGING
        # ======================================================

        self.get_logger().info(

            f"CurrentBearing: "
            f"{self.current_yaw:.2f} deg | "

            f"TargetBearing: "
            f"{target_bearing_deg:.2f} deg | "

            f"BearingError: "
            f"{bearing_error:.2f} deg | "

            f"Distance: "
            f"{horizontal_distance:.2f} m | "

            f"RelativeAlt: "
            f"{relative_alt:.2f} m | "

            f"HeightRemaining: "
            f"{height_remaining:.2f} m | "

            f"PitchCmd: "
            f"{pitch_deg:.2f} deg | "

            f"Thrust: "
            f"{thrust:.2f}"
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
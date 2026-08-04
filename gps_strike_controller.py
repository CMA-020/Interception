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

from sensor_msgs.msg import (
    Imu,
    NavSatFix
)

from haversine import haversine
from haversine import Unit

import math
import socket
import json

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

class GPSStrikeController(Node):

    def __init__(
        self,
        target_lat,
        target_lon,
        target_relative_alt=750.0,
        socket_enabled=False,
        socket_ip='127.0.0.1',
        socket_port=5000
    ):

        super().__init__(
            'gps_strike_controller'
        )

        self.socket_enabled = socket_enabled
        self.socket_connected = False

        if self.socket_enabled:

            self.server_socket = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )

            self.server_socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1
            )

            self.server_socket.bind(
                (socket_ip, socket_port)
            )

            self.server_socket.listen(1)
            self.server_socket.setblocking(False)

            self.client_socket = None

            self.connection_timer = self.create_timer(
                0.5,
                self.check_socket_connection
            )

        # ==================================================
        # TARGET
        # ==================================================

        self.target_lat = target_lat
        self.target_lon = target_lon
        self.filtered_target_bearing = None
        # RELATIVE HEIGHT ABOVE TAKEOFF
        self.target_alt = (
            target_relative_alt
        )

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
        # SUBSCRIBERS
        # ==================================================

        self.imu_sub = self.create_subscription(

            Imu,

            '/mavros/imu/data',

            self.imu_cb,

            qos
        )

        self.gps_sub = self.create_subscription(

            NavSatFix,

            '/mavros/global_position/global',

            self.gps_cb,

            qos
        )

        self.alt_sub = self.create_subscription(

            Altitude,

            '/mavros/altitude',

            self.alt_cb,

            qos
        )

        self.state_sub = self.create_subscription(

            State,

            '/mavros/state',

            self.state_cb,

            10
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
        # FCU STATE
        # ==================================================

        self.current_state = State()

        # ==================================================
        # FLIGHT STATE
        # ==================================================

        self.takeoff_complete = False

        # ==================================================
        # CURRENT STATE
        # ==================================================

        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0

        self.current_lat = None
        self.current_lon = None

        self.current_alt = 0.0

        self.base_altitude = None

        self.absolute_target_alt = None

        # ==================================================
        # LOG
        # ==================================================

        self.get_logger().info(

            f"GPS Strike Initialized -> "

            f"Lat: {self.target_lat}, "

            f"Lon: {self.target_lon}"
        )



    # ==========================================================
    # SOCKET CONNECTION CHECK
    # ==========================================================

    def check_socket_connection(self):

        if not self.socket_enabled:
            return

        if self.socket_connected:
            return

        try:

            self.client_socket, addr = (
                self.server_socket.accept()
            )

            self.socket_connected = True

            self.get_logger().info(
                f"Socket connected: {addr}"
            )

        except BlockingIOError:
            pass


    # ==========================================================
    # SOCKET SEND
    # ==========================================================

    def send_socket_message(
        self,
        data
    ):

        if not self.socket_enabled:
            return

        if not self.socket_connected:
            return

        try:

            self.client_socket.sendall(
    (json.dumps(data) + "\n").encode()
)

        except:

            self.socket_connected = False


    # ==========================================================
    # UPDATE TARGET
    # ==========================================================

    def set_target(
        self,
        lat,
        lon
    ):

        self.target_lat = lat
        self.target_lon = lon

        self.get_logger().info(

            f"New Target -> "

            f"{lat}, {lon}"
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
    # IMU CALLBACK
    # ==========================================================

    def imu_cb(
        self,
        msg
    ):

        q = msg.orientation

        # ======================================================
        # ROLL
        # ======================================================

        sinr_cosp = 2.0 * (
            q.w * q.x +
            q.y * q.z
        )

        cosr_cosp = 1.0 - 2.0 * (
            q.x * q.x +
            q.y * q.y
        )

        roll = math.atan2(
            sinr_cosp,
            cosr_cosp
        )

        self.current_roll = math.degrees(
            roll
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

        if self.socket_enabled:

            data = [

                self.current_roll,

                self.current_pitch,

                self.current_yaw
            ]

            self.send_socket_message(
                data
            )

    # ==========================================================
    # GPS CALLBACK
    # ==========================================================

    def gps_cb(
        self,
        msg
    ):

        self.current_lat = (
            msg.latitude
        )

        self.current_lon = (
            msg.longitude
        )

    # ==========================================================
    # ALTITUDE CALLBACK
    # ==========================================================

    def alt_cb(
        self,
        msg
    ):

        self.current_alt = (
            msg.amsl
        )

        # ======================================================
        # SAVE TAKEOFF AMSL ALTITUDE
        # ======================================================

        if self.base_altitude is None:

            self.base_altitude = (
                self.current_alt
            )

            # ==================================================
            # CONVERT RELATIVE -> ABSOLUTE AMSL
            # ==================================================

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

        if self.socket_enabled and not self.socket_connected:

            self.get_logger().info(
                "Waiting for socket client..."
            )

            return

        # ======================================================
        # WAIT FOR FCU
        # ======================================================

        if not self.current_state.connected:

            self.get_logger().info(
                "Waiting for FCU..."
            )

            return

        # ======================================================
        # WAIT FOR ALTITUDE INIT
        # ======================================================

        if self.base_altitude is None:

            return

        # ======================================================
        # ALWAYS STREAM SETPOINTS
        # ======================================================

        self.send_attitude(

            0.0,

            self.current_yaw,

            1.0
        )

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
        # WAIT FOR GPS
        # ======================================================

        if self.current_lat is None:

            self.get_logger().info(
                "Waiting for GPS..."
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

                    f"Taking off | "

                    f"Relative Alt: "

                    f"{relative_alt:.2f}"
                )

                return

            else:

                self.takeoff_complete = True

                self.get_logger().info(
                    "Takeoff complete"
                )

        # ======================================================
        # CURRENT / TARGET GPS
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
        # TARGET HIT
        # ======================================================

        if horizontal_distance < 5.0:

            self.get_logger().info(
                "TARGET HIT"
            )

            self.send_attitude(

                0.0,

                self.current_yaw,

                0.0
            )

            return

        # # ======================================================
        # # TRUE GPS BEARING
        # # ======================================================

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

            bearing_error += 360.0     #####v1
##############v2

        # #======================================================
        # # TARGET BEARING
        # # ======================================================

        # lat1 = math.radians(
        #     self.current_lat
        # )

        # lon1 = math.radians(
        #     self.current_lon
        # )

        # lat2 = math.radians(
        #     self.target_lat
        # )

        # lon2 = math.radians(
        #     self.target_lon
        # )

        # dlon = lon2 - lon1

        # x = (
        #     math.sin(dlon)
        #     *
        #     math.cos(lat2)
        # )

        # y = (
        #     math.cos(lat1)
        #     *
        #     math.sin(lat2)
        #     -
        #     math.sin(lat1)
        #     *
        #     math.cos(lat2)
        #     *
        #     math.cos(dlon)
        # )

        # bearing_rad = math.atan2(
        #     x,
        #     y
        # )

        # target_bearing_deg = (
        #     math.degrees(
        #         bearing_rad
        #     )
        #     + 360.0
        # ) % 360.0
        # # ======================================================
        # # BEARING ERROR
        # # ======================================================

        # bearing_error = (

        #     target_bearing_deg -

        #     self.current_yaw
        # )

        # while bearing_error > 180.0:
        #     bearing_error -= 360.0

        # while bearing_error < -180.0:
        #     bearing_error += 360.0

        # # ======================================================
        # # TERMINAL DAMPING
        # # ======================================================

        # if horizontal_distance < 100.0:

        #     bearing_error *= 0.6

        # if horizontal_distance < 50.0:

        #     bearing_error *= 0.4

        # if horizontal_distance < 20.0:

        #     bearing_error *= 0.2

        # # ======================================================
        # # DAMPED TARGET YAW
        # # ======================================================

        # target_bearing_deg = (
        #     self.current_yaw +
        #     bearing_error
        # )

        # target_bearing_deg %= 360.0

        # ======================================================
        # HEIGHT REMAINING
        # ======================================================

        height_remaining = (

            self.absolute_target_alt -

            self.current_alt
        )

        # ======================================================
        # FIX YAW FIRST
        # ======================================================

        if abs(bearing_error) > 10.0:

            pitch_deg = 0.0

            thrust = 0.6

        # ======================================================
        # INTERCEPT
        # ======================================================

        else:

            safe_distance = max(

                horizontal_distance,

                0.01
            )

            pitch_rad = math.atan2(

                safe_distance,

                height_remaining
            )

            pitch_deg = math.degrees(
                pitch_rad
            )

            # ==================================================
            # FINAL APPROACH
            # ==================================================

            # if (

            #     height_remaining > 0.0 and

            #     height_remaining < 300.0
            # ):

            #     pitch_deg *= 0.2

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

            f"Mode: "
            f"{self.current_state.mode} | "

            f"Armed: "
            f"{self.current_state.armed} | "

            f"Distance: "
            f"{horizontal_distance:.2f} m | "

            f"BearingError: "
            f"{bearing_error:.2f} deg | "

            f"Pitch: "
            f"{pitch_deg:.2f} deg | "

            f"Yaw: "
            f"{target_bearing_deg:.2f} deg | "

            f"Thrust: "
            f"{thrust:.2f}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    controller = GPSStrikeController(

        target_lat=40.59190553758221,

        target_lon=-79.8862427481675,

        target_relative_alt=750.0
    )

    rclpy.spin(controller)

    controller.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
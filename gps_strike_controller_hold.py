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
import threading
import sys
import termios
import tty

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)

# ==========================================================
# GLOBAL CONFIGURATION
# ==========================================================
DEFAULT_THRUST = 1.0  # Default thrust value for attitude control
PRESTREAM_COUNT = 50  # 50 loops at 0.05s = 2.5 seconds of pre-streaming

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
        super().__init__('gps_strike_controller')

        self.socket_enabled = socket_enabled
        self.socket_connected = False

        if self.socket_enabled:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((socket_ip, socket_port))
            self.server_socket.listen(1)
            self.server_socket.setblocking(False)
            self.client_socket = None
            self.connection_timer = self.create_timer(0.5, self.check_socket_connection)

        # ==================================================
        # TARGET
        # ==================================================
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.filtered_target_bearing = None
        self.target_alt = target_relative_alt

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
        self.imu_sub = self.create_subscription(Imu, '/mavros/imu/data', self.imu_cb, qos)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_cb, qos)
        self.alt_sub = self.create_subscription(Altitude, '/mavros/altitude', self.alt_cb, qos)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)

        # ==================================================
        # SERVICES
        # ==================================================
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # ==================================================
        # TIMER
        # ==================================================
        self.timer = self.create_timer(0.05, self.run)

        # ==================================================
        # FCU STATE
        # ==================================================
        self.current_state = State()

        # ==================================================
        # FLIGHT STATE
        # ==================================================
        self.takeoff_complete = False
        self.attack_started = False
        self.prestream_counter = 0
        self.loop_tick = 0  # Dedicated counter for command retries

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

        self.prev_distance = None
        self.prev_distance_time = None
        self.target_passed = False
        self.prev_distance_rate = None
        self.offboard_disabled = False
        self.manual_hold_requested = False

        self.get_logger().warn('SIGN CHANGE VERSION LOADED - FINAL FIX (TIMESTAMP + DEADLOCK)')

        # ==================================================
        # LOG
        # ==================================================
        threading.Thread(target=self.keyboard_monitor, daemon=True).start()
        self.get_logger().info(f"GPS Strike Initialized -> Lat: {self.target_lat}, Lon: {self.target_lon}")

    # ==========================================================
    # SOCKET CONNECTION CHECK
    # ==========================================================
    def check_socket_connection(self):
        if not self.socket_enabled: return
        if self.socket_connected: return
        try:
            self.client_socket, addr = self.server_socket.accept()
            self.socket_connected = True
            self.get_logger().info(f"Socket connected: {addr}")
        except BlockingIOError:
            pass

    # ==========================================================
    # SOCKET SEND
    # ==========================================================
    def send_socket_message(self, data):
        if not self.socket_enabled or not self.socket_connected: return
        try:
            self.client_socket.sendall((json.dumps(data) + "\n").encode())
        except:
            self.socket_connected = False

    # ==========================================================
    # UPDATE TARGET
    # ==========================================================
    def set_target(self, lat, lon):
        self.target_lat = lat
        self.target_lon = lon
        self.get_logger().info(f"New Target -> {lat}, {lon}")

    # ==========================================================
    # STATE CALLBACK
    # ==========================================================
    def state_cb(self, msg):
        self.current_state = msg

    # ==========================================================
    # IMU CALLBACK
    # ==========================================================
    def imu_cb(self, msg):
        q = msg.orientation

        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.current_roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self.current_pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        
        yaw_deg_enu = math.degrees(yaw_rad)
        self.current_yaw = (90.0 - yaw_deg_enu) % 360.0

        if self.socket_enabled:
            self.send_socket_message([self.current_roll, self.current_pitch, self.current_yaw])

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
        if self.base_altitude is None:
            self.base_altitude = self.current_alt
            self.absolute_target_alt = self.base_altitude + self.target_alt
            self.get_logger().info(f"BaseAltitude: {self.base_altitude:.2f} m | AbsoluteTargetAlt: {self.absolute_target_alt:.2f} m")

    # ==========================================================
    # SEND ATTITUDE (FIXED WITH TIMESTAMP)
    # ==========================================================
    def send_attitude(self, pitch_deg, compass_yaw_deg, thrust):
        msg = AttitudeTarget()
        
        # THIS IS THE CRITICAL FIX: PX4 ignores setpoints without a valid ROS timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        px4_yaw_deg = (90.0 - compass_yaw_deg)
        while px4_yaw_deg > 180.0: px4_yaw_deg -= 360.0
        while px4_yaw_deg < -180.0: px4_yaw_deg += 360.0

        q = quaternion_from_euler(0.0, math.radians(pitch_deg), math.radians(px4_yaw_deg))

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        
        # Explicitly zeroing out body rates just like the other script
        msg.body_rate.x = 0.0
        msg.body_rate.y = 0.0
        msg.body_rate.z = 0.0
        
        msg.thrust = thrust
        self.pub.publish(msg)

    # ==========================================================
    # KEYBOARD MONITOR
    # ==========================================================
    def keyboard_monitor(self):
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            while rclpy.ok():
                tty.setcbreak(fd)
                ch = sys.stdin.read(1)
                if ch.lower() == 'h':
                    self.manual_hold_requested = True
                    self.get_logger().warn("h PRESSED -> HOLD REQUESTED")
        except Exception as e:
            self.get_logger().error(f"Keyboard monitor error: {e}")
        finally:
            try: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except: pass

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    def run(self):

        if self.socket_enabled and not self.socket_connected:
            self.get_logger().info("Waiting for socket client...")
            return

        # WAIT FOR FCU
        if not self.current_state.connected:
            self.get_logger().info("Waiting for FCU...")
            return

        # WAIT FOR ALTITUDE INIT
        if self.base_altitude is None:
            return

        # ======================================================
        # ALWAYS STREAM SETPOINTS & INCREMENT TICKER
        # ======================================================
        self.send_attitude(0.0, self.current_yaw, DEFAULT_THRUST)
        self.loop_tick += 1

        # ======================================================
        # PRE-STREAM COUNTDOWN
        # ======================================================
        if self.prestream_counter < PRESTREAM_COUNT:
            self.prestream_counter += 1
            if self.prestream_counter % 10 == 0:
                self.get_logger().info(f"Pre-streaming setpoints... {self.prestream_counter}/{PRESTREAM_COUNT}")
            return

        # ======================================================
        # MANUAL HOLD (H KEY)
        # ======================================================
        if self.manual_hold_requested:
            self.target_passed = True
            self.offboard_disabled = True
            req = SetMode.Request()
            req.custom_mode = "AUTO.LOITER"
            self.mode_client.call_async(req)
            self.get_logger().warn("MANUAL HOLD ACTIVATED")
            return

        # ======================================================
        # TARGET PASSED
        # ======================================================
        if self.target_passed:
            self.get_logger().warn("TARGET PASSED - HOLDING")
            return

        # ======================================================
        # 1. ARM FIRST (Retries once every second)
        # ======================================================
        if not self.current_state.armed:
            if self.loop_tick % 20 == 0:  # 20 ticks at 0.05s = 1 second
                req = CommandBool.Request()
                req.value = True
                self.arm_client.call_async(req)
                self.get_logger().info("Attempting to ARM...")
            return

        # ======================================================
        # 2. THEN OFFBOARD (Retries once every second)
        # ======================================================
        if not self.offboard_disabled and self.current_state.mode != "OFFBOARD":
            if self.loop_tick % 20 == 0:
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self.mode_client.call_async(req)
                self.get_logger().info("Attempting OFFBOARD mode switch...")
            return

        # ======================================================
        # WAIT FOR GPS
        # ======================================================
        if self.current_lat is None:
            self.get_logger().info("Waiting for GPS...")
            return

        # ======================================================
        # TAKEOFF TO 10m
        # ======================================================
        relative_alt = self.current_alt - self.base_altitude
        if not self.takeoff_complete:
            if relative_alt < 10.0:
                self.send_attitude(0.0, self.current_yaw, DEFAULT_THRUST)
                self.get_logger().info(f"Taking off | Relative Alt: {relative_alt:.2f}")
                return
            else:
                self.takeoff_complete = True
                self.get_logger().info("Takeoff complete")

        # ======================================================
        # DISTANCE & PASS DETECTION
        # ======================================================
        current_gps = (self.current_lat, self.current_lon)
        target_gps = (self.target_lat, self.target_lon)

        horizontal_distance = haversine(current_gps, target_gps, unit=Unit.METERS)
        current_time = self.get_clock().now().nanoseconds * 1e-9
        distance_rate = 0.0

        if self.prev_distance is not None:
            dt = current_time - self.prev_distance_time
            if dt > 0.0:
                distance_rate = (horizontal_distance - self.prev_distance) / dt

        self.get_logger().warn(f"D={horizontal_distance:.1f} Rate={distance_rate:.1f}")

        if (not self.target_passed and horizontal_distance < 150.0 
            and self.prev_distance_rate is not None 
            and self.prev_distance_rate < 0.0 and distance_rate > 0.0):
            
            self.target_passed = True
            self.offboard_disabled = True
            self.get_logger().warn(f"TARGET PASSED! {self.prev_distance_rate:.1f} -> {distance_rate:.1f}")
            
            req = SetMode.Request()
            req.custom_mode = "AUTO.LOITER"
            self.mode_client.call_async(req)
            return

        self.prev_distance = horizontal_distance
        self.prev_distance_time = current_time
        self.prev_distance_rate = distance_rate

        # ======================================================
        # TRUE GPS BEARING
        # ======================================================
        lat1 = math.radians(self.current_lat)
        lon1 = math.radians(self.current_lon)
        lat2 = math.radians(self.target_lat)
        lon2 = math.radians(self.target_lon)
        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        target_bearing_deg = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

        # ======================================================
        # BEARING ERROR
        # ======================================================
        bearing_error = target_bearing_deg - self.current_yaw
        while bearing_error > 180.0: bearing_error -= 360.0
        while bearing_error < -180.0: bearing_error += 360.0

        # ======================================================
        # HEIGHT REMAINING
        # ======================================================
        height_remaining = self.absolute_target_alt - self.current_alt

        # ======================================================
        # ONE-TIME INITIAL YAW ALIGNMENT
        # ======================================================
        if not self.attack_started:
            if abs(bearing_error) > 1.0:
                # Stay level only during the initial alignment after takeoff.
                self.send_attitude(0.0, target_bearing_deg, DEFAULT_THRUST)
                self.get_logger().info(
                    f"Initial yaw alignment | BearingError: {bearing_error:.2f} deg"
                )
                return

            self.attack_started = True
            self.get_logger().info("Initial alignment complete. Starting attack run.")

        # ======================================================
        # ATTACK RUN
        # ======================================================
        # From this point onward yaw is corrected continuously,
        # but pitch is NEVER gated by bearing error again.
        safe_distance = max(horizontal_distance, 0.01)
        pitch_rad = math.atan2(safe_distance, height_remaining)
        pitch_deg = math.degrees(pitch_rad)
        # if 0.0 < height_remaining < 300.0:
        #     pitch_deg *= 0.2
        thrust = DEFAULT_THRUST

        # ======================================================
        # SEND FINAL COMMAND
        # ======================================================
        self.send_attitude(pitch_deg, target_bearing_deg, thrust)

        self.get_logger().info(
            f"Mode={self.current_state.mode} | "
            f"Armed={self.current_state.armed} | "
            f"Lat={self.current_lat:.7f} Lon={self.current_lon:.7f} | "
            f"TargetLat={self.target_lat:.7f} TargetLon={self.target_lon:.7f} | "
            f"HorizDist={horizontal_distance:.2f} m | "
            f"HeightRemain={height_remaining:.2f} m | "
            f"CurrentAlt={self.current_alt:.2f} m | "
            f"TargetAlt={self.absolute_target_alt:.2f} m | "
            f"CurrentYaw={self.current_yaw:.2f} deg | "
            f"TargetYaw={target_bearing_deg:.2f} deg | "
            f"BearingError={bearing_error:.2f} deg | "
            f"PitchCmd={pitch_deg:.2f} deg | "
            f"PitchRad={pitch_rad:.4f} | "
            f"Thrust={thrust:.2f}"
        )


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
#!/usr/bin/env python3
"""
GPS Strike Controller — VTOL Edition with Gazebo Visual Target Tracker
=====================================================================
"""

import math
import rclpy
import sys
import termios
import tty
import threading
import time
import cv2
import numpy as np
import os
import csv

from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget, Altitude, State
from mavros_msgs.srv import CommandBool, SetMode, CommandVtolTransition
from sensor_msgs.msg import Imu, NavSatFix
from haversine import haversine, Unit
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Gazebo Transport Imports
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage

# ──────────────────────────────────────────────────────────────
# PHASES & STATES
# ──────────────────────────────────────────────────────────────
PHASE_TAKEOFF       = "TAKEOFF"
PHASE_MC_CLIMB      = "MC_CLIMB"
PHASE_FW_TRANSITION = "FW_TRANSITION"
PHASE_FW_CRUISE     = "FW_CRUISE"
PHASE_TARGET_PASSED = "TARGET_PASSED"

TRACKING = "TRACKING"
CRUISE   = "CRUISE"


# ──────────────────────────────────────────────────────────────
# SIMPLE PID HELPER
# ──────────────────────────────────────────────────────────────
class SimplePID:
    def __init__(self, kp, ki, kd, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def compute(self, error, now):
        if self.last_time is None:
            self.last_time = now
            self.last_error = error
            return 0.0
        
        dt = now - self.last_time
        if dt <= 0.0:
            return 0.0

        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        output = max(-self.max_output, min(self.max_output, output))
        
        self.last_error = error
        self.last_time = now
        return output

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


# ──────────────────────────────────────────────────────────────
# QUATERNION HELPER
# ──────────────────────────────────────────────────────────────
def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw  * 0.5)
    sy = math.sin(yaw  * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)

    return [
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    ]


# ──────────────────────────────────────────────────────────────
# MAIN NODE CONTROLLER
# ──────────────────────────────────────────────────────────────
class GPSStrikeController(Node):

    def __init__(self, target_lat, target_lon, target_relative_alt=750.0):
        super().__init__("gps_strike_controller")

        # ── Target ────────────────────────────────────────────
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_relative_alt = target_relative_alt

        # ── Tunable parameters ────────────────────────────────
        self.fw_transition_threshold_m = 1000.0
        self.transition_duration_s = 5.0
        self.mc_climb_thrust = 0.75   
        self.fw_cruise_thrust = 0.70  
        self.fw_pitch_deg = 0.0    
        self.fw_yaw_gain = 0.5    
        self.fw_yaw_clamp = 30.0   
        self.pass_detection_radius_m = 350.0
        self.takeoff_alt_m = 10.0

        # ── Visual Tracking Parameters ────────────────────────
        self.hfov = 60.0  
        self.vfov = 45.0  
        self.track_on = False
        self.tracker_initialized = False
        self.current_image = None
        self.height = 0
        self.width = 0
        self._model_ready = True  
        self._seq_state = CRUISE  
        self.free_float = True
        self.TRACK_THRUST = 0.75

        self.pid_roll = SimplePID(kp=0.3, ki=0.001, kd=0.2, max_output=1.5)
        self.pid_pitch = SimplePID(kp=1.0, ki=0.00001, kd=0.4, max_output=1.5)
        self.pid_yaw = SimplePID(kp=1.5, ki=0.00001, kd=0.5, max_output=1.5) 

        # ── Flight State Machine Variables ────────────────────
        self.phase = PHASE_TAKEOFF
        self.current_state = State()
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0        
        self.fw_yaw = 0.0

        self.current_lat = None
        self.current_lon = None
        self.current_alt = 0.0        

        self.base_altitude = None
        self.absolute_target_alt = None
        
        self.start_lat = None
        self.start_lon = None

        self.takeoff_complete = False
        self.vtol_sent = False
        self.target_passed = False
        self.offboard_disabled = False
        self.manual_hold_requested = False

        self.prev_distance = None
        self.prev_distance_time = None
        self.prev_distance_rate = None

        self._current_bbox = None

        # ── Performance Metric Logging Variables ──────────────
        self.takeoff_start_time = None
        self.vtol_start_time = None
        self.vtol_transition_time = 0.0
        self.csv_logged = False
        
        # Velocity estimation tracking state
        self.prev_run_lat = None
        self.prev_run_lon = None
        self.prev_run_alt = None
        self.prev_run_time = None
        self.current_speed = 0.0

        # ── ROS2 QoS & Communication ─────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.att_pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)

        self.create_subscription(State, "/mavros/state", self.state_cb, 10)
        self.create_subscription(Imu, "/mavros/imu/data", self.imu_cb, qos)
        self.create_subscription(NavSatFix, "/mavros/global_position/global", self.gps_cb, qos)
        self.create_subscription(Altitude, "/mavros/altitude", self.alt_cb, qos)

        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.vtol_client = self.create_client(CommandVtolTransition, "/mavros/cmd/vtol_transition")

        # Split control architecture loops
        self.timer = self.create_timer(0.05, self.run)                  # Fast Navigation (20 Hz)
        self.tracker_timer = self.create_timer(0.2, self.tracker_ang)    # Tracking Control Loop (5 Hz)

        threading.Thread(target=self.keyboard_monitor, daemon=True).start()
        threading.Thread(target=self.gazebo_transport_subscriber, daemon=True).start()

        self.get_logger().info(
            f"GPSStrikeVTOL initialized | Target: ({target_lat:.6f}, {target_lon:.6f})"
        )

    def gazebo_transport_subscriber(self):
        gz_node = GzNode()
        gz_node.subscribe(GzImage, "/camera/image", self.image_callback)
        while rclpy.ok():
            time.sleep(0.01)

    def image_callback(self, msg):
        try:
            img_np = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = img_np.reshape((msg.height, msg.width, 3))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            if cv_image is None:
                return

            self.current_image = cv_image.copy()
            self.height = msg.height
            self.width = msg.width

            if self.track_on:
                detection = self.detect_largest_contour_center(self.current_image)
                if detection is not None and self._model_ready:
                    cx, cy, largest_contour, edges = detection
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    self._current_bbox = (x, y, w, h)

                    # Draw a green bounding box around the detected entity
                    cv2.rectangle(self.current_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if not self.tracker_initialized:
                        self.tracker_initialized = True
                else:
                    self._current_bbox = None

            cv2.imshow("Gazebo Camera Tracker", self.current_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Image subscriber loop error: {e}")

    def detect_largest_contour_center(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy, largest, edges

    def state_cb(self, msg):
        self.current_state = msg

    def gps_cb(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        
        if self.start_lat is None:
            self.start_lat = msg.latitude
            self.start_lon = msg.longitude
            self.get_logger().info(f"Startup coordinates locked: ({self.start_lat}, {self.start_lon})")

    def alt_cb(self, msg):
        self.current_alt = msg.amsl
        if self.base_altitude is None:
            self.base_altitude = self.current_alt
            self.absolute_target_alt = self.base_altitude + self.target_relative_alt

    def imu_cb(self, msg):
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.current_roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        self.current_pitch = math.degrees(math.asin(sinp))

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_enu = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.current_yaw = (90.0 - yaw_enu) % 360.0

        vec_x = 2.0 * (q.x * q.z + q.w * q.y)
        vec_y = 2.0 * (q.y * q.z - q.w * q.x)
        fw_yaw_enu = math.degrees(math.atan2(vec_y, vec_x))
        self.fw_yaw = (90.0 - fw_yaw_enu) % 360.0

    def set_target(self, lat, lon):
        self.target_lat = lat
        self.target_lon = lon

    def keyboard_monitor(self):
        try:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            while rclpy.ok():
                tty.setcbreak(fd)
                ch = sys.stdin.read(1)
                if ch.lower() == "h":
                    self.manual_hold_requested = True
        except Exception as e:
            self.get_logger().error(f"Keyboard monitor error: {e}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except:
                pass

    def request_fw_transition(self):
        req = CommandVtolTransition.Request()
        req.state = 4   
        self.vtol_client.call_async(req)

    def send_attitude(self, roll_deg, pitch_deg, compass_yaw_deg, thrust):
        msg = AttitudeTarget()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE  |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )
        px4_yaw = 90.0 - compass_yaw_deg
        while px4_yaw >  180.0: px4_yaw -= 360.0
        while px4_yaw < -180.0: px4_yaw += 360.0

        q = quaternion_from_euler(math.radians(roll_deg), math.radians(pitch_deg), math.radians(px4_yaw))
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust = thrust
        self.att_pub.publish(msg)

    def _publish_body_rates(self, roll_rate: float, pitch_rate: float, yaw_rate: float, thrust: float):
        msg = AttitudeTarget()
        msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE   
        msg.body_rate.x = float(roll_rate)
        msg.body_rate.y = float(pitch_rate)
        msg.body_rate.z = float(yaw_rate)
        msg.thrust = float(thrust)
        self.att_pub.publish(msg)

    def _bearing_to_target(self):
        lat1, lon1 = math.radians(self.current_lat), math.radians(self.current_lon)
        lat2, lon2 = math.radians(self.target_lat), math.radians(self.target_lon)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    @staticmethod
    def _wrap_error(err):
        while err >  180.0: err -= 360.0
        while err < -180.0: err += 360.0
        return err

    def _horizontal_distance(self):
        return haversine((self.current_lat, self.current_lon), (self.target_lat, self.target_lon), unit=Unit.METERS)

    def _mc_pitch_to_target(self, horizontal_distance):
        height_remaining = self.absolute_target_alt - self.current_alt
        safe_dist = max(horizontal_distance, 0.01)
        return math.degrees(math.atan2(safe_dist, height_remaining))

    def log_metrics_to_csv(self, total_time, total_dist, vtol_time, intercept_speed):
        """Appends flight performance details to interception.csv, generating it if non-existent."""
        file_name = "interception.csv"
        file_exists = os.path.exists(file_name)
        try:
            with open(file_name, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["total_time_s", "total_dist_m", "vtol_transition_time_s", "total_speed_at_interception_mps"])
                writer.writerow([total_time, total_dist, vtol_time, intercept_speed])
            self.get_logger().info(f"[CSV LOG] Data saved to {file_name}: Time={total_time:.2f}s, Dist={total_dist:.2f}m, VTOL_From_Launch={vtol_time:.2f}s, Speed={intercept_speed:.2f}m/s")
        except Exception as e:
            self.get_logger().error(f"Failed to record data in CSV: {e}")

    # ──────────────────────────────────────────────────────────
    # VISION TRACKING BODY RATE CONTROLLER (5 Hz Loop)
    # ──────────────────────────────────────────────────────────
    def tracker_ang(self):
        if self._seq_state != TRACKING:
            return

        bbox = self._current_bbox
        if self.current_image is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.track_on and bbox is not None:
            self.free_float = False
            x, y, w, h = bbox

            self.Track_centerx = x + w / 2.0
            self.Track_centery = y + h / 2.0

            target_x = self.width / 2.0
            target_y = self.height / 2.0

            error_x = self.Track_centerx - target_x   
            error_y = self.Track_centery - target_y   

            err_x_rad = math.radians(self.hfov * error_x * 1.0 / self.width)
            err_y_rad = math.radians(self.vfov * error_y / self.height)

            # Compute baseline time interval (dt) to resolve Derivative / Integral components
            roll_dt = 0.0 if self.pid_roll.last_time is None else (now - self.pid_roll.last_time)
            pitch_dt = 0.0 if self.pid_pitch.last_time is None else (now - self.pid_pitch.last_time)
            yaw_dt = 0.0 if self.pid_yaw.last_time is None else (now - self.pid_yaw.last_time)

            # Roll metrics extraction
            roll_p = err_x_rad
            roll_i = self.pid_roll.integral + (err_x_rad * roll_dt) if roll_dt > 0.0 else self.pid_roll.integral
            roll_d = (err_x_rad - self.pid_roll.last_error) / roll_dt if roll_dt > 0.0 else 0.0

            # Pitch metrics extraction
            pitch_p = err_y_rad
            pitch_i = self.pid_pitch.integral + (err_y_rad * pitch_dt) if pitch_dt > 0.0 else self.pid_pitch.integral
            pitch_d = (err_y_rad - self.pid_pitch.last_error) / pitch_dt if pitch_dt > 0.0 else 0.0

            # Yaw metrics extraction (Derived from horizontal frame tracking space shifts)
            yaw_p = err_x_rad
            yaw_i = self.pid_yaw.integral + (err_x_rad * yaw_dt) if yaw_dt > 0.0 else self.pid_yaw.integral
            yaw_d = (err_x_rad - self.pid_yaw.last_error) / yaw_dt if yaw_dt > 0.0 else 0.0

            self.get_logger().info(
                f"errors \n"
                f" errory : {err_y_rad:.4f} | errorx: {err_x_rad:.4f} \n")

            # Logging physical telemetry metrics for absolute orientation states
            self.get_logger().info(
                f"[TRACKING VEHICLE ATTITUDE]\n"
                f" -> Current Roll: {self.current_roll:.2f}° | Pitch: {self.current_pitch:.2f}° | Yaw: {self.current_yaw:.2f}° (FW Axis Yaw: {self.fw_yaw:.2f}°)"
            )

            # Included YAW P_err, I_err, and D_err lines inside standard prints
            self.get_logger().info(
                f"[TRACKING CONTROL ERRORS]\n"
                f" -> ROLL  | P_err: {roll_p:.4f} | I_err: {roll_i:.4f} | D_err: {roll_d:.4f}\n"
                f" -> PITCH | P_err: {pitch_p:.4f} | I_err: {pitch_i:.4f} | D_err: {pitch_d:.4f}\n"
                f" -> YAW   | P_err: {yaw_p:.4f} | I_err: {yaw_i:.4f} | D_err: {yaw_d:.4f}"
            )

            yaw_rate = self.pid_yaw.compute(err_x_rad, now)
            roll_rate = 0.0 
            pitch_rate = self.pid_pitch.compute(err_y_rad, now)

            self._publish_body_rates(
                roll_rate=roll_rate,
                pitch_rate=pitch_rate,
                yaw_rate=yaw_rate,
                thrust=self.TRACK_THRUST,
            )
        else:
            self.free_float = True
            self.pid_roll.reset()
            self.pid_pitch.reset()
            self.pid_yaw.reset()
            if self.current_lat is not None:
                target_bearing = self._bearing_to_target()
                self.send_attitude(0.0, self.fw_pitch_deg, target_bearing, self.TRACK_THRUST)

    # ──────────────────────────────────────────────────────────
    # PRIMARY NAVIGATION FLIGHT CONTROLLER LOOP (20 Hz Loop)
    # ──────────────────────────────────────────────────────────
    def run(self):
        if not self.current_state.connected or self.base_altitude is None or self.start_lat is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Start flight clock the first moment initialization hits loop execution
        if self.takeoff_start_time is None:
            self.takeoff_start_time = now

        # Calculate absolute 3D velocity tracking via localized update deltas over time
        if self.prev_run_lat is not None and self.prev_run_time is not None:
            dt_speed = now - self.prev_run_time
            if dt_speed > 0.0:
                d_horiz = haversine((self.prev_run_lat, self.prev_run_lon), (self.current_lat, self.current_lon), unit=Unit.METERS)
                d_vert = self.current_alt - self.prev_run_alt
                d_3d = math.sqrt(d_horiz**2 + d_vert**2)
                self.current_speed = d_3d / dt_speed

        self.prev_run_lat = self.current_lat
        self.prev_run_lon = self.current_lon
        self.prev_run_alt = self.current_alt
        self.prev_run_time = now

        horizontal_dist = self._horizontal_distance() if self.current_lat is not None else 9999.0
        
        if horizontal_dist <= 350.0 and not self.track_on:
            self.get_logger().warn(f"[TRACKER] Target within 350m ({horizontal_dist:.1f}m). Initializing Track Phase!")
            self.track_on = True
            self._seq_state = TRACKING

        # Calculate metrics when change of distance sign gets flipped (interception condition)
        distance_rate = 0.0
        if self.prev_distance is not None:
            dt = now - self.prev_distance_time
            if dt > 0.0:
                distance_rate = (horizontal_dist - self.prev_distance) / dt

        # INTERCEPTION DETECTED: When the closing rate derivative flips from negative to positive
        if (not self.target_passed and horizontal_dist < self.pass_detection_radius_m 
                and self.prev_distance_rate is not None and self.prev_distance_rate < 0.0 and distance_rate > 0.0):
            
            self.target_passed = True
            self.offboard_disabled = True
            
            # Extract final metric calculations at the exact interception delta flip
            total_flight_time = now - self.takeoff_start_time
            total_flight_dist = haversine((self.start_lat, self.start_lon), (self.current_lat, self.current_lon), unit=Unit.METERS)
            
            # Write metrics into performance database
            if not self.csv_logged:
                self.log_metrics_to_csv(total_flight_time, total_flight_dist, self.vtol_transition_time, self.current_speed)
                self.csv_logged = True

            req = SetMode.Request(); req.custom_mode = "AUTO.LOITER"
            self.mode_client.call_async(req)
            return

        self.prev_distance = horizontal_dist
        self.prev_distance_time = now
        self.prev_distance_rate = distance_rate

        # Guard clause: Bypass nav states if tracking timer handles offboard outputs
        if self._seq_state == TRACKING and self._current_bbox is not None:
            return  

        self.send_attitude(0.0, 0.0, self.current_yaw, 1.0)

        if self.manual_hold_requested or self.target_passed:
            return

        if not self.offboard_disabled and self.current_state.mode != "OFFBOARD":
            req = SetMode.Request(); req.custom_mode = "OFFBOARD"
            self.mode_client.call_async(req)
            return

        if not self.current_state.armed:
            req = CommandBool.Request(); req.value = True
            self.arm_client.call_async(req)
            return

        relative_alt = self.current_alt - self.base_altitude
        target_bearing = self._bearing_to_target() 
        
        if self.phase in [PHASE_FW_TRANSITION, PHASE_FW_CRUISE]:
            current_heading = (self.current_roll) % 360.0
        else:
            current_heading = self.current_yaw

        # ── PHASE Machine Steps ────────────────────────────────
        if self.phase == PHASE_TAKEOFF:
            if relative_alt < self.takeoff_alt_m:
                self.send_attitude(0.0, 0.0, self.current_yaw, 1.0)
                return
            else:
                self.takeoff_complete = True
                self.phase = PHASE_MC_CLIMB
                self.get_logger().info("[PHASE] Takeoff complete. Switching to PHASE_MC_CLIMB.")
                return 

        elif self.phase == PHASE_MC_CLIMB:
            distance_from_initial = haversine(
                (self.start_lat, self.start_lon), 
                (self.current_lat, self.current_lon), 
                unit=Unit.METERS
            )

            if self.current_alt < self.absolute_target_alt:
                safe_dist = 1000.0 - distance_from_initial
                pitch_deg = self._mc_pitch_to_target(safe_dist)
                thrust = 1.0

                self.get_logger().info(
                    f"[CLIMB] Covered: {distance_from_initial:.1f}m | "
                    f"Remaining Window (safe_dist): {safe_dist:.1f}m | "
                    f"Alt: {relative_alt:.1f}/{self.target_relative_alt:.1f}m | "
                    f"Pitch Cmd: {pitch_deg:.2f}°"
                )

                self.send_attitude(0.0, pitch_deg, target_bearing, thrust)
                return
            else:
                self.phase = PHASE_FW_TRANSITION
                self.get_logger().warn(
                    f"[CLIMB COMPLETE] Target absolute alt reached ({self.current_alt:.1f}m AMSL). Changing phase."
                )
                return

        elif self.phase == PHASE_FW_TRANSITION:
            if not self.vtol_sent:
                self.get_logger().warn(
                    f"[TRANSITION] Sending MAVROS VTOL transition command to FIXED-WING mode."
                )
                self.request_fw_transition()
                self.vtol_sent = True
                self.vtol_start_time = now
            
            self.send_attitude(0.0, 0.0, current_heading, self.fw_cruise_thrust)
            
            # Allow full duration window to compute fixed-wing transition elapsed value
            if now - self.vtol_start_time >= self.transition_duration_s:
                # MODIFIED: Measures time from initialization/launch up to transition completion
                self.vtol_transition_time = now - self.takeoff_start_time
                self.phase = PHASE_FW_CRUISE
                self.get_logger().info(f"[TRANSITION COMPLETE] Total time from launch to fixed-wing cruise: {self.vtol_transition_time:.2f} seconds.")
            return

        elif self.phase == PHASE_FW_CRUISE:
            fw_bearing_error = self._wrap_error(target_bearing - self.fw_yaw)
            roll_cmd = max(-self.fw_yaw_clamp, min(self.fw_yaw_clamp, fw_bearing_error * self.fw_yaw_gain))
            coordinated_pitch = self.fw_pitch_deg + (abs(roll_cmd) * 0.25)
            
            self.get_logger().info(f"[FW CRUISE] Remaining distance to target: {horizontal_dist:.1f}m")
            
            self.send_attitude(roll_cmd, coordinated_pitch, self.fw_yaw, self.fw_cruise_thrust)


def main():
    rclpy.init()
    node = GPSStrikeController(
        target_lat=40.59190553758221,
        target_lon=-79.8862427481675,
        target_relative_alt=750.0,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
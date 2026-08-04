#!/usr/bin/env python3
"""
GPS Strike Controller — VTOL Edition
=====================================
Combines gps_strike_controller_hold.py (MC intercept logic) with
tailsitter_vtol_fw_negative_offset.py (VTOL transition + FW cruise).

Flight phases
-------------
TAKEOFF        : Climb to 10 m relative (pure hover, pitch=0)
MC_CLIMB       : distance > 1000 m  →  stay in MC, pitch toward target
                 using atan2(horizontal_dist, height_remaining).
                 Fly until distance shrinks to 1000 m.
FW_TRANSITION  : Trigger VTOL transition service, pitch smoothly to 0°
                 (body horizontal) over transition_duration_s seconds.
FW_CRUISE      : pitch = 0  (fully horizontal, body axis)
                 Bearing error → corrected via YAW  (new roll in FW frame).
                 Roll = bearing_error * yaw_gain   (clamped to ±yaw_clamp)
TARGET_PASSED  : Distance inflection detected → AUTO.LOITER
"""

import math
import rclpy
import sys
import termios
import tty
import threading

from rclpy.node import Node

from mavros_msgs.msg import AttitudeTarget, Altitude, State
from mavros_msgs.srv import CommandBool, SetMode, CommandVtolTransition

from sensor_msgs.msg import Imu, NavSatFix

from haversine import haversine, Unit

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ──────────────────────────────────────────────────────────────
# PHASES
# ──────────────────────────────────────────────────────────────

PHASE_TAKEOFF       = "TAKEOFF"
PHASE_MC_CLIMB      = "MC_CLIMB"
PHASE_FW_TRANSITION = "FW_TRANSITION"
PHASE_FW_CRUISE     = "FW_CRUISE"
PHASE_TARGET_PASSED = "TARGET_PASSED"


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
# NODE
# ──────────────────────────────────────────────────────────────

class GPSStrikeController(Node):

    def __init__(
        self,
        target_lat,
        target_lon,
        target_relative_alt = 750.0,
    ):

        super().__init__("gps_strike_controller")

        # ── Target ────────────────────────────────────────────
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_relative_alt = target_relative_alt

        # ── Tunable parameters ────────────────────────────────

        # Distance threshold that separates MC_CLIMB from FW phases
        self.fw_transition_threshold_m = 1000.0

        # How long (s) to sweep pitch from MC_CLIMB angle → 0° (FW)
        self.transition_duration_s = 5.0

        # MC hover parameters
        self.mc_climb_thrust   = 0.75   # thrust during MC_CLIMB

        # FW cruise parameters
        self.fw_cruise_thrust  = 0.70   # thrust during FW_CRUISE
        self.fw_pitch_deg      = 0.0    # body pitch in FW (0 = horizontal)

        # In FW convention the old ROLL axis steers laterally.
        # We map bearing_error → roll to correct heading.
        self.fw_yaw_gain       = 0.5    # bearing_error → roll (°/°)
        self.fw_yaw_clamp      = 30.0   # max roll in FW (°)

        # Pass-detection zone
        self.pass_detection_radius_m = 150.0

        # Minimum relative altitude before MC_CLIMB pitch-toward-target starts
        self.takeoff_alt_m = 10.0

        # ── State ─────────────────────────────────────────────

        self.phase = PHASE_TAKEOFF

        self.current_state  = State()
        self.current_roll   = 0.0
        self.current_pitch  = 0.0
        self.current_yaw    = 0.0        # compass degrees

        self.current_lat    = None
        self.current_lon    = None
        self.current_alt    = 0.0        # AMSL

        self.base_altitude          = None
        self.absolute_target_alt    = None

        # Tracking starting coordinate for precise window-based climb
        self.start_climb_lat        = None
        self.start_climb_lon        = None

        self.takeoff_complete       = False
        self.vtol_sent              = False
        self.target_passed          = False
        self.offboard_disabled      = False
        self.manual_hold_requested  = False

        self._trans_pitch_start     = None   # pitch when FW_TRANSITION began
        self._trans_start_time      = None

        self.prev_distance          = None
        self.prev_distance_time     = None
        self.prev_distance_rate     = None

        # ── QoS ───────────────────────────────────────────────

        qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 10
        )

        # ── Publishers ────────────────────────────────────────

        self.att_pub = self.create_publisher(
            AttitudeTarget,
            "/mavros/setpoint_raw/attitude",
            10
        )

        # ── Subscribers ───────────────────────────────────────

        self.create_subscription(State,     "/mavros/state",                    self.state_cb, 10)
        self.create_subscription(Imu,       "/mavros/imu/data",                 self.imu_cb,   qos)
        self.create_subscription(NavSatFix, "/mavros/global_position/global",   self.gps_cb,   qos)
        self.create_subscription(Altitude,  "/mavros/altitude",                 self.alt_cb,   qos)

        # ── Service clients ───────────────────────────────────

        self.arm_client  = self.create_client(CommandBool,           "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode,               "/mavros/set_mode")
        self.vtol_client = self.create_client(CommandVtolTransition, "/mavros/cmd/vtol_transition")

        # ── Timer ─────────────────────────────────────────────

        self.timer = self.create_timer(0.05, self.run)

        # ── Keyboard monitor (h = manual hold) ────────────────

        threading.Thread(target=self.keyboard_monitor, daemon=True).start()

        self.get_logger().info(
            f"GPSStrikeVTOL ready | "
            f"target=({target_lat:.6f}, {target_lon:.6f}) "
            f"alt_rel={target_relative_alt:.0f}m"
        )


    # ──────────────────────────────────────────────────────────
    # CALLBACKS
    # ──────────────────────────────────────────────────────────

    def state_cb(self, msg):
        self.current_state = msg

    def gps_cb(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    def alt_cb(self, msg):
        self.current_alt = msg.amsl

        if self.base_altitude is None:
            self.base_altitude       = self.current_alt
            self.absolute_target_alt = self.base_altitude + self.target_relative_alt

            self.get_logger().info(
                f"Base AMSL: {self.base_altitude:.2f} m | "
                f"Target AMSL: {self.absolute_target_alt:.2f} m"
            )

    def imu_cb(self, msg):
        q = msg.orientation

        # ── Standard MC Euler Extraction (Fails at 90 deg pitch) ──
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.current_roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        self.current_pitch = math.degrees(math.asin(sinp))

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_enu   = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.current_yaw = (90.0 - yaw_enu) % 360.0

        # ── Bulletproof Fixed-Wing Heading (Immune to Gimbal Lock) ──
        # In FW mode, the tailsitter Z-axis (top of drone) points forward.
        # We rotate the Body Z-axis (0,0,1) into the Earth frame.
        vec_x = 2.0 * (q.x * q.z + q.w * q.y)
        vec_y = 2.0 * (q.y * q.z - q.w * q.x)
        
        fw_yaw_enu = math.degrees(math.atan2(vec_y, vec_x))
        self.fw_yaw = (90.0 - fw_yaw_enu) % 360.0

    # ──────────────────────────────────────────────────────────
    # KEYBOARD MONITOR
    # ──────────────────────────────────────────────────────────

    def keyboard_monitor(self):
        try:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            while rclpy.ok():
                tty.setcbreak(fd)
                ch = sys.stdin.read(1)
                if ch.lower() == "h":
                    self.manual_hold_requested = True
                    self.get_logger().warn("h PRESSED → HOLD REQUESTED")
        except Exception as e:
            self.get_logger().error(f"Keyboard monitor: {e}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except:
                pass


    # ──────────────────────────────────────────────────────────
    # VTOL TRANSITION SERVICE
    # ──────────────────────────────────────────────────────────

    def request_fw_transition(self):
        req       = CommandVtolTransition.Request()
        req.state = 4   # STATE_FW
        self.vtol_client.call_async(req)
        self.get_logger().info("VTOL TRANSITION COMMAND SENT → FW")


    # ──────────────────────────────────────────────────────────
    # SEND ATTITUDE
    # ──────────────────────────────────────────────────────────

    def send_attitude(self, roll_deg, pitch_deg, compass_yaw_deg, thrust):
        """
        roll_deg        : body roll (used as lateral steering in FW)
        pitch_deg       : body pitch
        compass_yaw_deg : desired heading in compass degrees (0=N, 90=E)
        thrust          : [0, 1]
        """
        msg = AttitudeTarget()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE  |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        # Compass → PX4 ENU yaw
        px4_yaw = 90.0 - compass_yaw_deg
        while px4_yaw >  180.0: px4_yaw -= 360.0
        while px4_yaw < -180.0: px4_yaw += 360.0

        q = quaternion_from_euler(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(px4_yaw)
        )

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        msg.thrust        = thrust

        self.att_pub.publish(msg)


    # ──────────────────────────────────────────────────────────
    # GEOMETRY HELPERS
    # ──────────────────────────────────────────────────────────

    def _bearing_to_target(self):
        """Returns compass bearing (0-360°) from current pos to target."""
        lat1 = math.radians(self.current_lat)
        lon1 = math.radians(self.current_lon)
        lat2 = math.radians(self.target_lat)
        lon2 = math.radians(self.target_lon)
        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = (  math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    @staticmethod
    def _wrap_error(err):
        """Wrap an angle error into [-180, 180]."""
        while err >  180.0: err -= 360.0
        while err < -180.0: err += 360.0
        return err

    def _horizontal_distance(self):
        return haversine(
            (self.current_lat, self.current_lon),
            (self.target_lat,  self.target_lon),
            unit=Unit.METERS
        )

    def _mc_pitch_to_target(self, horizontal_distance):
        """
        Standard quad pitch-toward-target.
        Identical to original gps_strike_controller_hold.py logic.
        Positive pitch = nose forward = fly toward target (quad convention).
        No tailsitter sign flip here — this is pure MC behaviour.
        """
        height_remaining = self.absolute_target_alt - self.current_alt
        safe_dist        = max(horizontal_distance, 0.01)

        pitch_rad = math.atan2(safe_dist, height_remaining)
        return math.degrees(pitch_rad)


    # ──────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────

    def run(self):

        # ── Guard: FCU + altitude init ────────────────────────
        if not self.current_state.connected:
            return
        if self.base_altitude is None:
            return

        # ── Always stream a setpoint so OFFBOARD doesn't drop ─
        self.send_attitude(0.0, 0.0, self.current_yaw, 0.6)

        # ── Manual hold ───────────────────────────────────────
        if self.manual_hold_requested:
            self.target_passed    = True
            self.offboard_disabled = True
            req = SetMode.Request(); req.custom_mode = "AUTO.LOITER"
            self.mode_client.call_async(req)
            self.get_logger().warn("MANUAL HOLD ACTIVATED")
            return

        if self.target_passed:
            self.get_logger().warn("TARGET PASSED – HOLDING")
            return

        # ── OFFBOARD / ARM ────────────────────────────────────
        if not self.offboard_disabled and self.current_state.mode != "OFFBOARD":
            req = SetMode.Request(); req.custom_mode = "OFFBOARD"
            self.mode_client.call_async(req)
            return

        if not self.current_state.armed:
            req = CommandBool.Request(); req.value = True
            self.arm_client.call_async(req)
            return

        # ── GPS guard ─────────────────────────────────────────
        if self.current_lat is None:
            self.get_logger().info("Waiting for GPS...")
            return

        # ── Common geometry ───────────────────────────────────
        relative_alt      = self.current_alt - self.base_altitude
        horizontal_dist   = self._horizontal_distance()
        target_bearing    = self._bearing_to_target()
        
        # ── DYNAMIC HEADING EXTRACTION (The Axis Swap) ────────
        if self.phase in [PHASE_FW_TRANSITION, PHASE_FW_CRUISE]:
            # Gimbal lock shifts the heading to the roll axis.
            # We use current_roll and apply the 90-degree offset to correct it.
            current_heading = (self.current_roll ) % 360.0
        else:
            # MC Hover uses standard Yaw
            current_heading = self.current_yaw

        bearing_error     = self._wrap_error(target_bearing - current_heading)
        now               = self.get_clock().now().nanoseconds * 1e-9

        # ── Distance derivative / pass detection ──────────────
        distance_rate = 0.0
        if self.prev_distance is not None:
            dt = now - self.prev_distance_time
            if dt > 0.0:
                distance_rate = (horizontal_dist - self.prev_distance) / dt

        if (
            not self.target_passed
            and horizontal_dist < self.pass_detection_radius_m
            and self.prev_distance_rate is not None
            and self.prev_distance_rate < 0.0
            and distance_rate > 0.0
        ):
            self.target_passed    = True
            self.offboard_disabled = True
            req = SetMode.Request(); req.custom_mode = "AUTO.LOITER"
            self.mode_client.call_async(req)
            self.get_logger().warn(
                f"TARGET PASSED! "
                f"{self.prev_distance_rate:.1f} → {distance_rate:.1f}"
            )
            return

        self.prev_distance      = horizontal_dist
        self.prev_distance_time = now
        self.prev_distance_rate = distance_rate

        # ══════════════════════════════════════════════════════
        #  PHASE STATE MACHINE
        # ══════════════════════════════════════════════════════

        # ── PHASE: TAKEOFF ────────────────────────────────────
        if self.phase == PHASE_TAKEOFF:

            if relative_alt < self.takeoff_alt_m:

                self.send_attitude(0.0, 0.0, self.current_yaw, 1.0)

                self.get_logger().info(
                    f"[TAKEOFF] rel_alt={relative_alt:.1f} m"
                )
                return

            else:
                self.takeoff_complete = True
                self.phase = PHASE_MC_CLIMB
                self.get_logger().info("TAKEOFF COMPLETE → MC_CLIMB")

        # ── PHASE: MC_CLIMB  (distance > 1000 m) ─────────────
        if self.phase == PHASE_MC_CLIMB:
            # Capture the starting coordinate of the climb phase on the first frame entry
            if self.start_climb_lat is None:
                self.start_climb_lat = self.current_lat
                self.start_climb_lon = self.current_lon

            # Calculate distance covered away from the initial start point
            distance_covered = haversine(
                (self.start_climb_lat, self.start_climb_lon),
                (self.current_lat, self.current_lon),
                unit=Unit.METERS
            )

            # Stay here until the desired target relative altitude is reached
            if relative_alt < self.target_relative_alt:

                # ── Yaw to face target first, then pitch ─────
                if abs(bearing_error) > 10.0:

                    pitch_deg = 0.0
                    thrust    = 0.6

                else:
                    # Calculate safe remaining horizontal window distance (1000m total window)
                    safe_dist = 1000.0 - distance_covered
                    pitch_deg = self._mc_pitch_to_target(safe_dist)
                    thrust    = 1.0

                self.send_attitude(
                    0.0, pitch_deg, target_bearing, thrust
                )

                self.get_logger().info(
                    f"[MC_CLIMB] "
                    f"dist_to_target={horizontal_dist:.0f}m "
                    f"dist_covered={distance_covered:.1f}m "
                    f"pitch={pitch_deg:.1f}° "
                    f"rel_alt={relative_alt:.1f}m / target={self.target_relative_alt:.1f}m"
                )

                return  # stay in MC_CLIMB until target altitude is achieved

            else:
                # Target altitude met -> Switch immediately to fixed-wing transition phase
                self.phase              = PHASE_FW_TRANSITION
                self._trans_start_time  = now
                self._trans_pitch_start = self.current_pitch
                self.get_logger().info(
                    f"Target altitude of {self.target_relative_alt:.1f}m reached! "
                    f"Distance covered: {distance_covered:.1f}m → FW_TRANSITION"
                )

        # ── PHASE: FW_TRANSITION ──────────────────────────────
        if self.phase == PHASE_FW_TRANSITION:
            if not self.vtol_sent:
                self.request_fw_transition()
                self.vtol_sent = True

            pitch_deg = 0 
            fw_roll = 0 

            self.send_attitude(
                fw_roll, pitch_deg, current_heading,
                self.fw_cruise_thrust
            )

            self.get_logger().info(
                f"pitch={pitch_deg:.1f}° "
                f"roll={fw_roll:.1f}° "
                f"dist={horizontal_dist:.0f}m"
            )

            self.phase = PHASE_FW_CRUISE
            self.get_logger().info("FW_TRANSITION COMPLETE → FW_CRUISE")

            return

        
        # ── PHASE: FW_CRUISE ──────────────────────────────────
        if self.phase == PHASE_FW_CRUISE:

            # Recalculate true bearing error using the Gimbal-Lock-free heading!
            fw_bearing_error = self._wrap_error(target_bearing - self.fw_yaw)

            # 1. THE ROLL COMMAND (Bank-to-turn)
            roll_cmd = fw_bearing_error * self.fw_yaw_gain
            roll_cmd = max(-self.fw_yaw_clamp, min(self.fw_yaw_clamp, roll_cmd))

            # 2. THE PITCH PULL (Coordinated Turn)
            coordinated_pitch = self.fw_pitch_deg + (abs(roll_cmd) * 0.25)

            # 3. NEUTRALIZE YAW
            yaw_cmd = self.fw_yaw

            self.send_attitude(
                roll_cmd,                 
                coordinated_pitch,        
                yaw_cmd,                  
                self.fw_cruise_thrust
            )
# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():

    rclpy.init()

    node = GPSStrikeController(

        target_lat         = 40.59190553758221,
        target_lon         = -79.8862427481675,
        target_relative_alt = 750.0,
    )

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
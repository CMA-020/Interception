#!/usr/bin/env python3

import math

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

from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)


# ==========================================================
# FLIGHT PHASES
# ==========================================================

PHASE_CLIMB   = "CLIMB"    # vertical climb at 0.6 thrust up to 200 m
PHASE_CRUISE  = "CRUISE"   # pitch 45 deg forward at 0.8 thrust


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
# NODE
# ==========================================================

class TailsitterLiftTest(Node):

    def __init__(self):

        super().__init__(
            "tailsitter_lift_test"
        )

        # ==================================================
        # FCU STATE
        # ==================================================

        self.current_state  = State()
        self.current_yaw    = 0.0
        self.current_alt_m  = 0.0      # relative altitude in metres

        # ==================================================
        # FLIGHT PHASE
        # ==================================================

        self.phase = PHASE_CLIMB

        # Target altitude before transitioning to cruise
        self.climb_target_m = 100.0    # metres

        # ==================================================
        # PHASE PARAMETERS
        # ==================================================

        #  CLIMB  – wings level, straight up
        self.climb_thrust   = 0.6
        self.climb_pitch    = 0.0      # degrees  (flat / vertical for tailsitter)
        self.climb_roll     = 0.0

        #  CRUISE – nose pitched 45 deg forward, more thrust
        self.cruise_thrust  = 0.8
        self.cruise_pitch   = 80.0    # degrees  (negative = nose down / forward)
        self.cruise_roll    = 0.0

        # ==================================================
        # PUB
        # ==================================================

        self.pub = self.create_publisher(

            AttitudeTarget,

            "/mavros/setpoint_raw/attitude",

            10
        )

        # ==================================================
        # QoS
        # ==================================================

        qos = QoSProfile(

            reliability=ReliabilityPolicy.BEST_EFFORT,

            history=HistoryPolicy.KEEP_LAST,

            depth=10
        )

        # ==================================================
        # SUBS
        # ==================================================

        self.state_sub = self.create_subscription(

            State,

            "/mavros/state",

            self.state_cb,

            10
        )

        self.imu_sub = self.create_subscription(

            Imu,

            "/mavros/imu/data",

            self.imu_cb,

            qos
        )

        # mavros_msgs/Altitude  →  msg.relative = AGL height above home
        # If this stays 0, run:  ros2 topic echo /mavros/altitude
        # and check which field is non-zero (local / amsl / relative / terrain)
        self.alt_sub = self.create_subscription(

            Altitude,

            "/mavros/altitude",

            self.alt_cb,

            qos
        )

        # FALLBACK – swap the two lines below if rel_alt Float64 is easier
        # self.alt_sub = self.create_subscription(
        #     Float64, "/mavros/global_position/rel_alt",
        #     lambda msg: setattr(self, "current_alt_m", msg.data), qos
        # )

        # ==================================================
        # SERVICES
        # ==================================================

        self.arm_client = self.create_client(

            CommandBool,

            "/mavros/cmd/arming"
        )

        self.mode_client = self.create_client(

            SetMode,

            "/mavros/set_mode"
        )

        # ==================================================
        # TIMER  (20 Hz)
        # ==================================================

        self.timer = self.create_timer(

            0.05,

            self.run
        )

        self.get_logger().info(
            "TAILSITTER LIFT TEST NODE STARTED"
        )

        self.get_logger().info(
            f"Phase 1 : CLIMB  — thrust={self.climb_thrust}  "
            f"pitch={self.climb_pitch} deg  "
            f"target alt={self.climb_target_m} m"
        )

        self.get_logger().info(
            f"Phase 2 : CRUISE — thrust={self.cruise_thrust}  "
            f"pitch={self.cruise_pitch} deg"
        )

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def state_cb(self, msg):
        self.current_state = msg

    # ----------------------------------------------------------

    def imu_cb(self, msg):

        q = msg.orientation

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

        yaw_deg_enu = math.degrees(yaw_rad)

        self.current_yaw = (90.0 - yaw_deg_enu) % 360.0

    # ----------------------------------------------------------

    def alt_cb(self, msg):
        # mavros_msgs/Altitude fields:
        #   msg.relative  → height above home point (AGL)  ← we use this
        #   msg.local     → EKF local Z
        #   msg.amsl      → above mean sea level
        self.current_alt_m = msg.relative

    # ==========================================================
    # SEND ATTITUDE
    # ==========================================================

    def send_attitude(

        self,

        roll_deg,

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

        px4_yaw_deg = (90.0 - compass_yaw_deg)

        while px4_yaw_deg >  180.0:
            px4_yaw_deg -= 360.0

        while px4_yaw_deg < -180.0:
            px4_yaw_deg += 360.0

        roll  = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw   = math.radians(px4_yaw_deg)

        q = quaternion_from_euler(roll, pitch, yaw)

        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        msg.thrust = thrust

        self.pub.publish(msg)

    # ==========================================================
    # PHASE MANAGER
    # ==========================================================

    def update_phase(self):
        """
        Transition CLIMB → CRUISE once the target altitude is reached.
        Add further transitions here as needed.
        """

        if (
            self.phase == PHASE_CLIMB and
            self.current_alt_m >= self.climb_target_m
        ):

            self.phase = PHASE_CRUISE

            self.get_logger().info(
                f"Altitude {self.current_alt_m:.1f} m reached — "
                f"transitioning to CRUISE  "
                f"(pitch={self.cruise_pitch} deg, "
                f"thrust={self.cruise_thrust})"
            )

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    def run(self):

        if not self.current_state.connected:

            self.get_logger().info(
                "Waiting for FCU..."
            )

            return

        # --------------------------------------------------
        # DETERMINE SETPOINT FOR CURRENT PHASE
        # --------------------------------------------------

        self.update_phase()

        if self.phase == PHASE_CLIMB:

            roll   = self.climb_roll
            pitch  = self.climb_pitch
            thrust = self.climb_thrust

        else:   # PHASE_CRUISE

            roll   = self.cruise_roll
            pitch  = self.cruise_pitch
            thrust = self.cruise_thrust

        # --------------------------------------------------
        # ALWAYS STREAM SETPOINTS  (required before OFFBOARD)
        # --------------------------------------------------

        self.send_attitude(

            roll_deg        = roll,
            pitch_deg       = pitch,
            compass_yaw_deg = self.current_yaw,
            thrust          = thrust
        )

        # --------------------------------------------------
        # OFFBOARD MODE
        # --------------------------------------------------

        if self.current_state.mode != "OFFBOARD":

            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"

            self.mode_client.call_async(req)

            self.get_logger().info(
                "Trying OFFBOARD..."
            )

            return

        # --------------------------------------------------
        # ARM
        # --------------------------------------------------

        if not self.current_state.armed:

            req = CommandBool.Request()
            req.value = True

            self.arm_client.call_async(req)

            self.get_logger().info(
                "Trying ARM..."
            )

            return

        # --------------------------------------------------
        # STATUS LOG
        # --------------------------------------------------

        self.get_logger().info(

            f"[{self.phase}]  "
            f"alt={self.current_alt_m:.1f} m  "
            f"pitch={pitch:.1f} deg  "
            f"thrust={thrust:.2f}  "
            f"yaw={self.current_yaw:.1f} deg"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    node = TailsitterLiftTest()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()
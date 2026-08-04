import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix

import math
import threading
import time

# Gazebo Transport Imports -> Reverted back to Pose_V
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy


class CubeGPSConverter(Node):

    def __init__(self):

        super().__init__('cube_gps_converter')

        # ======================================================
        # QOS
        # ======================================================

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ======================================================
        # PUBLIC VARIABLES
        # ======================================================

        self.converted_lat = None
        self.converted_lon = None
        self.converted_height = None

        # ======================================================
        # GPS ORIGIN
        # ======================================================

        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

        # ======================================================
        # GPS SUBSCRIBER
        # ======================================================

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_cb,
            qos
        )

        # ======================================================
        # GAZEBO TRANSPORT SUBSCRIBER
        # ======================================================
        
        self.gz_node = GzNode()
        
        # Start Gazebo subscriber setup in a background thread
        self.gazebo_thread = threading.Thread(target=self.init_gazebo_transport, daemon=True)
        self.gazebo_thread.start()

    # ==========================================================
    # GPS CALLBACK
    # ==========================================================

    def gps_cb(self, msg):

        if self.origin_lat is None:

            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.origin_alt = msg.altitude

            self.get_logger().info(
                f"GPS origin initialized: Lat={self.origin_lat}, Lon={self.origin_lon}"
            )

    # ==========================================================
    # GAZEBO TRANSPORT INITIALIZER & CALLBACK
    # ==========================================================

    def init_gazebo_transport(self):
        """Initializes the native Gazebo Transport node."""
        gazebo_topic = '/model/floating_cube/pose'
        
        # Subscribe using GzPoseV instead of GzPose
        if self.gz_node.subscribe(GzPoseV, gazebo_topic, self.gazebo_pose_cb):
            self.get_logger().info(f"Successfully subscribed directly to Gazebo topic: {gazebo_topic}")
        else:
            self.get_logger().error(f"Failed to subscribe to Gazebo topic: {gazebo_topic}")

    def gazebo_pose_cb(self, msg: GzPoseV):
        """Native Gazebo callback parsed directly from the transport layer."""
        
        # Wait until GPS origin is received from MAVROS
        if self.origin_lat is None:
            return

        # Safely loop through the vector array to find the cube's position data
        for pose in msg.pose:
            if pose.name != "floating_cube":
                continue

            cube_x = pose.position.x
            cube_y = pose.position.y
            cube_z = pose.position.z

            # ==================================================
            # XYZ -> GPS
            # ==================================================

            meters_per_deg_lat = 111111.0

            meters_per_deg_lon = (
                111111.0 *
                math.cos(
                    math.radians(
                        self.origin_lat
                    )
                )
            )

            delta_lat = (
                cube_y /
                meters_per_deg_lat
            )

            delta_lon = (
                cube_x /
                meters_per_deg_lon
            )

            self.converted_lat = (
                self.origin_lat +
                delta_lat
            )

            self.converted_lon = (
                self.origin_lon +
                delta_lon
            )

            self.converted_height = (
                cube_z
            )

            # Print immediately upon receiving and calculating the data
            print(
                f"LAT: {self.converted_lat:.8f}, "
                f"LON: {self.converted_lon:.8f}, "
                f"ALT: {self.converted_height:.2f}", 
                end="\r", 
                flush=True
            )
            break


# ==========================================================
# STANDALONE RUN
# ==========================================================

def main():

    rclpy.init()

    node = CubeGPSConverter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from tf2_msgs.msg import TFMessage

import math

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
        # TF SUBSCRIBER
        # ======================================================

        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_cb,
            qos
        )

    # ==========================================================
    # GPS CALLBACK
    # ==========================================================

    def gps_cb(self, msg):

        if self.origin_lat is None:

            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.origin_alt = msg.altitude

            self.get_logger().info(
                "GPS origin initialized"
            )

    # ==========================================================
    # TF CALLBACK
    # ==========================================================

    def tf_cb(self, msg):

        # wait until gps is received
        if self.origin_lat is None:
            return

        for t in msg.transforms:

            # only use moving_cube
            if t.child_frame_id != "moving_cube":
                continue

            cube_x = t.transform.translation.x
            cube_y = t.transform.translation.y
            cube_z = t.transform.translation.z

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
                # self.origin_alt +
                cube_z
            )


# ==========================================================
# OPTIONAL STANDALONE RUN
# ==========================================================

def main():

    rclpy.init()

    node = CubeGPSConverter()

    while rclpy.ok():

        rclpy.spin_once(node)

        if (
            node.converted_lat is not None
        ):

            print(
                f"LAT: {node.converted_lat:.8f}, "
                f"LON: {node.converted_lon:.8f}, "
                f"ALT: {node.converted_height:.2f}"
            )

    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np


class stream_viewer(Node):

    def __init__(
        self,
        topic_name='/rgb_compressed',
        window_name='RGB Camera',
        width=640,
        height=480
    ):

        super().__init__('stream_viewer')

        self.topic_name = topic_name
        self.window_name = window_name

        self.width = width
        self.height = height

        # ==========================================
        # SUBSCRIBER
        # ==========================================

        self.image_subscription = self.create_subscription(
            CompressedImage,
            self.topic_name,
            self.listener_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Subscribed to {self.topic_name}"
        )

    # ==============================================
    # IMAGE CALLBACK
    # ==============================================

    def listener_callback(self, msg):

        try:

            # Convert compressed image
            np_arr = np.frombuffer(
                msg.data,
                np.uint8
            )

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                return

            # Resize image
            frame = cv2.resize(
                frame,
                (self.width, self.height)
            )

            # Display image
            cv2.imshow(
                self.window_name,
                frame
            )

            cv2.waitKey(1)

        except Exception as e:

            self.get_logger().error(
                f"Image callback error: {str(e)}"
            )


# ====================================================
# OPTIONAL STANDALONE RUN
# ====================================================

def main(args=None):

    rclpy.init(args=args)

    node = stream_viewer()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        print("\nShutting down stream viewer")

    finally:

        node.destroy_node()

        rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


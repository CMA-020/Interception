import rclpy

from rclpy.executors import MultiThreadedExecutor

# from gps_strike_controller_hold import GPSStrikeController
from gps_tracker_controller_og import GPSStrikeController
# IMPORT STREAM VIEWER
from stream_viewer import stream_viewer


# ======================================================
# HARDCODED TARGET
# ======================================================


TARGET_LAT = 40.56938018
TARGET_LON = -79.87440573
TARGET_ALT = 750.0


def main():

    rclpy.init()

    # ======================================================
    # STRIKE CONTROLLER
    # ======================================================

    controller = GPSStrikeController(

        target_lat=TARGET_LAT,

        target_lon=TARGET_LON,

        target_relative_alt=TARGET_ALT
    )

    # ======================================================
    # CAMERA VIEWER NODE
    # ======================================================

    # viewer = stream_viewer(
    #     topic_name='/rgb_compressed',
    #     window_name='RGB Camera',
    #     width=640,
    #     height=480
    # )

    # ======================================================
    # EXECUTOR
    # ======================================================

    executor = MultiThreadedExecutor()

    executor.add_node(controller)

    # executor.add_node(viewer)

    # ======================================================
    # SPIN
    # ======================================================

    try:

        executor.spin()

    except KeyboardInterrupt:

        print("\nShutting down...")

    finally:

        controller.destroy_node()

        # viewer.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
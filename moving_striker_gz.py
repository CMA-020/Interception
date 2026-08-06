import rclpy

from rclpy.executors import MultiThreadedExecutor

from gps_tracker_logger import GPSStrikeController
# from gps_strike_controller_hold import GPSStrikeController

# MODIFIED: Import the updated Gazebo transport-based converter class
from target_gps_converter_gz import CubeGPSConverter

# IMPORT STREAM VIEWER
from stream_viewer import stream_viewer


def main():

    rclpy.init()

    # ======================================================
    # NODES
    # ======================================================

    gps_converter = CubeGPSConverter()

    controller = GPSStrikeController(

        target_lat=0.0,
        target_lon=0.0,

        target_relative_alt=0.0
    )

    # # ======================================================
    # # CAMERA VIEWER NODE
    # # ======================================================

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

    executor.add_node(gps_converter)

    executor.add_node(controller)

    # executor.add_node(viewer)

    # ======================================================
    # TARGET UPDATE TIMER
    # ======================================================

    def update_target():

        if (

            gps_converter.converted_lat is not None and

            gps_converter.converted_lon is not None and

            gps_converter.converted_height is not None
        ):

            controller.set_target(

                gps_converter.converted_lat,

                gps_converter.converted_lon
            )

            controller.target_alt = (
                gps_converter.converted_height
            )

            if controller.base_altitude is not None:

                controller.absolute_target_alt = (

                    controller.base_altitude +

                    controller.target_alt
                )

    controller.create_timer(
        0.1,
        update_target
    )

    # ======================================================
    # SPIN
    # ======================================================

    try:

        executor.spin()

    except KeyboardInterrupt:

        print("\nShutting down...")

    finally:

        controller.destroy_node()

        gps_converter.destroy_node()

        # viewer.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
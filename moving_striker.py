import rclpy

from rclpy.executors import MultiThreadedExecutor

from gps_strike_controller import GPSStrikeController
from target_gps_converter import CubeGPSConverter


def main():

    rclpy.init()

    gps_converter = CubeGPSConverter()

    controller = GPSStrikeController(

        target_lat=0.0,
        target_lon=0.0,

        target_relative_alt=0.0
    )

    # ======================================================
    # EXECUTOR
    # ======================================================

    executor = MultiThreadedExecutor()

    executor.add_node(gps_converter)
    executor.add_node(controller)

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

    try:

        executor.spin()

    except KeyboardInterrupt:

        pass

    controller.destroy_node()
    gps_converter.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
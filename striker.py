import rclpy
import threading
import time

from gps_strike_controller import GPSStrikeController


# ==========================================================
# TARGET PROVIDER
# ==========================================================

# Replace this with:
# - socket input
# - vision tracking
# - another ROS topic
# - API
# - telemetry feed
#
# This function should ALWAYS return
# latest target lat/lon

def get_latest_target():

    # EXAMPLE:
    # moving target

    base_lat = 40.59190553758221
    base_lon = -79.8862427481675

    t = time.time()

    moving_lat = base_lat + (0.00001 * (t % 20))
    moving_lon = base_lon + (0.00001 * (t % 20))

    return moving_lat, moving_lon


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    # ======================================================
    # INITIAL TARGET
    # ======================================================

    initial_lat, initial_lon = get_latest_target()

    controller = GPSStrikeController(

        target_lat=initial_lat,
        target_lon=initial_lon,

        target_relative_alt=750.0
    )

    # ======================================================
    # ROS SPIN THREAD
    # ======================================================

    spin_thread = threading.Thread(

        target=rclpy.spin,
        args=(controller,),
        daemon=True
    )

    spin_thread.start()

    # ======================================================
    # TARGET UPDATE LOOP
    # ======================================================

    try:

        while rclpy.ok():

            target_lat, target_lon = get_latest_target()

            controller.set_target(
                target_lat,
                target_lon
            )

            controller.get_logger().info(

                f"Updated Target -> "
                f"{target_lat}, {target_lon}"
            )

            # update rate
            time.sleep(0.1)

    except KeyboardInterrupt:

        pass

    controller.destroy_node()

    rclpy.shutdown()


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == '__main__':

    main()
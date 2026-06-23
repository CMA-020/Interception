#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "user_control.hpp"
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);

    init_vs_pipeline();

    auto node = rclcpp::Node::make_shared("tracker_init_node");

    auto track_coords_sub = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/track_coords",
        10,
        [](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
            if (msg->data.size() < 4) {
                printf("track_coords: not enough values\n");
                return;
            }

            float tr_val1 = msg->data[0];
            float tr_val2 = msg->data[1];
            float tr_val3 = msg->data[2];
            float tr_val4 = msg->data[3];

            // Reload-only signal
            if (tr_val1 == -20.0f && tr_val2 == -20.0f &&
                tr_val3 == -20.0f && tr_val4 == -20.0f) {
                printf("Received reload signal (-20,-20,-20,-20), reloading tracker\n");
                load_c_tracker();
                return;
            }

            // Normal init
            boundingBoxXywh box = {tr_val1, tr_val2, tr_val3, tr_val4};
            printf("tracker init\n");
            load_c_tracker();
            sleep(2);
            // set_tracker_threshold_score(0.4, 0.4);
            init_tracker_with_bbox(box);
        }
    );

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
#!/usr/bin/env python3

import time
import cv2
import numpy as np

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image


def image_callback(msg):
    print(f"Received {msg.width}x{msg.height}")

    img = np.frombuffer(msg.data, dtype=np.uint8)

    # RGB8 image
    img = img.reshape((msg.height, msg.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Gazebo Camera", img)
    cv2.waitKey(1)


node = Node()

node.subscribe(
    Image,
    "/camera/image",
    image_callback,
)

print("Subscribed!")

while True:
    time.sleep(0.1)
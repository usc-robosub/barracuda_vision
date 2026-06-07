#!/usr/bin/env python3
"""Test publisher: repeatedly publishes a static image to the YOLO input topic."""

import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2


class ImageTestPublisher(Node):
    def __init__(self):
        super().__init__('image_test_publisher')

        default_image = os.path.join(
            get_package_share_directory('barracuda_vision'),
            'images', 'shark.png')

        self.declare_parameter('image_path', default_image)
        self.declare_parameter('topic', 'yolo_input_image')
        self.declare_parameter('rate', 10.0)

        self.image_path = self.get_parameter(
            'image_path').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        rate = self.get_parameter('rate').get_parameter_value().double_value

        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0 / rate, self.publish_image)

        self.get_logger().info(
            f'Publishing "{self.image_path}" to "{topic}" at {rate} Hz')

    def publish_image(self):
        cv_image = cv2.imread(self.image_path)
        if cv_image is None:
            self.get_logger().error(
                f'Failed to read image: {self.image_path}')
            return
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.get_logger().info('Publishing image')
        self.pub.publish(ros_image)


def main(args=None):
    rclpy.init(args=args)
    node = ImageTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

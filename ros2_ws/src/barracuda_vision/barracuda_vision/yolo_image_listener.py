#!/usr/bin/env python3
"""YOLO object-detection node for ROS 2.

Subscribes to an image topic, runs a local Ultralytics YOLO model on each
frame, and publishes the detection count, the bounding boxes as a
vision_msgs/Detection2DArray, and an annotated image.
"""

import glob
import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import torch


def _set_bbox_center(bbox, cx, cy):
    """Set a BoundingBox2D center, tolerant of vision_msgs API versions.

    Older vision_msgs (<=4.0, geometry_msgs/Pose2D) exposes ``center.x``;
    newer releases (vision_msgs/Pose2D with Point2D) expose
    ``center.position.x``.
    """
    center = bbox.center
    if hasattr(center, 'position'):
        center.position.x = cx
        center.position.y = cy
    else:
        center.x = cx
        center.y = cy


class YoloImageListener(Node):
    def __init__(self):
        super().__init__('yolo_image_listener')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_dir', '/models')
        self.declare_parameter('input_topic', 'yolo_input_image')

        model_path = self._resolve_model_path()
        self.get_logger().info(f'Using model file: {model_path}')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()

        input_topic = self.get_parameter(
            'input_topic').get_parameter_value().string_value

        self.sub = self.create_subscription(
            Image, input_topic, self.infer, 10)
        self.pub_object_detector = self.create_publisher(
            Int32, 'object_detector', 10)
        self.pub_detections = self.create_publisher(
            Detection2DArray, 'detections', 10)
        self.pub_detection_image = self.create_publisher(
            Image, 'detection_image', 10)

        self.get_logger().info(
            f'yolo_image_listener ready, subscribed to "{input_topic}"')

    def _resolve_model_path(self):
        """Return the model file to load.

        Prefers the explicit ``model_path`` parameter; otherwise scans
        ``model_dir`` and requires exactly one model file to be present.
        """
        model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        if model_path:
            if not os.path.isfile(model_path):
                self.get_logger().fatal(
                    f'model_path does not exist: {model_path}')
                raise SystemExit(1)
            return model_path

        models_dir = self.get_parameter(
            'model_dir').get_parameter_value().string_value
        model_files = [
            f for f in glob.glob(os.path.join(models_dir, '*'))
            if os.path.isfile(f)
        ]
        if not model_files:
            self.get_logger().fatal(
                f'No model file found in {models_dir}. Shutting down.')
            raise SystemExit(1)
        if len(model_files) > 1:
            self.get_logger().fatal(
                f'Multiple model files found in {models_dir}. '
                'Only one model should be present. Shutting down.')
            raise SystemExit(1)
        return model_files[0]

    def infer(self, data):
        """Run inference on an incoming image using the local model."""
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        with torch.no_grad():
            outputs = self.model(image)
        result = outputs[0]
        self.get_logger().info(
            f'Local model inference: {len(result)} detections')
        self.process_result(data, image, result)

    def process_result(self, data, image, result):
        """Convert results to ROS messages and publish them."""
        detections = sv.Detections.from_ultralytics(result)
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_id = detections.class_id
        names = getattr(result, 'names', {}) or {}

        det_array = Detection2DArray()
        det_array.header = data.header

        for i in range(len(detections)):
            xmin, ymin, xmax, ymax = (float(v) for v in xyxy[i])

            det = Detection2D()
            det.header = data.header
            _set_bbox_center(
                det.bbox, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
            det.bbox.size_x = xmax - xmin
            det.bbox.size_y = ymax - ymin

            cid = int(class_id[i])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(names.get(cid, cid))
            hyp.hypothesis.score = float(confidence[i])
            det.results.append(hyp)

            det_array.detections.append(det)

        # Annotate the frame for visualization.
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), detections=detections)
        detection_image = self.bridge.cv2_to_imgmsg(
            annotated_frame, encoding='bgr8')
        detection_image.header = data.header

        count = Int32()
        count.data = len(detections)

        self.pub_object_detector.publish(count)
        self.pub_detections.publish(det_array)
        self.pub_detection_image.publish(detection_image)


def main(args=None):
    rclpy.init(args=args)
    node = YoloImageListener()
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

#!/usr/bin/env python3
"""YOLO object-detection node for ROS 2.

Subscribes to an image topic and a depth topic, runs a local Ultralytics YOLO
model on each RGB frame, and publishes the detection count, the bounding boxes
as a vision_msgs/Detection2DArray (with robust median depth in position.z), and
an annotated image.

Improvement 1 – Mask-Based / Bbox Median Depth
-----------------------------------------------
Instead of reading a single center pixel (which lands on background water for
thin pipes or hollow gates), we collect every finite, positive depth pixel
inside the bounding box and return the **median**.  This is robust to:
  * holes in disparity maps (stereo occlusion)
  * specular reflections that saturate to inf/NaN
  * thin pipes whose center pixel hits water behind them
  * hollow gate frames whose interior is open water

The depth result is written to Detection2D.results[0].pose.pose.position.z
so downstream consumers (factor graph, behavior tree) can lift the 2D bbox to
a 3D point without any other change.
"""

import glob
import os
import threading

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import CameraInfo, Image
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_bbox_center(bbox, cx, cy):
    center = bbox.center
    if hasattr(center, 'position'):
        center.position.x = cx
        center.position.y = cy
    else:
        center.x = cx
        center.y = cy


def sample_depth_bbox(depth_image: np.ndarray,
                      xmin: int, ymin: int,
                      xmax: int, ymax: int,
                      min_valid: int = 5) -> float | None:
    """Return the median depth (metres) of valid pixels inside a bounding box."""
    h, w = depth_image.shape[:2]
    x0 = max(0, int(xmin))
    y0 = max(0, int(ymin))
    x1 = min(w, int(xmax))
    y1 = min(h, int(ymax))

    if x1 <= x0 or y1 <= y0:
        return None

    roi = depth_image[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]

    if len(valid) < min_valid:
        return None

    return float(np.median(valid))

def sample_depth_border(depth_image: np.ndarray,
                        xmin: int, ymin: int,
                        xmax: int, ymax: int,
                        border: int = 8,
                        min_valid: int = 5) -> float | None:
    """Return median depth of pixels along the bbox border only.

    Better than full-bbox median for hollow objects (gate) where the interior
    is open water.  Samples a ring of `border` pixels wide around the edge.
    """
    h, w = depth_image.shape[:2]
    x0 = max(0, int(xmin));  y0 = max(0, int(ymin))
    x1 = min(w, int(xmax));  y1 = min(h, int(ymax))
    if x1 <= x0 or y1 <= y0:
        return None

    roi = depth_image[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    b = min(border, rh // 2, rw // 2)
    if b < 1:
        return sample_depth_bbox(depth_image, xmin, ymin, xmax, ymax, min_valid)

    mask = np.zeros((rh, rw), dtype=bool)
    mask[:b, :]  = True   # top strip
    mask[-b:, :] = True   # bottom strip
    mask[:, :b]  = True   # left strip
    mask[:, -b:] = True   # right strip

    border_px = roi[mask]
    valid = border_px[np.isfinite(border_px) & (border_px > 0.0)]
    if len(valid) < min_valid:
        return None
    return float(np.median(valid))


def sample_depth_pipe(depth_image: np.ndarray,
                      xmin: int, ymin: int,
                      xmax: int, ymax: int,
                      min_valid: int = 5) -> float | None:
    """Return median depth for thin pipe objects.

    Samples a narrow horizontal strip through the bbox centre rather than the
    full bbox, reducing contamination from background water that bleeds through
    a thin pipe.
    """
    h, w = depth_image.shape[:2]
    x0 = max(0, int(xmin));  x1 = min(w, int(xmax))
    cy = int((ymin + ymax) / 2)
    strip_h = max(1, int((ymax - ymin) * 0.2))
    y0 = max(0, cy - strip_h // 2)
    y1 = min(h, cy + strip_h // 2 + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = depth_image[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if len(valid) < min_valid:
        return None
    return float(np.median(valid))



# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def estimate_orientation(
    depth_image: np.ndarray,
    xmin: int, ymin: int,
    xmax: int, ymax: int,
    fx: float, fy: float,
    cx: float, cy: float,
    min_valid: int = 5,
):
    """Estimate object orientation by fitting a plane to 9 sample points.

    Samples depth at 9 positions (4 corners, 4 edge midpoints, 1 centre),
    projects each to 3D using camera intrinsics, fits a plane via SVD, and
    returns the yaw angle (rotation about Y axis) as a quaternion [x,y,z,w].
    Returns None if fewer than min_valid points have valid depth.
    """
    h, w = depth_image.shape[:2]
    x0, x1 = max(0, int(xmin)), min(w - 1, int(xmax))
    y0, y1 = max(0, int(ymin)), min(h - 1, int(ymax))
    xm = (x0 + x1) // 2
    ym = (y0 + y1) // 2

    sample_uvs = [
        (x0, y0), (xm, y0), (x1, y0),
        (x0, ym), (xm, ym), (x1, ym),
        (x0, y1), (xm, y1), (x1, y1),
    ]

    points = []
    for u, v in sample_uvs:
        z = float(depth_image[v, u])
        if not (np.isfinite(z) and z > 0.0):
            continue
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        points.append([X, Y, z])

    if len(points) < min_valid:
        return None

    pts = np.array(points)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]  # plane normal (smallest singular value)

    # ensure normal points toward camera (positive Z component)
    if normal[2] < 0:
        normal = -normal

    # yaw = rotation of normal projected onto XZ plane
    yaw = float(np.arctan2(normal[0], normal[2]))

    # convert yaw to quaternion [x, y, z, w]
    r = Rotation.from_euler('y', yaw)
    qx, qy, qz, qw = r.as_quat()
    return float(qx), float(qy), float(qz), float(qw)

class YoloImageListener(Node):
    def __init__(self):
        super().__init__('yolo_image_listener')

        self.declare_parameter('model_path', '')
        self.declare_parameter('model_dir', '/models')
        self.declare_parameter('input_topic', 'yolo_input_image')
        self.declare_parameter('depth_topic', 'zed/depth/depth_registered')
        self.declare_parameter('depth_sync_slop', 0.1)
        self.declare_parameter('depth_min_valid_pixels', 5)

        model_path = self._resolve_model_path()
        self.get_logger().info(f'Using model file: {model_path}')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        self._depth_lock = threading.Lock()
        self._latest_depth: np.ndarray | None = None

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        depth_sync_slop = self.get_parameter('depth_sync_slop').get_parameter_value().double_value

        rgb_sub = Subscriber(self, Image, input_topic)
        depth_sub = Subscriber(self, Image, depth_topic)

        self._sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=10,
            slop=depth_sync_slop,
        )
        self._sync.registerCallback(self._synced_callback)

        self.pub_object_detector = self.create_publisher(Int32, 'object_detector', 10)
        self.pub_detections = self.create_publisher(Detection2DArray, 'detections', 10)
        self.pub_detection_image = self.create_publisher(Image, 'detection_image', 10)

        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._cam_info_lock = threading.Lock()
        self.create_subscription(
            CameraInfo,
            'zed/rgb/camera_info',
            self._camera_info_callback,
            10,
        )

                self.get_logger().info(
            f'yolo_image_listener ready — RGB: "{input_topic}", Depth: "{depth_topic}"')

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        with self._cam_info_lock:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]

        def _resolve_model_path(self):
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if model_path:
            if not os.path.isfile(model_path):
                self.get_logger().fatal(f'model_path does not exist: {model_path}')
                raise SystemExit(1)
            return model_path

        models_dir = self.get_parameter('model_dir').get_parameter_value().string_value
        model_files = [f for f in glob.glob(os.path.join(models_dir, '*')) if os.path.isfile(f)]
        if not model_files:
            self.get_logger().fatal(f'No model file found in {models_dir}. Shutting down.')
            raise SystemExit(1)
        if len(model_files) > 1:
            self.get_logger().fatal(
                f'Multiple model files found in {models_dir}. Only one model should be present. Shutting down.')
            raise SystemExit(1)
        return model_files[0]

    def _synced_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            depth_np = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough').astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'Failed to decode depth image: {e}')
            depth_np = None

        image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        with torch.no_grad():
            outputs = self.model(image)
        result = outputs[0]

        self.get_logger().info(f'Local model inference: {len(result)} detections')
        self.process_result(rgb_msg, image, result, depth_np)

    def process_result(self, data, image, result, depth_np):
        detections = sv.Detections.from_ultralytics(result)
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_id = detections.class_id
        names = getattr(result, 'names', {}) or {}

        min_valid_px = self.get_parameter('depth_min_valid_pixels').get_parameter_value().integer_value

        det_array = Detection2DArray()
        det_array.header = data.header

        for i in range(len(detections)):
            xmin, ymin, xmax, ymax = (float(v) for v in xyxy[i])

            det = Detection2D()
            det.header = data.header
            _set_bbox_center(det.bbox, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
            det.bbox.size_x = xmax - xmin
            det.bbox.size_y = ymax - ymin

            cid = int(class_id[i])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(names.get(cid, cid))
            hyp.hypothesis.score = float(confidence[i])

            if depth_np is not None:
                class_name = hyp.hypothesis.class_id
                if class_name == 'gate':
                    z = sample_depth_border(
                        depth_np,
                        int(xmin), int(ymin), int(xmax), int(ymax),
                        min_valid=min_valid_px,
                    )
                elif class_name in ('red_pipe', 'white_pipe'):
                    z = sample_depth_pipe(
                        depth_np,
                        int(xmin), int(ymin), int(xmax), int(ymax),
                        min_valid=min_valid_px,
                    )
                else:
                    z = sample_depth_bbox(
                        depth_np,
                        int(xmin), int(ymin), int(xmax), int(ymax),
                        min_valid=min_valid_px,
                    )
                if z is not None:
                    hyp.pose.pose.position.z = z

                    # impr3: orientation estimation (gate only for now)
                    if class_name == 'gate':
                        with self._cam_info_lock:
                            fx, fy, cx, cy = self._fx, self._fy, self._cx, self._cy
                        if fx is not None:
                            ori = estimate_orientation(
                                depth_np,
                                int(xmin), int(ymin), int(xmax), int(ymax),
                                fx=fx, fy=fy, cx=cx, cy=cy,
                                min_valid=min_valid_px,
                            )
                            if ori is not None:
                                qx, qy, qz, qw = ori
                                hyp.pose.pose.orientation.x = qx
                                hyp.pose.pose.orientation.y = qy
                                hyp.pose.pose.orientation.z = qz
                                hyp.pose.pose.orientation.w = qw
                else:
                    self.get_logger().warn(
                        f'Detection {i} ({hyp.hypothesis.class_id}): '
                        'not enough valid depth pixels in bbox — z left as 0')

            det.results.append(hyp)
            det_array.detections.append(det)

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        detection_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
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

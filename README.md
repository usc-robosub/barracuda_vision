# barracuda-vision

ROS 2 (Humble) YOLO object-detection node. It subscribes to an image topic,
runs a local [Ultralytics](https://docs.ultralytics.com/) YOLO model on each
frame, and publishes:

| Topic              | Type                              | Description                       |
| ------------------ | --------------------------------- | --------------------------------- |
| `object_detector`  | `std_msgs/Int32`                  | Number of detections in the frame |
| `detections`       | `vision_msgs/Detection2DArray`    | Bounding boxes, class, confidence |
| `detection_image`  | `sensor_msgs/Image`               | Input frame annotated with boxes  |

It subscribes to `yolo_input_image` (`sensor_msgs/Image`).

## Running with Docker

### Step 1: Prepare the Model

Place your YOLO model file (e.g., `yolov8n.pt`) in a directory on your host,
such as `/localmodels`. **Only one model file should be present** in this
directory (the node refuses to start otherwise).

### Step 2: Build and Run the Docker Container

The provided `Dockerfile` and `docker-compose.yaml` build the workspace with
`colcon` and launch the node. Mount your model directory with the `MODEL_PATH`
environment variable (defaults to `/localmodels`); it is mounted to `/models`
inside the container, where the node looks for the model by default.

```bash
MODEL_PATH=/localmodels docker compose up --build
```

### Step 3: Publish a Test Image

In another shell, exec into the container and run the test publisher, which
publishes a static image to `yolo_input_image`:

```bash
docker exec -it barracuda-vision bash
source /opt/barracuda-vision/ros2_ws/install/setup.bash
ros2 run barracuda_vision test_pub_image
```

### Step 4: View the Results

```bash
ros2 topic echo /detections          # bounding boxes
ros2 topic echo /object_detector     # detection count
rqt                                  # view /detection_image (Image View plugin)
```

## Building Locally (without Docker)

Requires ROS 2 Humble plus `ros-humble-cv-bridge` and `ros-humble-vision-msgs`,
and the Python deps in `requirements.txt`.

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Launch the detector (override model location as needed)
ros2 launch barracuda_vision barracuda_vision.launch.py model_dir:=/path/to/models
# or point directly at a model file:
ros2 launch barracuda_vision barracuda_vision.launch.py model_path:=/path/to/model.pt
```

### Parameters

`yolo_image_listener`:

- `model_path` (string, default `""`): explicit path to a `.pt` model file.
- `model_dir` (string, default `/models`): directory scanned for exactly one
  model file when `model_path` is not set.
- `input_topic` (string, default `yolo_input_image`): image topic to subscribe to.

`test_pub_image`:

- `image_path` (string): image to publish (defaults to the bundled `shark.png`).
- `topic` (string, default `yolo_input_image`).
- `rate` (double, default `10.0`): publish rate in Hz.

---

**Note:**
- Only local model inference is supported (no Roboflow / remote server).
- The node exits if zero or more than one model file is found in `model_dir`
  (when `model_path` is unset).

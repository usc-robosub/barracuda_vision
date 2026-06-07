#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

cd /opt/barracuda-vision/ros2_ws
colcon build --symlink-install
source install/setup.bash

ros2 launch barracuda_vision barracuda_vision.launch.py

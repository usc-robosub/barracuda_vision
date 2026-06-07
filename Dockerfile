FROM ros:humble-ros-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends git vim wget \
    python3-pip \
    python3-colcon-common-extensions \
    libopencv-dev \
    libx11-dev \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN echo "source /opt/barracuda-vision/ros2_ws/install/setup.bash" >> /root/.bashrc

COPY . /opt/barracuda-vision
WORKDIR /opt
# Build and launch the workspace on container start
CMD ["/bin/bash", "/opt/barracuda-vision/entrypoint.sh"]

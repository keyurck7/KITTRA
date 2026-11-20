# Ubuntu 24.04 + ROS 2 Jazzy + MuJoCo + Kittra.
#   docker build -f docker/Dockerfile -t kittra .                         # simulation + ROS 2
#   docker build -f docker/Dockerfile --build-arg INSTALL_TRAIN=1 -t kittra-train .
#   docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix kittra
FROM osrf/ros:jazzy-desktop

ARG INSTALL_TRAIN=0
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv python3-pip git ffmpeg \
        libegl1 libgl1 libosmesa6 libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /kittra
COPY requirements*.txt pyproject.toml README.md ./
RUN python3 -m venv /opt/venv --system-site-packages \
    && /opt/venv/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-sim.txt \
    && if [ "$INSTALL_TRAIN" = "1" ]; then /opt/venv/bin/pip install --no-cache-dir -r requirements-train.txt; fi
ENV PATH=/opt/venv/bin:$PATH

COPY . .
RUN pip install --no-cache-dir -e . \
    && . /opt/ros/jazzy/setup.sh \
    && cd ros2_ws && colcon build --symlink-install

RUN echo "source /opt/ros/jazzy/setup.bash && source /kittra/ros2_ws/install/setup.bash" >> /root/.bashrc
CMD ["bash"]

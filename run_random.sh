xhost +si:localuser:root
docker run --gpus all -it  --rm --ipc host --privileged \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTHORITY \
      -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -v ./ros_ws/:/root/ros_workspace \
    -v ./data_collector/:/root/data_collector \
    -v /dev:/dev -e NVIDIA_DRIVER_CAPABILITIES=all \
    --name ros-zed-random \
    zed-ros2-humble:latest


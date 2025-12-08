xhost +si:localuser:root
docker run --gpus all -it --rm --net host --ipc host --privileged \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTHORITY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -v ./ros_ws/:/root/ros_workspace \
    --name ros-zed \
    zed-ros2-humble:latest

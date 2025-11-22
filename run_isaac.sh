xhost +
docker run -it --rm  --ipc host --privileged \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v ~/.Xauthority:/root/.Xauthority \
 -e DISPLAY=$DISPLAY \
 -e XAUTHORITY=$XAUTHORITY \
 -v ./ros_ws/:/root/ros_workspace \
 --name ros-isaac \
 ros:isaac bash
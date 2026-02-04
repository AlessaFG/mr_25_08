# Visualization of the ZED Camera Mounted on a TurtleBot3 in Isaac Sim

This project presents the development and implementation of an integrated **object detection** and **autonomous navigation** system on a TurtleBot3 mobile platform, simulated in **NVIDIA Isaac Sim**.  
The system uses a stereoscopic **ZED Camera**.

---

## Project Objectives

The main goal is to demonstrate how the integration of the TurtleBot3’s differential-drive mobility with the advanced computer vision of the ZED Camera allows the robot to:

- 🗺️ Map the surrounding environment.  
- 📦 Detect, classify, and locate specific targets (warehouse boxes) in real time.  
- 🤖 Navigate autonomously toward a selected target while avoiding obstacles.

---

## Technology Stack

- **Simulator:** NVIDIA Isaac Sim  
- **Robot:** TurtleBot3  
- **Vision Sensor:** ZED Camera  
- **Framework:** ROS 2  
- **Object Detection:** YOLO  
- **Programming Language:** Python  

---

## Technical Features

- Robot motion is managed through an **Action Graph in Isaac Sim** that subscribes to `Twist` messages on the `/cmd_vel` topic.  
- A **Differential Controller** node calculates the individual wheel velocities based on the robot’s physical specifications.  
- YOLO is used for **real-time target detection** and then to globally localize it within the environment map.  
- To stabilize the estimation of object centroids detected by the ZED Camera, an **Exponential Moving Average (EMA)** filter has been implemented.  
  This mathematical filter reduces noise from raw data, ensuring more reliable target localization during the approach.

---

## Results and Conclusions

The system was successfully validated in multiple operational scenarios within a warehouse environment. Despite variations in object arrangements, the robot demonstrated:

- **Robustness:** Excellent generalization capabilities in target identification.  
- **Precision:** Accurate positioning in front of the selected target.  
- **Effectiveness:** Full integration between stereoscopic vision algorithms and navigation control.

P.S Delate folder turtlebot3 gazebo

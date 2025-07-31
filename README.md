# 3D-visual-SLAM

A ROS 2-based real-time stereo camera 3D Visual SLAM system for localization and mapping, with RViz2 visualization and rosbag2 support.

- Written in Python with C++ extension support
- ROS 2 nodes for SLAM, control, and offline processing
- RViz2 visualization of camera feed, map, trajectory
- Record and replay with rosbag2
- Configurable via YAML

## Stack
- ROS 2 Humble (or later)
- Python (rclpy)
- Open3D, NumPy, Matplotlib
- RViz2
- rosbag2_py
- OpenCV
- PyYAML, loguru, tqdm, requests

## Setup
1. Install ROS 2 here: https://docs.ros.org/en/humble/Installation.html
2. Clone this repo and install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the ROS 2 workspace:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```


- **Launch SLAM and RViz2:**
  ```bash
  ros2 launch stereo_slam_pkg stereo_slam_launch.py
  ```
- **Record data:**
  ```bash
  python3 scripts/rosbag_recorder.py
  ```
- **Offline SLAM on rosbag:**
  ```bash
  python3 scripts/offline_processor.py --bag <bag_path>
  ```
- **Control node (reset, extract map):**
  ```bash
  ros2 run stereo_slam_pkg slam_control_interface
  ```
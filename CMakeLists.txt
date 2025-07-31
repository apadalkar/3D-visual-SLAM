cmake_minimum_required(VERSION 3.8)
project(stereo_slam_pkg)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(pybind11 REQUIRED)

ament_python_install_module(stereo_slam)

ament_package()

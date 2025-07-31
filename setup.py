from setuptools import setup
import os
from glob import glob

package_name = 'stereo_slam_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anit Padalkar',
    maintainer_email='apadalkar@berkeley.edu',
    description='Stereo Visual SLAM Package for ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_slam_node = stereo_slam.stereo_slam_node:main',
            'slam_control_interface = scripts.slam_control_interface:main',
            'rosbag_recorder = scripts.rosbag_recorder:main',
            'offline_processor = scripts.offline_processor:main',
        ],
    },
)

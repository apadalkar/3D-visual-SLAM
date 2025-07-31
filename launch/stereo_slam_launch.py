#Launch file for Stereo Visual SLAM system

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    """Generate launch description for SLAM system"""
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to SLAM configuration file'
    )
    
    # SLAM node
    slam_node = Node(
        package='stereo_slam_pkg',
        executable='stereo_slam_node',
        name='stereo_slam_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[
            ('/stereo/left/image_raw', '/camera/left/image_raw'),
            ('/stereo/right/image_raw', '/camera/right/image_raw'),
        ]
    )
    
    # Control node
    control_node = Node(
        package='stereo_slam_pkg',
        executable='slam_control_node',
        name='slam_control_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            os.path.dirname(__file__), '..', 'config', 'slam_visualization.rviz'
        )],
        output='screen'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        slam_node,
        control_node,
        rviz_node,
    ])
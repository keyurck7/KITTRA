"""Inspect the URDF alone: robot_state_publisher + joint sliders + RViz2.

    ros2 launch kittra_description display.launch.py
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("kittra_description"))
    urdf = (share / "urdf" / "kittra_arm.urdf").read_text()
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["--frame-id", "world", "--child-frame-id", "base_link"]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(share / "rviz" / "kittra.rviz")]),
    ])

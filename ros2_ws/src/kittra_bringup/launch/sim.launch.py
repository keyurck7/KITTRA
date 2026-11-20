"""Simulated Kittra cell: MuJoCo bridge + robot_state_publisher + RViz2.

    ros2 launch kittra_bringup sim.launch.py                       # idle cell, send /joint_commands yourself
    ros2 launch kittra_bringup sim.launch.py autopilot:=expert     # scripted pick-and-place demo
    ros2 launch kittra_bringup sim.launch.py env_config:=randomized rviz:=false

use_sim:=false is reserved for the Phase 3 hardware driver, which exposes the same topics.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    desc = Path(get_package_share_directory("kittra_description"))
    urdf = (desc / "urdf" / "kittra_arm.urdf").read_text()
    use_sim = LaunchConfiguration("use_sim")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim", default_value="true", description="true: MuJoCo bridge; false: real robot (Phase 3)"),
        DeclareLaunchArgument("env_config", default_value="nominal", description="config name or YAML path"),
        DeclareLaunchArgument("autopilot", default_value="none", description="none | expert"),
        DeclareLaunchArgument("publish_images", default_value="true"),
        DeclareLaunchArgument("image_size", default_value="128"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("rviz", default_value="true"),

        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": urdf}]),

        Node(package="kittra_bridge", executable="mujoco_bridge", output="screen", condition=IfCondition(use_sim),
             parameters=[{
                 "env_config": LaunchConfiguration("env_config"),
                 "autopilot": LaunchConfiguration("autopilot"),
                 "publish_images": LaunchConfiguration("publish_images"),
                 "image_size": LaunchConfiguration("image_size"),
                 "seed": LaunchConfiguration("seed"),
             }]),
        LogInfo(condition=UnlessCondition(use_sim),
                msg="use_sim:=false selects the SO-101 hardware driver, which is added in Phase 3."),

        Node(package="rviz2", executable="rviz2", output="log", condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", str(desc / "rviz" / "kittra.rviz")]),
    ])

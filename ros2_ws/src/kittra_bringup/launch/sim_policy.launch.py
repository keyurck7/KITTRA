"""Simulated cell driven by a trained policy (the Phase 3 deployment path, against simulation).

    ros2 launch kittra_bringup sim_policy.launch.py \
        policy_type:=act policy_path:=$PWD/outputs/act_randomized/checkpoints/last/pretrained_model \
        env_config:=randomized
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim = Path(get_package_share_directory("kittra_bringup")) / "launch" / "sim.launch.py"
    return LaunchDescription([
        DeclareLaunchArgument("policy_type", default_value="act"),
        DeclareLaunchArgument("policy_path", description="LeRobot pretrained_model directory (absolute path)"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("env_config", default_value="nominal"),
        DeclareLaunchArgument("rviz", default_value="true"),

        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(sim)), launch_arguments={
            "env_config": LaunchConfiguration("env_config"),
            "rviz": LaunchConfiguration("rviz"),
            "autopilot": "none",
            "publish_images": "true",
        }.items()),

        Node(package="kittra_policy", executable="policy_runner", output="screen", parameters=[{
            "policy_type": LaunchConfiguration("policy_type"),
            "policy_path": LaunchConfiguration("policy_path"),
            "device": LaunchConfiguration("device"),
        }]),
    ])

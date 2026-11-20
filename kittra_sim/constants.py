"""Names and constants shared by the simulation, the learning code and the ROS 2 nodes.

This module has no third-party dependencies so it can be imported anywhere
(including ROS 2 nodes and lightweight tests).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
#: Repository root. Override with the KITTRA_ROOT environment variable if the
#: package is used from outside the repository (e.g. from a ROS 2 install space).
REPO_ROOT = Path(os.environ.get("KITTRA_ROOT", Path(__file__).resolve().parents[1]))
ASSETS_DIR = REPO_ROOT / "assets"
CONFIGS_DIR = REPO_ROOT / "configs"
SCENE_PATH = ASSETS_DIR / "scenes" / "kitting_cell.xml"

# --------------------------------------------------------------------------- #
# Robot
# --------------------------------------------------------------------------- #
#: Actuated joints, in actuator / action order. Names follow the LeRobot SO-101 convention.
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = JOINT_NAMES[:5]
GRIPPER_JOINT = "gripper"
#: Passive finger joint coupled to the gripper joint by an equality constraint.
GRIPPER_MIRROR_JOINT = "gripper_mirror"
#: Joint names published on /joint_states (robot_state_publisher needs the mirror joint too).
ROS_JOINT_NAMES = JOINT_NAMES + (GRIPPER_MIRROR_JOINT,)

#: Gripper command values (metres of finger travel per side). Open = 4.8 cm between the pads:
#: 9 mm clearance on each side of the 3 cm cube, narrow enough to clear neighbouring parts.
GRIPPER_OPEN = 0.024
GRIPPER_CLOSED = 0.0

#: Home configuration: gripper pointing down above the table, open.
HOME_QPOS = (0.0, -1.3, 1.7, 1.17, 0.0, GRIPPER_OPEN)

TCP_SITE = "tcp"
#: The TCP is placed this far above the part centre when grasping, so the fingertips stay above
#: the tray rim and the pads still cover the upper 2/3 of the cube.
GRASP_OFFSET_Z = 0.004

# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
TARGET_PART = "part_target"
DISTRACTOR_PARTS = ("part_distractor_0", "part_distractor_1")
ALL_PARTS = (TARGET_PART,) + DISTRACTOR_PARTS
#: Half-extent of the target cube (m). Must match assets/scenes/kitting_cell.xml.
TARGET_HALF_SIZE = 0.015

TRAY_SPAWN_SITE = "tray_spawn"
KIT_AREA_SITE = "kit_area"
KIT_SLOT_SITE = "kit_slot"
REJECT_AREA_SITE = "reject_area"

OVERHEAD_CAMERA = "overhead"
WRIST_CAMERA = "wrist"
CAMERAS = (OVERHEAD_CAMERA, WRIST_CAMERA)

#: Where unused distractors are parked (under the table, outside every camera view).
PARK_POSITIONS = ((-0.1, 0.5, -0.7), (-0.1, 0.6, -0.7))

# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
CONTROL_HZ = 30
CONTROL_DT = 1.0 / CONTROL_HZ

TASK_DESCRIPTION = "Pick the red cube from the tray and place it in the kit box."

"""The MJCF used for simulation, the URDF used by ROS 2 and the Python constants must agree.
Parsed as XML, so this runs without MuJoCo or ROS."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kittra_sim import constants as C

MJCF = C.ASSETS_DIR / "so101" / "so101_simplified.xml"
SCENE = C.SCENE_PATH
URDF = C.REPO_ROOT / "ros2_ws" / "src" / "kittra_description" / "urdf" / "kittra_arm.urdf"


def vec(s, n=3):
    return np.array([float(x) for x in (s or " ".join(["0"] * n)).split()])


def mjcf_joints():
    """joint name -> (position of its body relative to the parent body, axis, range)"""
    root = ET.parse(MJCF).getroot()
    out = {}

    def walk(body):
        for j in body.findall("joint"):
            out[j.get("name")] = (vec(body.get("pos")), vec(j.get("axis")), vec(j.get("range"), 2))
        for child in body.findall("body"):
            walk(child)

    for b in root.find("worldbody").findall("body"):
        walk(b)
    return out


def urdf_joints():
    root = ET.parse(URDF).getroot()
    out = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        origin = j.find("origin")
        lim = j.find("limit")
        out[j.get("name")] = (vec(origin.get("xyz")), vec(j.find("axis").get("xyz")),
                              np.array([float(lim.get("lower")), float(lim.get("upper"))]))
    return out


def test_joint_sets_match():
    assert set(mjcf_joints()) == set(C.ROS_JOINT_NAMES)
    assert set(urdf_joints()) == set(C.ROS_JOINT_NAMES)


@pytest.mark.parametrize("name", C.ROS_JOINT_NAMES)
def test_kinematics_match(name):
    m_pos, m_axis, m_range = mjcf_joints()[name]
    u_pos, u_axis, u_range = urdf_joints()[name]
    np.testing.assert_allclose(m_pos, u_pos, atol=1e-9, err_msg=f"{name} origin")
    np.testing.assert_allclose(m_axis, u_axis, atol=1e-9, err_msg=f"{name} axis")
    np.testing.assert_allclose(m_range, u_range, atol=1e-9, err_msg=f"{name} limits")


def test_actuator_order_matches_action_space():
    root = ET.parse(MJCF).getroot()
    acts = [a.get("joint") for a in root.find("actuator")]
    assert tuple(acts) == C.JOINT_NAMES
    for a, (lo, hi) in zip(root.find("actuator"), (mjcf_joints()[n][2] for n in C.JOINT_NAMES)):
        np.testing.assert_allclose(vec(a.get("ctrlrange"), 2), [lo, hi])


def test_home_pose_within_limits():
    joints = mjcf_joints()
    for name, q in zip(C.JOINT_NAMES, C.HOME_QPOS):
        lo, hi = joints[name][2]
        assert lo <= q <= hi, name


def test_scene_names_and_timing():
    root = ET.parse(SCENE).getroot()
    sites = {s.get("name") for s in root.iter("site")}
    for s in (C.TRAY_SPAWN_SITE, C.KIT_AREA_SITE, C.KIT_SLOT_SITE, C.REJECT_AREA_SITE):
        assert s in sites
    bodies = {b.get("name") for b in root.iter("body")}
    assert set(C.ALL_PARTS) <= bodies
    target = root.find(".//geom[@name='part_target_geom']")
    np.testing.assert_allclose(vec(target.get("size")), [C.TARGET_HALF_SIZE] * 3)
    ts = float(root.find("option").get("timestep"))
    assert abs(16 * ts - C.CONTROL_DT) < 1e-6
    cams = {c.get("name") for c in root.iter("camera")} | {c.get("name") for c in ET.parse(MJCF).getroot().iter("camera")}
    assert set(C.CAMERAS) <= cams

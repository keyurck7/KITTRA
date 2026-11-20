"""Inverse kinematics reaches every pose the expert needs (needs MuJoCo)."""
import numpy as np
import pytest

from conftest import requires_mujoco

pytestmark = [requires_mujoco, pytest.mark.mujoco]


@pytest.fixture(scope="module")
def ik_and_env():
    from kittra_sim.env import KittraPickPlaceEnv
    from kittra_sim.ik import DLSIK

    env = KittraPickPlaceEnv("nominal", obs_mode="state")
    env.reset(seed=0)
    yield DLSIK(env.model), env
    env.close()


def test_forward_kinematics_matches_simulation(ik_and_env):
    ik, env = ik_and_env
    q = env.joint_positions()[:5]
    pos, _ = ik.forward(q)
    np.testing.assert_allclose(pos, env.data.site_xpos[env.h.tcp_site], atol=1e-9)


def test_topdown_poses_over_workspace(ik_and_env):
    from kittra_sim import constants as C
    from kittra_sim.transforms import closest_symmetric_angle, topdown_gripper_mat

    ik, env = ik_and_env
    tray = env.h.area(env.data, C.TRAY_SPAWN_SITE)
    kit = env.h.area(env.data, C.KIT_AREA_SITE)
    rng = np.random.default_rng(0)
    q = np.array(C.HOME_QPOS[:5])
    failures = 0
    targets = []
    for _ in range(40):
        xy = tray.sample(rng)
        z = tray.z + C.TARGET_HALF_SIZE + rng.uniform(0.0, 0.08)
        targets.append(np.array([xy[0], xy[1], z]))
    for _ in range(20):
        xy = kit.sample(rng, margin=0.02)
        targets.append(np.array([xy[0], xy[1], kit.z + rng.uniform(0.02, 0.11)]))
    for t in targets:
        natural = np.arctan2(t[1], t[0]) + np.pi / 2
        yaw = closest_symmetric_angle(rng.uniform(-np.pi, np.pi), natural, np.pi / 2)
        res = ik.solve(q, t, topdown_gripper_mat(yaw))
        failures += not res.success
        if res.success:
            pos, mat = ik.forward(res.qpos)
            assert np.linalg.norm(pos - t) < 2e-3
            assert mat[2, 0] < -0.99            # approach axis points down
    assert failures == 0, f"{failures}/{len(targets)} IK failures"

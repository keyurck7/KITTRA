"""Pick-and-place environment (needs MuJoCo)."""
import numpy as np
import pytest

from conftest import requires_mujoco, requires_render

pytestmark = [requires_mujoco, pytest.mark.mujoco]


@pytest.fixture
def env():
    from kittra_sim.env import KittraPickPlaceEnv

    e = KittraPickPlaceEnv("nominal", obs_mode="state")
    yield e
    e.close()


def test_model_loads_with_expected_timing(env):
    from kittra_sim.constants import CONTROL_DT

    assert env.n_substeps == 16
    assert env.control_dt == pytest.approx(CONTROL_DT, rel=1e-6)


def test_reset_step_shapes_and_spaces(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert set(obs) == {"joint_pos", "joint_vel", "ee_pos"}
    assert "target_pos" in info and "target_pos" not in obs   # privileged info stays out of obs
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)


def test_parts_spawn_in_tray_and_settle(env):
    from kittra_sim import constants as C

    for seed in range(10):
        _, info = env.reset(seed=seed)
        tray = env.h.area(env.data, C.TRAY_SPAWN_SITE)
        assert tray.contains(info["target_pos"][:2], margin=-0.01)
        assert info["target_pos"][2] == pytest.approx(tray.z + C.TARGET_HALF_SIZE, abs=0.004)
        assert env.h.part_speed(env.data) < 0.05
        positions = [env.h.part_pose(env.data, n)[0][:2] for n in C.ALL_PARTS]
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.linalg.norm(positions[i] - positions[j]) > 0.03


def test_seeding_is_deterministic(env):
    o1, i1 = env.reset(seed=123)
    o2, i2 = env.reset(seed=123)
    np.testing.assert_allclose(i1["target_pos"], i2["target_pos"])
    a = env.current_command()
    for _ in range(5):
        o1, *_ = env.step(a)
    env.reset(seed=123)
    for _ in range(5):
        o2, *_ = env.step(a)
    np.testing.assert_allclose(o1["joint_pos"], o2["joint_pos"], atol=1e-9)


def test_holding_home_pose_is_stable(env):
    env.reset(seed=0)
    q0 = env.joint_positions()
    cmd = env.current_command()
    for _ in range(60):
        env.step(cmd)
    np.testing.assert_allclose(env.joint_positions()[:5], q0[:5], atol=0.02)


def test_actions_are_clipped(env):
    env.reset(seed=0)
    env.step(np.full(6, 100.0))
    np.testing.assert_allclose(env.current_command(), env.h.ctrl_hi)


def test_randomization_changes_model_and_restores():
    from kittra_sim.env import KittraPickPlaceEnv

    rnd = KittraPickPlaceEnv("randomized", obs_mode="state")
    nominal_mass = rnd.physics_rand.nominal["mass"].copy()
    masses = []
    for s in range(5):
        rnd.reset(seed=s)
        masses.append(rnd.model.body_mass.copy())
    assert not np.allclose(masses[0], masses[1])
    rnd.physics_rand.cfg = {**rnd.physics_rand.cfg, "enabled": False}
    rnd.reset(seed=0)
    np.testing.assert_array_equal(rnd.model.body_mass, nominal_mass)
    rnd.close()


def test_gym_make_registration():
    import gymnasium as gym

    import kittra_sim  # noqa: F401

    e = gym.make("KittraPickPlace-v0", config="randomized", obs_mode="state")
    obs, _ = e.reset(seed=0)
    e.step(e.action_space.sample())
    e.close()
    p = gym.make("KittraPlace-v0", config="nominal")
    obs, _ = p.reset(seed=0)
    assert obs.shape == (22,)
    p.close()


@requires_render
@pytest.mark.render
def test_rgb_observations():
    from kittra_sim.env import KittraPickPlaceEnv

    e = KittraPickPlaceEnv("nominal", obs_mode="rgb", image_size=64)
    obs, _ = e.reset(seed=0)
    assert obs["overhead_rgb"].shape == (64, 64, 3) and obs["overhead_rgb"].dtype == np.uint8
    assert obs["overhead_rgb"].std() > 5          # not a blank frame
    assert obs["wrist_rgb"].std() > 5
    e.close()

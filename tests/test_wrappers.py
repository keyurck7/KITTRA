"""Sensor/actuator wrappers on the real environment (needs MuJoCo + Gymnasium)."""
import numpy as np
import pytest

from conftest import requires_mujoco

pytestmark = [requires_mujoco, pytest.mark.mujoco]


def _base(obs_mode="state"):
    from kittra_sim.env import KittraPickPlaceEnv

    return KittraPickPlaceEnv("nominal", obs_mode=obs_mode)


def test_latency_wrapper_delays_commands_exactly():
    from kittra_sim.randomization.wrappers import ActionLatencyWrapper

    env = ActionLatencyWrapper(_base(), min_delay=2, max_delay=2)
    _, info = env.reset(seed=0)
    assert info["action_delay"] == 2
    home = env.unwrapped.current_command()
    cmds = [home + np.r_[0.01 * (i + 1), np.zeros(5)] for i in range(5)]
    sent = []
    for c in cmds:
        env.step(c)
        sent.append(env.unwrapped.current_command())
    np.testing.assert_allclose(sent[0], home)
    np.testing.assert_allclose(sent[1], home)
    np.testing.assert_allclose(sent[2], cmds[0])
    np.testing.assert_allclose(sent[4], cmds[2])
    env.close()


def test_backlash_wrapper_ignores_small_reversals():
    from kittra_sim.randomization.wrappers import BacklashWrapper

    env = BacklashWrapper(_base(), max_play=0.02)
    env.reset(seed=1)
    play = env.play
    assert np.all(play[:5] <= 0.02) and play[5] == 0.0
    home = env.unwrapped.current_command()
    tiny = home + np.r_[0.4 * play[:5], 0.0]          # inside half the play: arm must not move
    env.step(tiny)
    np.testing.assert_allclose(env.unwrapped.current_command()[:5], home[:5])
    env.close()


def test_sensor_noise_statistics_and_reproducibility():
    from kittra_sim.config import load_config
    from kittra_sim.randomization.wrappers import SensorNoiseWrapper

    cfg = load_config("randomized")["sensor"]
    env = SensorNoiseWrapper(_base(), cfg)
    obs, _ = env.reset(seed=5)
    true_q = env.unwrapped.joint_positions()
    err = obs["joint_pos"] - true_q
    assert np.all(np.abs(err) < cfg["joint_bias"] + 5 * cfg["joint_sigma"])
    obs2, _ = env.reset(seed=5)
    np.testing.assert_allclose(obs["joint_pos"], obs2["joint_pos"])   # same seed, same noise
    obs3, _ = env.reset(seed=6)
    assert not np.allclose(obs["joint_pos"], obs3["joint_pos"])
    env.close()


def test_factory_builds_wrapper_stack():
    from kittra_sim.factory import make_pick_place_env
    from kittra_sim.randomization.wrappers import ActionLatencyWrapper, BacklashWrapper, SensorNoiseWrapper

    env = make_pick_place_env("randomized", obs_mode="state")
    kinds = []
    e = env
    while hasattr(e, "env"):
        kinds.append(type(e))
        e = e.env
    assert kinds == [SensorNoiseWrapper, ActionLatencyWrapper, BacklashWrapper]
    nominal = make_pick_place_env("nominal", obs_mode="state")
    assert not hasattr(nominal, "env")
    env.close()
    nominal.close()

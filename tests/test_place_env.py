"""Placement environment and its scripted expert (needs MuJoCo)."""
import numpy as np
import pytest

from conftest import requires_mujoco

pytestmark = [requires_mujoco, pytest.mark.mujoco]


def test_reset_starts_with_part_in_gripper():
    from kittra_sim.place_env import KittraPlaceEnv

    env = KittraPlaceEnv("nominal")
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (22,) and np.all(np.isfinite(obs))
        assert np.linalg.norm(info["target_pos"] - info["tcp_pos"]) < 0.02
        assert info["tcp_pos"][2] > info["slot_pos"][2] + 0.04
        # Holding still keeps the part in the gripper
        for _ in range(10):
            obs, r, term, trunc, info = env.step(np.r_[np.zeros(5), -1.0])
        assert not term and not info["part_dropped"]
    env.close()


def test_immediate_release_fails():
    from kittra_sim.place_env import KittraPlaceEnv

    env = KittraPlaceEnv("nominal")
    env.reset(seed=0)
    _, r, term, _, info = env.step(np.r_[np.zeros(5), 1.0])   # drop from 6-10 cm, usually off the slot
    assert term
    if info["is_success"]:
        assert info["placement_error"] < env.success_tol
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("config,min_success", [("nominal", 8), ("randomized", 6)])
def test_place_expert(config, min_success):
    from kittra_sim.place_env import KittraPlaceEnv
    from kittra_sim.place_expert import ScriptedPlaceExpert

    env = KittraPlaceEnv(config)
    expert = ScriptedPlaceExpert(env)
    succ = 0
    for seed in range(10):
        obs, info = env.reset(seed=seed)
        expert.reset(obs, info)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(expert.act(obs, info))
            done = term or trunc
        succ += bool(info["is_success"])
    env.close()
    assert succ >= min_success, f"place expert {succ}/10 on {config}"

"""The scripted expert must be reliable, otherwise the datasets are biased (needs MuJoCo; slow)."""
import pytest

from conftest import requires_mujoco

pytestmark = [requires_mujoco, pytest.mark.mujoco, pytest.mark.slow]


def run(config: str, episodes: int, seed0: int = 0) -> tuple[int, list[int]]:
    from kittra_sim.factory import make_pick_place_env
    from kittra_sim.scripted_expert import ScriptedPickPlaceExpert

    env = make_pick_place_env(config, obs_mode="state")
    expert = ScriptedPickPlaceExpert(env)
    successes, lengths = 0, []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        expert.reset(obs, info)
        done, steps = False, 0
        while not done:
            obs, _, term, trunc, info = env.step(expert.act(obs, info))
            steps += 1
            done = term or trunc
        successes += bool(info["is_success"])
        lengths.append(steps)
    env.close()
    return successes, lengths


def test_expert_nominal():
    succ, lengths = run("nominal", 10)
    assert succ >= 9, f"expert succeeded in only {succ}/10 nominal episodes"
    assert max(lengths) < 300


def test_expert_randomized():
    succ, _ = run("randomized", 10, seed0=100)
    assert succ >= 7, f"expert succeeded in only {succ}/10 randomized episodes"

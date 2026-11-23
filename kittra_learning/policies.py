"""One interface for every policy Kittra evaluates::

    policy = load_policy(policy_type, path, env=env, device="cuda")
    policy.reset(obs, info)
    action = policy.act(obs, info)

Types
    expert      scripted pick-and-place expert (privileged; upper bound / sanity check)
    place_expert scripted placement expert (privileged)
    act         LeRobot ACT checkpoint (pretrained_model directory)
    diffusion   LeRobot Diffusion Policy checkpoint
    sb3         Stable-Baselines3 PPO .zip (+ VecNormalize stats next to it)
    bc          Kittra behaviour-cloning MLP .pt
    random      uniform random actions (lower bound)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kittra_sim.constants import TASK_DESCRIPTION


class Policy(Protocol):
    def reset(self, obs: Any = None, info: dict | None = None) -> None: ...
    def act(self, obs: Any, info: dict) -> np.ndarray: ...


class LeRobotPolicy:
    def __init__(self, path: str | Path, policy_type: str, device: str = "cpu", task: str = TASK_DESCRIPTION):
        import torch

        from kittra_learning import lerobot_compat as lc

        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"{path} is not a checkpoint directory (expected .../checkpoints/last/pretrained_model); "
                "did training finish?")
        self.torch = torch
        self.device = torch.device(device)
        self.policy = lc.get_policy_class(policy_type).from_pretrained(str(path))
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = lc.make_processors(self.policy, path)
        self.task = task
        self._lc = lc

    def reset(self, obs=None, info=None) -> None:
        self.policy.reset()  # clears ACT's action-chunk queue / diffusion observation history
        for proc in (self.pre, self.post):
            if proc is not None and hasattr(proc, "reset"):
                proc.reset()

    def act(self, obs, info=None) -> np.ndarray:
        batch = self._lc.obs_to_batch(obs, self.task, self.device)
        with self.torch.inference_mode():
            if self.pre is not None:
                batch = self.pre(batch)
            action = self.policy.select_action(batch)
            if self.post is not None:
                action = self.post(action)
        return action.squeeze(0).float().cpu().numpy()


class SB3Policy:
    def __init__(self, path: str | Path, env, device: str = "cpu"):
        import re

        from stable_baselines3.common.vec_env import DummyVecEnv

        from kittra_learning.numpy_compat import load_ppo, load_vecnormalize

        path = Path(path)
        zip_path = path if path.suffix == ".zip" else path.with_suffix(".zip")
        self.model = load_ppo(zip_path, device=device, spaces_from=env)
        # best_model.zip -> best_model_vecnormalize.pkl; ppo_place_N_steps.zip -> ppo_place_vecnormalize_N_steps.pkl
        stats = zip_path.parent / (zip_path.stem + "_vecnormalize.pkl")
        m = re.fullmatch(r"(.+)_(\d+)_steps", zip_path.stem)
        if not stats.exists() and m:
            stats = zip_path.parent / f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl"
        self.vecnorm = None
        if stats.exists():
            dummy = DummyVecEnv([lambda: env])
            self.vecnorm = load_vecnormalize(stats, dummy)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
        else:
            print(f"warning: no observation statistics found next to {zip_path.name}; using raw observations")

    def reset(self, obs=None, info=None) -> None:
        pass

    def act(self, obs, info=None) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if self.vecnorm is not None:
            x = self.vecnorm.normalize_obs(x[None])[0]
        action, _ = self.model.predict(x, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class RandomPolicy:
    def __init__(self, env, seed: int = 0):
        self.space = env.action_space
        self.space.seed(seed)

    def reset(self, obs=None, info=None) -> None:
        pass

    def act(self, obs, info=None) -> np.ndarray:
        return self.space.sample()


def load_policy(policy_type: str, path: str | Path | None = None, env=None, device: str = "cpu") -> Policy:
    if policy_type == "expert":
        from kittra_sim.scripted_expert import ScriptedPickPlaceExpert

        return ScriptedPickPlaceExpert(env)
    if policy_type == "place_expert":
        from kittra_sim.place_expert import ScriptedPlaceExpert

        return ScriptedPlaceExpert(env)
    if policy_type in ("act", "diffusion", "smolvla"):
        if path is None:
            raise ValueError("LeRobot policies need --policy <pretrained_model dir>")
        return LeRobotPolicy(path, policy_type, device)
    if policy_type == "sb3":
        return SB3Policy(path, env, device)
    if policy_type == "bc":
        from kittra_learning.bc import BCPolicy

        return BCPolicy(path, device)
    if policy_type == "random":
        return RandomPolicy(env)
    raise ValueError(f"unknown policy type '{policy_type}'")


#: Which environment each policy type is evaluated in.
POLICY_ENV = {
    "expert": "pick_place", "act": "pick_place", "diffusion": "pick_place", "smolvla": "pick_place",
    "place_expert": "place", "sb3": "place", "bc": "place",
}

"""Composable Gymnasium wrappers for sensor and actuator imperfections (pick-and-place env).

Signal chain used by :func:`kittra_sim.factory.make_pick_place_env`::

    policy --action--> ActionLatencyWrapper --> BacklashWrapper --> env (MuJoCo)
    policy <--obs----- SensorNoiseWrapper   <-------------------------- env

Each wrapper draws its per-episode parameters right *after* the inner ``reset`` (which may have
re-seeded the environment), from a private generator seeded by the environment's generator.
That keeps episodes reproducible from ``reset(seed=...)``.
"""
from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from kittra_sim.randomization.core import (
    DelayLine,
    apply_brightness_contrast,
    apply_gaussian_noise,
    apply_motion_blur,
    backlash_step,
    to_uint8,
)

IMAGE_KEYS = ("overhead_rgb", "wrist_rgb")


def _child_rng(env: gym.Env) -> np.random.Generator:
    return np.random.default_rng(int(env.unwrapped.np_random.integers(0, 2**63 - 1)))


class ActionLatencyWrapper(gym.Wrapper):
    """Delays every action by k control steps; k is uniform in [min_delay, max_delay] per episode."""

    def __init__(self, env: gym.Env, min_delay: int = 0, max_delay: int = 3):
        super().__init__(env)
        if not 0 <= min_delay <= max_delay:
            raise ValueError("need 0 <= min_delay <= max_delay")
        self.min_delay, self.max_delay = int(min_delay), int(max_delay)
        self.delay = 0
        self._line: DelayLine | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rng = _child_rng(self.env)
        self.delay = int(rng.integers(self.min_delay, self.max_delay + 1))
        # Pre-fill with "hold current position" commands.
        self._line = DelayLine(self.delay, self.env.unwrapped.current_command())
        info = dict(info, action_delay=self.delay)
        return obs, info

    def step(self, action):
        assert self._line is not None, "call reset() first"
        return self.env.step(self._line.push(np.asarray(action, dtype=float)))


class BacklashWrapper(gym.Wrapper):
    """Mechanical play on the arm joints (not the gripper), sampled per joint per episode."""

    def __init__(self, env: gym.Env, max_play: float = 0.01, n_arm_joints: int = 5):
        super().__init__(env)
        self.max_play = float(max_play)
        self.n_arm = int(n_arm_joints)
        self.play = np.zeros(self.env.action_space.shape[0])
        self._out: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rng = _child_rng(self.env)
        self.play = np.zeros(self.env.action_space.shape[0])
        self.play[: self.n_arm] = rng.uniform(0.0, self.max_play, self.n_arm)
        self._out = np.asarray(self.env.unwrapped.current_command(), dtype=float).copy()
        return obs, info

    def step(self, action):
        assert self._out is not None, "call reset() first"
        self._out = backlash_step(self._out, np.asarray(action, dtype=float), self.play)
        return self.env.step(self._out.copy())


class SensorNoiseWrapper(gym.ObservationWrapper):
    """Joint encoder noise + bias, image noise, motion blur, brightness/contrast changes."""

    def __init__(self, env: gym.Env, cfg: Mapping[str, Any]):
        super().__init__(env)
        self.cfg = dict(cfg)
        self._rng = np.random.default_rng(0)
        self.bias = np.zeros(self.env.action_space.shape[0])
        self.img_sigma = 0.0
        self.brightness = 1.0
        self.contrast = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._rng = _child_rng(self.env)
        c = self.cfg
        n = self.env.action_space.shape[0]
        self.bias = self._rng.uniform(-c["joint_bias"], c["joint_bias"], n)
        self.img_sigma = float(self._rng.uniform(0.0, c["img_sigma"]))
        self.brightness = 1.0 + float(self._rng.uniform(-c["brightness"], c["brightness"]))
        self.contrast = 1.0 + float(self._rng.uniform(-c["contrast"], c["contrast"]))
        return self.observation(obs), info

    def observation(self, obs: dict) -> dict:
        c = self.cfg
        rng = self._rng
        out = dict(obs)  # never mutate the inner env's arrays
        jp = np.asarray(obs["joint_pos"], dtype=np.float64)
        out["joint_pos"] = (jp + self.bias + rng.normal(0.0, c["joint_sigma"], jp.shape)).astype(np.float32)
        for key in IMAGE_KEYS:
            if key not in obs:
                continue
            img = obs[key].astype(np.float32)
            img = apply_brightness_contrast(img, self.brightness, self.contrast)
            img = apply_gaussian_noise(img, self.img_sigma, rng)
            if rng.random() < c["blur_p"]:
                k = int(rng.integers(1, int(c["blur_max_kernel"]) + 1))
                img = apply_motion_blur(img, k, horizontal=bool(rng.random() < 0.5))
            out[key] = to_uint8(img)
        return out

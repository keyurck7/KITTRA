"""Environment factories: build an environment plus the wrappers its config asks for.

Use these (or ``gymnasium.make("KittraPickPlace-v0", config=...)``) instead of instantiating the
classes directly, so that sensor and actuator models are always applied consistently.
"""
from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

from kittra_sim.config import load_config


def make_pick_place_env(
    config: str | Mapping[str, Any] | None = None,
    obs_mode: str = "rgb",
    image_size: int = 128,
    render_mode: str | None = None,
) -> gym.Env:
    from kittra_sim.env import KittraPickPlaceEnv
    from kittra_sim.randomization.wrappers import ActionLatencyWrapper, BacklashWrapper, SensorNoiseWrapper

    cfg = load_config(config)
    env: gym.Env = KittraPickPlaceEnv(cfg, obs_mode=obs_mode, image_size=image_size, render_mode=render_mode)
    act = cfg["actuator"]
    if act["enabled"]:
        if float(act["backlash"]) > 0:
            env = BacklashWrapper(env, max_play=float(act["backlash"]))
        if int(act["max_delay_steps"]) > 0:
            env = ActionLatencyWrapper(env, int(act["min_delay_steps"]), int(act["max_delay_steps"]))
    if cfg["sensor"]["enabled"]:
        env = SensorNoiseWrapper(env, cfg["sensor"])
    return env


def make_place_env(config: str | Mapping[str, Any] | None = None, **kwargs) -> gym.Env:
    """State-based placement env. Sensor and actuator models are applied inside the env because its
    actions are joint *deltas* (see ``kittra_sim.place_env``)."""
    from kittra_sim.place_env import KittraPlaceEnv

    return KittraPlaceEnv(load_config(config), **kwargs)

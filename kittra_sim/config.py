"""Environment configuration: defaults, YAML loading, merging and validation.

A config has five sections. Every section except ``task`` has an ``enabled`` flag, so a single
YAML file fully describes one evaluation condition (nominal, randomized, out-of-distribution or a
single-perturbation ablation).

Pure Python (only PyYAML), so it is usable from tests, scripts and ROS 2 nodes alike.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from kittra_sim.constants import CONFIGS_DIR

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "default",
    "task": {
        "n_distractors": 2,          # 0, 1 or 2 distractor parts in the tray
        "max_episode_steps": 300,    # 10 s at 30 Hz
        "reward": "sparse",          # "sparse" or "shaped"
        "init_qpos_noise": 0.02,     # rad, uniform noise on the home pose at reset
        "settle_steps": 60,          # physics steps after spawning so parts come to rest
        "min_part_distance": 0.055,  # m, centre distance; open finger reach + distractor radius + ~7 mm margin
    },
    "physics": {
        "enabled": False,
        "friction": [1.0, 1.0],      # scale of sliding friction (parts + finger pads)
        "mass": [1.0, 1.0],          # scale of part masses (inertia scaled consistently)
        "damping": [1.0, 1.0],       # scale of arm joint damping
        "kp": [1.0, 1.0],            # scale of arm actuator position gains
    },
    "visual": {
        "enabled": False,
        "light_pos_offset": 0.0,     # m, uniform offset per axis
        "light_intensity": [1.0, 1.0],
        "color_jitter": 0.0,         # +/- per RGB channel for table and distractors
        "target_color_jitter": 0.0,  # +/- per RGB channel for the target (kept reddish)
        "cam_pos_offset": 0.0,       # m, uniform offset per axis (overhead camera)
        "cam_rot_deg": 0.0,          # deg, random-axis rotation (overhead camera)
    },
    "sensor": {
        "enabled": False,
        "joint_sigma": 0.0,          # rad, white noise on joint positions
        "joint_bias": 0.0,           # rad, per-episode constant bias (uniform +/-)
        "img_sigma": 0.0,            # max std of Gaussian pixel noise (0-255 scale)
        "blur_p": 0.0,               # probability of motion blur per frame
        "blur_max_kernel": 5,        # px
        "brightness": 0.0,           # +/- fraction, per episode
        "contrast": 0.0,             # +/- fraction, per episode
    },
    "actuator": {
        "enabled": False,
        "min_delay_steps": 0,        # action latency, uniform in {min..max} control steps per episode
        "max_delay_steps": 0,
        "backlash": 0.0,             # rad, max play per arm joint (uniform per joint per episode)
    },
}

_RANGE_KEYS = {
    "physics": ("friction", "mass", "damping", "kp"),
    "visual": ("light_intensity",),
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config_path(name_or_path: str | Path) -> Path:
    """Accept a file path or a bare name such as ``"randomized"`` (looked up in configs/env/)."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    for candidate in (
        CONFIGS_DIR / "env" / f"{name_or_path}.yaml",
        CONFIGS_DIR / "env" / "ablation" / f"{name_or_path}.yaml",
        CONFIGS_DIR / "env" / str(name_or_path),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Environment config not found: {name_or_path}")


def load_config(config: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a full, validated config dict.

    ``config`` may be ``None`` (defaults = nominal), a dict, a YAML path, or a config name.
    A YAML file may contain ``inherit: <name>`` to start from another config.
    """
    if config is None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    elif isinstance(config, Mapping):
        cfg = deep_merge(DEFAULT_CONFIG, config)
    else:
        path = resolve_config_path(config)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        parent = raw.pop("inherit", None)
        base = load_config(parent) if parent else DEFAULT_CONFIG
        cfg = deep_merge(base, raw)
        cfg.setdefault("name", path.stem)
        if "name" not in raw:
            cfg["name"] = path.stem
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` on malformed values."""
    for section in ("task", "physics", "visual", "sensor", "actuator"):
        if section not in cfg:
            raise ValueError(f"missing config section '{section}'")
    for section, keys in _RANGE_KEYS.items():
        for key in keys:
            lo_hi = cfg[section][key]
            if len(lo_hi) != 2 or float(lo_hi[0]) > float(lo_hi[1]) or float(lo_hi[0]) < 0:
                raise ValueError(f"{section}.{key} must be [lo, hi] with 0 <= lo <= hi, got {lo_hi}")
    task = cfg["task"]
    if int(task["n_distractors"]) not in (0, 1, 2):
        raise ValueError("task.n_distractors must be 0, 1 or 2")
    if task["reward"] not in ("sparse", "shaped"):
        raise ValueError("task.reward must be 'sparse' or 'shaped'")
    lo_d, hi_d = int(cfg["actuator"]["min_delay_steps"]), int(cfg["actuator"]["max_delay_steps"])
    if not 0 <= lo_d <= hi_d:
        raise ValueError("actuator delay steps must satisfy 0 <= min_delay_steps <= max_delay_steps")
    for key in ("joint_sigma", "joint_bias", "img_sigma", "brightness", "contrast"):
        if float(cfg["sensor"][key]) < 0:
            raise ValueError(f"sensor.{key} must be >= 0")
    if not 0.0 <= float(cfg["sensor"]["blur_p"]) <= 1.0:
        raise ValueError("sensor.blur_p must be in [0, 1]")
    if int(cfg["sensor"]["blur_max_kernel"]) < 1:
        raise ValueError("sensor.blur_max_kernel must be >= 1")

"""Thin compatibility layer over LeRobot.

LeRobot's module layout and APIs change between releases (``lerobot.common.*`` in early versions,
``lerobot.*`` from 0.3, normalization moved from the policy into processor pipelines in 0.4,
``add_frame`` taking ``task`` as a keyword in some versions). Everything Kittra needs from
LeRobot goes through this module, so a version change only touches this file.

Tested against the APIs documented for LeRobot 0.3.x and 0.4.x. Pin the version you use in
requirements.txt once your pipeline works.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _first_import(*paths: str):
    errors = []
    for path in paths:
        module_name, _, attr = path.rpartition(".")
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as e:  # pragma: no cover - depends on installed version
            errors.append(f"{path}: {e}")
    raise ImportError("Could not import any of:\n  " + "\n  ".join(errors))


def lerobot_version() -> str:
    try:
        from importlib.metadata import version

        return version("lerobot")
    except Exception:  # pragma: no cover
        return "unknown"


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def get_dataset_class():
    return _first_import(
        "lerobot.datasets.lerobot_dataset.LeRobotDataset",
        "lerobot.common.datasets.lerobot_dataset.LeRobotDataset",
    )


def create_dataset(repo_id: str, root: str | Path, fps: int, features: Mapping[str, Any],
                   robot_type: str = "kittra_so101_sim", use_videos: bool = True):
    cls = get_dataset_class()
    kwargs = dict(repo_id=repo_id, fps=fps, features=dict(features), root=str(root), robot_type=robot_type)
    params = inspect.signature(cls.create).parameters
    if "use_videos" in params:
        kwargs["use_videos"] = use_videos
    if "image_writer_threads" in params:
        kwargs["image_writer_threads"] = 4
    return cls.create(**kwargs)


def add_frame(dataset, frame: dict[str, Any], task: str) -> None:
    """Add one frame, passing the task the way the installed LeRobot version expects."""
    params = inspect.signature(dataset.add_frame).parameters
    if "task" in params:
        dataset.add_frame(frame, task=task)
    else:
        dataset.add_frame({**frame, "task": task})


def finish_dataset(dataset) -> None:
    """Flush pending writes. Newer versions need ``finalize()``; older ones used ``consolidate()``."""
    for name in ("finalize", "consolidate"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            fn()
            return


def build_features(joint_names, image_size: int, use_videos: bool) -> dict[str, Any]:
    img_dtype = "video" if use_videos else "image"
    img = {"dtype": img_dtype, "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": list(joint_names)},
        "observation.images.overhead": dict(img),
        "observation.images.wrist": dict(img),
        "action": {"dtype": "float32", "shape": (len(joint_names),), "names": list(joint_names)},
    }


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
_POLICY_CLASSES = {
    "act": ("lerobot.policies.act.modeling_act.ACTPolicy",
            "lerobot.common.policies.act.modeling_act.ACTPolicy"),
    "diffusion": ("lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy",
                  "lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy"),
    "smolvla": ("lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",),
}


def get_policy_class(policy_type: str):
    if policy_type not in _POLICY_CLASSES:
        raise ValueError(f"unknown LeRobot policy type '{policy_type}', expected one of {list(_POLICY_CLASSES)}")
    return _first_import(*_POLICY_CLASSES[policy_type])


def make_processors(policy, pretrained_path: str | Path):
    """Return (preprocessor, postprocessor) for LeRobot >= 0.4, or (None, None) when normalization
    still lives inside the policy (older versions)."""
    try:
        make = _first_import("lerobot.policies.factory.make_pre_post_processors")
    except ImportError:
        return None, None
    try:
        return make(policy.config, pretrained_path=str(pretrained_path))
    except TypeError:  # pragma: no cover - signature differences between minor versions
        return make(policy_cfg=policy.config, pretrained_path=str(pretrained_path))


def obs_to_batch(obs: Mapping[str, np.ndarray], task: str, device) -> dict[str, Any]:
    """Kittra observation dict -> LeRobot batch (batch size 1, images float CHW in [0, 1])."""
    import torch

    def img(x: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    return {
        "observation.state": torch.from_numpy(np.asarray(obs["joint_pos"], dtype=np.float32)).unsqueeze(0).to(device),
        "observation.images.overhead": img(obs["overhead_rgb"]),
        "observation.images.wrist": img(obs["wrist_rgb"]),
        "task": [task],
    }

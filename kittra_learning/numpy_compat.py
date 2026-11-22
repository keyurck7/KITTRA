"""Load Stable-Baselines3 files across NumPy major versions.

NumPy 2 renamed ``numpy.core`` to ``numpy._core``. Pickled objects saved under NumPy 2 (inside an
SB3 ``.zip`` or a ``VecNormalize`` ``.pkl``) reference ``numpy._core.*`` and fail to load under
NumPy 1.x with ``ModuleNotFoundError: No module named 'numpy._core.numeric'``. This happens when a
model trained on one machine (or venv) is evaluated on another, e.g. the Ubuntu/ROS 2 laptop
(NumPy 1.26) and a GPU server (NumPy 2).

``load_ppo`` avoids unpickling anything that is not needed (spaces, schedules, rollout state) and
``numpy2_pickle_compat`` aliases the renamed modules for the rest (observation statistics).
"""
from __future__ import annotations

import contextlib
import importlib
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_SUBMODULES = ("", ".multiarray", ".numeric", "._multiarray_umath", ".umath", ".fromnumeric", "._methods")


@contextlib.contextmanager
def numpy2_pickle_compat() -> Iterator[None]:
    """Temporarily let NumPy 1.x resolve ``numpy._core.*`` names written by NumPy 2. No-op on NumPy 2."""
    if int(np.__version__.split(".")[0]) >= 2:
        yield
        return
    added: list[str] = []
    for sub in _SUBMODULES:
        alias = "numpy._core" + sub
        if alias in sys.modules:
            continue
        try:
            importlib.import_module(alias)  # a real module of that name wins
            continue
        except ImportError:
            pass
        try:
            sys.modules[alias] = importlib.import_module("numpy.core" + sub)
            added.append(alias)
        except ImportError:
            continue
    try:
        yield
    finally:
        for alias in added:
            sys.modules.pop(alias, None)


def load_ppo(path: str | Path, env: Any = None, device: str = "cpu", spaces_from: Any = None,
             clip_range: float = 0.2, **kwargs: Any):
    """``PPO.load`` that works across NumPy versions.

    Spaces come from ``env`` (or ``spaces_from``) instead of the file; schedules are rebuilt from the
    saved hyperparameters; rollout state is reset (SB3 rebuilds it on the next ``learn``).
    """
    from stable_baselines3 import PPO

    src = env if env is not None else spaces_from
    custom: dict[str, Any] = {
        "lr_schedule": lambda _: 0.0,          # rebuilt from the saved learning_rate by _setup_model
        "clip_range": clip_range,              # wrapped into a schedule by _setup_model
        "_last_obs": None,
        "_last_original_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }
    if src is not None:
        custom["observation_space"] = src.observation_space
        custom["action_space"] = src.action_space
    with numpy2_pickle_compat():
        return PPO.load(str(path), env=env, device=device, custom_objects=custom, **kwargs)


class _RNGPlaceholder:
    """Stands in for a NumPy 2 random generator that NumPy 1.x cannot rebuild."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # NumPy 2 pickles generators as (class, (state, seed_seq)); NumPy 1.x only knows names + dicts.
        if module == "numpy.random._pickle":
            return _RNGPlaceholder
        return super().find_class(module, name)


def _drop_placeholder_rngs(obj: Any) -> None:
    """Spaces re-create their RNG lazily when ``_np_random`` is None."""
    import gymnasium as gym

    def clean(space: Any) -> None:
        if isinstance(getattr(space, "_np_random", None), _RNGPlaceholder):
            space._np_random = None
        subs = getattr(space, "spaces", None)
        for sub in (subs.values() if isinstance(subs, dict) else subs or ()):
            clean(sub)

    for key, value in list(vars(obj).items()):
        if isinstance(value, _RNGPlaceholder):
            setattr(obj, key, None)
        elif isinstance(value, gym.spaces.Space):
            clean(value)


def load_vecnormalize(path: str | Path, venv: Any):
    """``VecNormalize.load`` that works across NumPy versions.

    Only the observation/return statistics matter for inference and resuming; saved RNG states
    of the spaces are dropped when they cannot be rebuilt.
    """
    from stable_baselines3.common.vec_env import VecNormalize

    if int(np.__version__.split(".")[0]) >= 2:
        return VecNormalize.load(str(path), venv)
    with numpy2_pickle_compat(), open(path, "rb") as f:
        vec_normalize = _CompatUnpickler(f).load()
    _drop_placeholder_rngs(vec_normalize)
    vec_normalize.set_venv(venv)
    return vec_normalize

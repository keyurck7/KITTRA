"""Shared fixtures. Tests that need MuJoCo/Gymnasium skip cleanly when they are missing, so the
pure-NumPy tests (configs, randomization math, metrics, model consistency) run anywhere."""
from __future__ import annotations

import os

import pytest


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


HAS_MUJOCO = _has("mujoco") and _has("gymnasium")
CAN_RENDER = HAS_MUJOCO and os.environ.get("KITTRA_SKIP_RENDER", "0") != "1"

requires_mujoco = pytest.mark.skipif(not HAS_MUJOCO, reason="needs mujoco + gymnasium")
requires_render = pytest.mark.skipif(not CAN_RENDER, reason="needs an OpenGL context (MUJOCO_GL=egl/osmesa)")

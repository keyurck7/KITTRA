"""Framework-independent randomization and noise models (NumPy + OpenCV only).

The Gymnasium wrappers and the environments use these classes. Keeping them free of MuJoCo and
Gymnasium imports makes them unit-testable on any machine.

Every randomizer stores the *nominal* model values once and writes ``nominal * scale`` on every
call, so randomization never accumulates across episodes, and calling ``apply`` with the
section disabled restores the nominal model exactly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from kittra_sim.transforms import quat_mul, quat_normalize, random_small_rotation


def _uniform(rng: np.random.Generator, lo_hi: Sequence[float], size=None):
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if lo == hi:
        return np.full(size, lo) if size is not None else lo
    return rng.uniform(lo, hi, size=size)


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
@dataclass
class PhysicsRandomizer:
    """Randomizes friction, part mass/inertia, arm joint damping and actuator gains.

    Index arrays select *which* elements are randomized (only parts and finger pads for
    friction, only parts for mass, only arm joints/actuators for damping and kp).
    """

    cfg: Mapping[str, Any]
    friction_geoms: np.ndarray
    mass_bodies: np.ndarray
    damping_dofs: np.ndarray
    kp_actuators: np.ndarray
    nominal: dict[str, np.ndarray] = field(default_factory=dict)
    last_sample: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model, cfg, friction_geoms, mass_bodies, damping_dofs, kp_actuators):
        r = cls(
            cfg=cfg,
            friction_geoms=np.asarray(friction_geoms, dtype=int),
            mass_bodies=np.asarray(mass_bodies, dtype=int),
            damping_dofs=np.asarray(damping_dofs, dtype=int),
            kp_actuators=np.asarray(kp_actuators, dtype=int),
        )
        r.nominal = {
            "friction": np.array(model.geom_friction, dtype=float, copy=True),
            "mass": np.array(model.body_mass, dtype=float, copy=True),
            "inertia": np.array(model.body_inertia, dtype=float, copy=True),
            "damping": np.array(model.dof_damping, dtype=float, copy=True),
            "gainprm": np.array(model.actuator_gainprm, dtype=float, copy=True),
            "biasprm": np.array(model.actuator_biasprm, dtype=float, copy=True),
        }
        return r

    def apply(self, model, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.cfg
        enabled = bool(cfg.get("enabled", False))
        one = (1.0, 1.0)
        n = self.nominal

        f_scale = _uniform(rng, cfg["friction"] if enabled else one, len(self.friction_geoms))
        m_scale = _uniform(rng, cfg["mass"] if enabled else one, len(self.mass_bodies))
        d_scale = _uniform(rng, cfg["damping"] if enabled else one, len(self.damping_dofs))
        k_scale = _uniform(rng, cfg["kp"] if enabled else one, len(self.kp_actuators))

        friction = n["friction"].copy()
        friction[self.friction_geoms, 0] *= f_scale          # sliding friction only
        model.geom_friction[:] = friction

        mass = n["mass"].copy()
        inertia = n["inertia"].copy()
        mass[self.mass_bodies] *= m_scale
        inertia[self.mass_bodies] *= m_scale[:, None]       # same density scaling keeps it physical
        model.body_mass[:] = mass
        model.body_inertia[:] = inertia

        damping = n["damping"].copy()
        damping[self.damping_dofs] *= d_scale
        model.dof_damping[:] = damping

        # MuJoCo position actuator: force = kp * ctrl - kp * qpos  ->  gainprm[0] = kp, biasprm[1] = -kp
        gain = n["gainprm"].copy()
        bias = n["biasprm"].copy()
        gain[self.kp_actuators, 0] *= k_scale
        bias[self.kp_actuators, 1] *= k_scale
        model.actuator_gainprm[:] = gain
        model.actuator_biasprm[:] = bias

        self.last_sample = {"friction": f_scale, "mass": m_scale, "damping": d_scale, "kp": k_scale}
        return self.last_sample


# --------------------------------------------------------------------------- #
# Visuals
# --------------------------------------------------------------------------- #
@dataclass
class VisualRandomizer:
    """Randomizes light position/intensity, object colours and the overhead camera pose."""

    cfg: Mapping[str, Any]
    light_ids: np.ndarray
    color_geoms: np.ndarray       # table, containers, distractors
    target_geoms: np.ndarray      # target part (kept reddish)
    camera_ids: np.ndarray        # cameras whose pose is perturbed
    nominal: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model, cfg, light_ids, color_geoms, target_geoms, camera_ids):
        r = cls(
            cfg=cfg,
            light_ids=np.asarray(light_ids, dtype=int),
            color_geoms=np.asarray(color_geoms, dtype=int),
            target_geoms=np.asarray(target_geoms, dtype=int),
            camera_ids=np.asarray(camera_ids, dtype=int),
        )
        r.nominal = {
            "light_pos": np.array(model.light_pos, dtype=float, copy=True),
            "light_diffuse": np.array(model.light_diffuse, dtype=float, copy=True),
            "geom_rgba": np.array(model.geom_rgba, dtype=float, copy=True),
            "cam_pos": np.array(model.cam_pos, dtype=float, copy=True),
            "cam_quat": np.array(model.cam_quat, dtype=float, copy=True),
        }
        return r

    def apply(self, model, rng: np.random.Generator) -> None:
        n = self.nominal
        light_pos = n["light_pos"].copy()
        light_diffuse = n["light_diffuse"].copy()
        rgba = n["geom_rgba"].copy()
        cam_pos = n["cam_pos"].copy()
        cam_quat = n["cam_quat"].copy()

        if self.cfg.get("enabled", False):
            c = self.cfg
            k = len(self.light_ids)
            light_pos[self.light_ids] += rng.uniform(-1, 1, (k, 3)) * float(c["light_pos_offset"])
            light_diffuse[self.light_ids] *= _uniform(rng, c["light_intensity"], k)[:, None]
            light_diffuse = np.clip(light_diffuse, 0.0, 1.0)

            j = float(c["color_jitter"])
            rgba[self.color_geoms, :3] += rng.uniform(-j, j, (len(self.color_geoms), 3))
            jt = float(c["target_color_jitter"])
            rgba[self.target_geoms, :3] += rng.uniform(-jt, jt, (len(self.target_geoms), 3))
            rgba[:, :3] = np.clip(rgba[:, :3], 0.0, 1.0)

            max_rot = np.deg2rad(float(c["cam_rot_deg"]))
            for cid in self.camera_ids:
                cam_pos[cid] += rng.uniform(-1, 1, 3) * float(c["cam_pos_offset"])
                if max_rot > 0:
                    cam_quat[cid] = quat_normalize(quat_mul(cam_quat[cid], random_small_rotation(rng, max_rot)))

        model.light_pos[:] = light_pos
        model.light_diffuse[:] = light_diffuse
        model.geom_rgba[:] = rgba
        model.cam_pos[:] = cam_pos
        model.cam_quat[:] = cam_quat


# --------------------------------------------------------------------------- #
# Actuator models
# --------------------------------------------------------------------------- #
class DelayLine:
    """Fixed-length FIFO: ``push(x)`` returns the value pushed ``delay`` calls earlier.

    With ``delay = 0`` the input is returned unchanged. The line is pre-filled with ``fill``
    (normally the current joint positions, i.e. "hold still" commands).
    """

    def __init__(self, delay: int, fill: np.ndarray):
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.delay = int(delay)
        fill = np.array(fill, dtype=float, copy=True)
        self._buf: deque[np.ndarray] = deque([fill.copy() for _ in range(self.delay)], maxlen=self.delay + 1)

    def push(self, x: np.ndarray) -> np.ndarray:
        self._buf.append(np.array(x, dtype=float, copy=True))
        return self._buf.popleft() if len(self._buf) > self.delay else self._buf[0]


def backlash_step(prev_output: np.ndarray, command: np.ndarray, play: np.ndarray) -> np.ndarray:
    """Classic backlash (play) model.

    The output only moves once the command has moved more than ``play / 2`` away from it, i.e.
    output = clip(prev_output, command - play/2, command + play/2). With ``play = 0`` the output
    equals the command.
    """
    half = 0.5 * np.asarray(play, dtype=float)
    return np.clip(prev_output, command - half, command + half)


# --------------------------------------------------------------------------- #
# Image noise
# --------------------------------------------------------------------------- #
def apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """``brightness``/``contrast`` are multiplicative factors around 1.0. Works on float images 0..255."""
    mean = img.mean()
    return (img - mean) * contrast + mean * brightness


def apply_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img
    return img + rng.normal(0.0, sigma, img.shape)


def apply_motion_blur(img: np.ndarray, kernel: int, horizontal: bool) -> np.ndarray:
    """Box blur along one axis (approximates camera motion during exposure)."""
    if kernel <= 1:
        return img
    import cv2  # local import keeps the module importable without OpenCV for physics-only use

    ksize = (kernel, 1) if horizontal else (1, kernel)
    return cv2.blur(img.astype(np.float32), ksize)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)

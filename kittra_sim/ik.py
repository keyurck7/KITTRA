"""Damped-least-squares inverse kinematics on MuJoCo Jacobians.

Chosen over an external IK library because it needs nothing beyond MuJoCo, works for any
MJCF (including the official SO-101 model in Phase 3) and is easy to read:

    dq = J^T (J J^T + lambda^2 I)^-1 e

The solver runs on its own ``MjData`` so it never disturbs the simulation state.
The arm has 5 DoF, so a full 6-D pose is generally over-constrained; for the top-down grasps used
here the target orientation is reachable (pan sets the azimuth, wrist_roll sets the yaw), and in
other cases the weighted least-squares solution trades orientation for position accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from kittra_sim.constants import ARM_JOINT_NAMES, TCP_SITE
from kittra_sim.transforms import mat_to_quat


@dataclass
class IKResult:
    qpos: np.ndarray
    success: bool
    pos_error: float
    rot_error: float
    iterations: int


class DLSIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site: str = TCP_SITE,
        joint_names: Sequence[str] = ARM_JOINT_NAMES,
        damping: float = 1e-2,
        rot_weight: float = 0.3,
        max_step: float = 0.2,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        jids = [model.joint(n).id for n in joint_names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.damping = damping
        self.rot_weight = rot_weight
        self.max_step = max_step
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site position and rotation matrix for arm configuration ``q``."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def _rot_error(self, target_quat: np.ndarray) -> np.ndarray:
        cur_quat = np.empty(4)
        mujoco.mju_mat2Quat(cur_quat, self.data.site_xmat[self.site_id])
        cur_conj = np.empty(4)
        mujoco.mju_negQuat(cur_conj, cur_quat)
        err_quat = np.empty(4)
        mujoco.mju_mulQuat(err_quat, target_quat, cur_conj)
        err = np.empty(3)
        mujoco.mju_quat2Vel(err, err_quat, 1.0)
        return err

    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_mat: np.ndarray | None = None,
        max_iters: int = 100,
        tol_pos: float = 1e-3,
        tol_rot: float = 2e-2,
    ) -> IKResult:
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.lo, self.hi)
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = mat_to_quat(target_mat) if target_mat is not None else None
        pos_err_n = rot_err_n = np.inf
        it = 0
        for it in range(1, max_iters + 1):
            self.forward(q)
            pos_err = target_pos - self.data.site_xpos[self.site_id]
            pos_err_n = float(np.linalg.norm(pos_err))
            if target_quat is not None:
                rot_err = self._rot_error(target_quat)
                rot_err_n = float(np.linalg.norm(rot_err))
            else:
                rot_err, rot_err_n = np.zeros(3), 0.0
            if pos_err_n < tol_pos and rot_err_n < tol_rot:
                return IKResult(q, True, pos_err_n, rot_err_n, it)

            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            jp = self._jacp[:, self.dadr]
            if target_quat is not None:
                J = np.vstack([jp, self.rot_weight * self._jacr[:, self.dadr]])
                e = np.concatenate([pos_err, self.rot_weight * rot_err])
            else:
                J, e = jp, pos_err
            JJt = J @ J.T + (self.damping**2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            norm = np.linalg.norm(dq)
            if norm > self.max_step:
                dq *= self.max_step / norm
            q = np.clip(q + dq, self.lo, self.hi)
        self.forward(q)
        return IKResult(q, False, pos_err_n, rot_err_n, it)

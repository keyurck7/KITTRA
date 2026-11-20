"""Scripted expert for KittraPlace-v0 (privileged). Used for behaviour-cloning data and as a sanity
baseline for the PPO placement policy.

Strategy: align the part over the slot at the current height, descend to just above the floor,
release. Commands are converted into the env's delta action space.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from kittra_sim import constants as C
from kittra_sim.transforms import gripper_yaw_from_mat, topdown_gripper_mat


class ScriptedPlaceExpert:
    def __init__(self, env: gym.Env, align_tol: float = 0.003, release_clearance: float = 0.006):
        self.base = env.unwrapped
        self.align_tol = align_tol
        self.release_clearance = release_clearance
        self.yaw = 0.0

    def reset(self, obs=None, info=None) -> None:
        q = self.base.current_command()[:5]
        _, mat = self.base.ik.forward(q)
        self.yaw = gripper_yaw_from_mat(mat)

    def act(self, obs, info) -> np.ndarray:
        b = self.base
        part = np.asarray(info["target_pos"])
        slot = np.asarray(info["slot_pos"])
        tcp = np.asarray(info["tcp_pos"])
        grasp_offset = tcp - part                       # where the part sits relative to the TCP
        target_xy = slot[:2] + grasp_offset[:2]
        xy_err = np.linalg.norm(part[:2] - slot[:2])
        release_part_z = slot[2] + C.TARGET_HALF_SIZE + self.release_clearance

        if xy_err > self.align_tol:
            target = np.array([target_xy[0], target_xy[1], tcp[2]])
        else:
            target = np.array([target_xy[0], target_xy[1], release_part_z + grasp_offset[2]])

        if xy_err <= self.align_tol and part[2] <= release_part_z + 0.002:
            action = np.zeros(6)
            action[5] = 1.0                              # release
            return action.astype(np.float32)

        cmd = b.current_command()
        res = b.ik.solve(cmd[:5], target, topdown_gripper_mat(self.yaw), max_iters=50)
        action = np.zeros(6)
        action[:5] = np.clip((res.qpos - cmd[:5]) / b.delta_scale, -1.0, 1.0)
        action[5] = -1.0                                 # keep holding
        return action.astype(np.float32)

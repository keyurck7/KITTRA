"""Scripted pick-and-place expert used to generate demonstrations.

It uses privileged ground truth from ``info`` (the true part pose), which is intentional: the
expert only produces training data, while the learned policies see camera images and joint
states only.

Motion: a sequence of Cartesian segments for the tool centre point (TCP), each interpolated with
a minimum-jerk profile and converted to joint targets by damped-least-squares IK at every control
step (warm-started from the previous solution, so the elbow-up branch is kept).

    approach -> descend -> close -> settle -> lift -> transport -> lower -> release -> retreat -> hold

The approach and descend targets are refreshed from ``info`` until the gripper closes, so a part
that was nudged during the approach is still grasped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from kittra_sim import constants as C
from kittra_sim.ik import DLSIK
from kittra_sim.transforms import (
    closest_symmetric_angle,
    gripper_yaw_from_mat,
    quat_to_yaw,
    topdown_gripper_mat,
    wrap_angle,
)


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass
class ExpertParams:
    pregrasp_height: float = 0.07     # m above the part centre
    grasp_offset: float = C.GRASP_OFFSET_Z  # m, TCP above the part centre at grasp
    lift_height: float = 0.10         # m above the grasp height
    transport_height: float = 0.12    # m, absolute TCP height while moving to the kit box
    release_clearance: float = 0.012  # m between part bottom and kit floor at release
    retreat_height: float = 0.06      # m
    durations: dict = field(default_factory=lambda: {
        "approach": 40, "descend": 25, "close": 12, "settle": 6, "lift": 25,
        "transport": 45, "lower": 25, "release": 10, "retreat": 20,
    })


@dataclass
class Segment:
    name: str
    steps: int
    start_pos: np.ndarray
    end_pos: np.ndarray
    start_yaw: float
    end_yaw: float
    grip_start: float
    grip_end: float


class ScriptedPickPlaceExpert:
    PHASES = ("approach", "descend", "close", "settle", "lift", "transport", "lower", "release", "retreat")

    def __init__(self, env: gym.Env, params: ExpertParams | None = None):
        self.base = env.unwrapped
        self.p = params or ExpertParams()
        self.ik = DLSIK(self.base.model)
        self.segments: list[Segment] = []
        self.seg_idx = 0
        self.t = 0
        self.q = np.array(C.HOME_QPOS[:5], dtype=float)
        self.ik_failures = 0

    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> str:
        return self.segments[self.seg_idx].name if self.seg_idx < len(self.segments) else "hold"

    def reset(self, obs=None, info=None) -> None:
        if info is None:
            raise ValueError("the scripted expert needs the env info dict (privileged part pose)")
        self.q = np.asarray(self.base.joint_positions()[:5], dtype=float)
        tcp_pos, tcp_mat = self.ik.forward(self.q)
        self.ik_failures = 0
        self._plan(tcp_pos, gripper_yaw_from_mat(tcp_mat), info)
        self.seg_idx, self.t = 0, 0

    def _grasp_yaw(self, part_pos: np.ndarray, part_quat: np.ndarray) -> float:
        # Natural gripper yaw for this azimuth (pan rotates the finger axis): azimuth + 90 deg.
        natural = np.arctan2(part_pos[1], part_pos[0]) + np.pi / 2
        # Cube faces repeat every 90 deg; pick the face pair closest to the natural yaw.
        return closest_symmetric_angle(quat_to_yaw(part_quat), natural, np.pi / 2)

    def _plan(self, start_pos: np.ndarray, start_yaw: float, info) -> None:
        p, d = self.p, self.p.durations
        part = np.asarray(info["target_pos"], dtype=float)
        yaw_g = self._grasp_yaw(part, info["target_quat"])
        slot = np.asarray(info["slot_pos"], dtype=float)
        yaw_k_nat = np.arctan2(slot[1], slot[0]) + np.pi / 2
        yaw_k = closest_symmetric_angle(yaw_g, yaw_k_nat, np.pi / 2)

        pre = part + [0, 0, p.pregrasp_height]
        grasp = part + [0, 0, p.grasp_offset]
        lift = grasp + [0, 0, p.lift_height]
        above_kit = np.array([slot[0], slot[1], max(p.transport_height, lift[2])])
        # While holding, the part centre is grasp_offset below the TCP: centre = floor + half size + clearance.
        release = np.array([slot[0], slot[1], slot[2] + C.TARGET_HALF_SIZE + p.release_clearance + p.grasp_offset])
        retreat = release + [0, 0, p.retreat_height]
        o, c = C.GRIPPER_OPEN, C.GRIPPER_CLOSED
        start_yaw = closest_symmetric_angle(start_yaw, yaw_g, 2 * np.pi)

        self.segments = [
            Segment("approach", d["approach"], start_pos, pre, start_yaw, yaw_g, o, o),
            Segment("descend", d["descend"], pre, grasp, yaw_g, yaw_g, o, o),
            Segment("close", d["close"], grasp, grasp, yaw_g, yaw_g, o, c),
            Segment("settle", d["settle"], grasp, grasp, yaw_g, yaw_g, c, c),
            Segment("lift", d["lift"], grasp, lift, yaw_g, yaw_g, c, c),
            Segment("transport", d["transport"], lift, above_kit, yaw_g, yaw_k, c, c),
            Segment("lower", d["lower"], above_kit, release, yaw_k, yaw_k, c, c),
            Segment("release", d["release"], release, release, yaw_k, yaw_k, c, o),
            Segment("retreat", d["retreat"], release, retreat, yaw_k, yaw_k, o, o),
        ]

    def _refresh_grasp_target(self, info) -> None:
        """Update approach/descend end points (and everything that depends on them) from the latest part pose."""
        seg0 = self.segments[0]
        cur_seg = self.segments[self.seg_idx]
        keep_start, keep_yaw = cur_seg.start_pos, cur_seg.start_yaw
        saved_idx, saved_t = self.seg_idx, self.t
        self._plan(seg0.start_pos, seg0.start_yaw, info)
        # Keep the current segment's start so the motion stays continuous.
        self.segments[saved_idx].start_pos = keep_start
        self.segments[saved_idx].start_yaw = keep_yaw
        self.seg_idx, self.t = saved_idx, saved_t

    # ------------------------------------------------------------------ #
    def act(self, obs, info) -> np.ndarray:
        if self.phase in ("approach", "descend"):
            self._refresh_grasp_target(info)

        if self.seg_idx < len(self.segments):
            seg = self.segments[self.seg_idx]
            s = min_jerk((self.t + 1) / seg.steps)
            pos = seg.start_pos + s * (seg.end_pos - seg.start_pos)
            yaw = seg.start_yaw + s * wrap_angle(seg.end_yaw - seg.start_yaw)
            grip = seg.grip_start + s * (seg.grip_end - seg.grip_start)
            self.t += 1
            if self.t >= seg.steps:
                self.seg_idx, self.t = self.seg_idx + 1, 0
        else:  # hold the final pose with the gripper open
            last = self.segments[-1]
            pos, yaw, grip = last.end_pos, last.end_yaw, last.grip_end

        res = self.ik.solve(self.q, pos, topdown_gripper_mat(yaw))
        if not res.success:
            self.ik_failures += 1
        self.q = res.qpos
        return np.concatenate([self.q, [grip]]).astype(np.float32)

    @property
    def total_steps(self) -> int:
        return sum(self.p.durations[n] for n in self.PHASES)

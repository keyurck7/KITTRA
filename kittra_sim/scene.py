"""Scene handles (ids, addresses, task geometry) and part spawning.

All task geometry is read from named sites in the MJCF, so the Python code never duplicates
numbers from the XML.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from kittra_sim import constants as C
from kittra_sim.transforms import yaw_quat


@dataclass
class Box2D:
    """Axis-aligned rectangle on the table: centre (x, y), half extents (hx, hy), surface height z."""

    center: np.ndarray
    half: np.ndarray
    z: float

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        d = np.abs(np.asarray(xy[:2]) - self.center)
        return bool(np.all(d <= self.half - margin))

    def sample(self, rng: np.random.Generator, margin: float = 0.0) -> np.ndarray:
        h = np.maximum(self.half - margin, 0.0)
        return self.center + rng.uniform(-h, h)


class SceneHandles:
    """Pre-computed ids and addresses for fast access in the control loop."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        jids = [model.joint(n).id for n in C.JOINT_NAMES]
        self.joint_qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.joint_dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.mirror_qadr = model.jnt_qposadr[model.joint(C.GRIPPER_MIRROR_JOINT).id]
        self.mirror_dadr = model.jnt_dofadr[model.joint(C.GRIPPER_MIRROR_JOINT).id]
        self.actuator_ids = np.array([model.actuator(n).id for n in C.JOINT_NAMES])
        self.ctrl_lo = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.tcp_site = model.site(C.TCP_SITE).id

        self.part_body = {n: model.body(n).id for n in C.ALL_PARTS}
        self.part_qadr = {n: model.jnt_qposadr[model.body(n).jntadr[0]] for n in C.ALL_PARTS}
        self.part_dadr = {n: model.jnt_dofadr[model.body(n).jntadr[0]] for n in C.ALL_PARTS}
        self.part_geom = {n: model.geom(f"{n}_geom").id for n in C.ALL_PARTS}
        # Half height of each part above its centre (box: size[2], cylinder: size[1]).
        self.part_half_height = {}
        for n, g in self.part_geom.items():
            size = model.geom_size[g]
            self.part_half_height[n] = float(size[2] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else size[1])

        self.finger_geoms = [model.geom("finger_left_geom").id, model.geom("finger_right_geom").id]
        self.arm_dofs = self.joint_dadr[:5]
        self.arm_actuators = self.actuator_ids[:5]

        self.site_ids = {n: model.site(n).id for n in (C.TRAY_SPAWN_SITE, C.KIT_AREA_SITE, C.KIT_SLOT_SITE, C.REJECT_AREA_SITE)}
        self.camera_ids = {n: model.camera(n).id for n in C.CAMERAS}
        self.light_ids = [model.light("main_light").id]

        # Geoms whose colour is randomized (everything static and visible + distractors).
        color_names = ["table", "tray_floor", "kit_floor", "reject_floor",
                       "part_distractor_0_geom", "part_distractor_1_geom"]
        color_names += [f"{c}_wall_{s}" for c in ("tray", "kit", "reject") for s in ("px", "nx", "py", "ny")]
        self.color_geoms = [model.geom(n).id for n in color_names]
        self.target_geoms = [self.part_geom[C.TARGET_PART]]

    # ------------------------------------------------------------------ #
    def area(self, data: mujoco.MjData, site_name: str) -> Box2D:
        sid = self.site_ids[site_name]
        pos = data.site_xpos[sid]
        size = self.model.site_size[sid]
        return Box2D(center=pos[:2].copy(), half=size[:2].copy(), z=float(pos[2]))

    def part_pose(self, data: mujoco.MjData, name: str = C.TARGET_PART) -> tuple[np.ndarray, np.ndarray]:
        a = self.part_qadr[name]
        return data.qpos[a : a + 3].copy(), data.qpos[a + 3 : a + 7].copy()

    def part_speed(self, data: mujoco.MjData, name: str = C.TARGET_PART) -> float:
        d = self.part_dadr[name]
        return float(np.linalg.norm(data.qvel[d : d + 3]))

    def set_part_pose(self, data: mujoco.MjData, name: str, pos, quat) -> None:
        a, d = self.part_qadr[name], self.part_dadr[name]
        data.qpos[a : a + 3] = pos
        data.qpos[a + 3 : a + 7] = quat
        data.qvel[d : d + 6] = 0.0

    def set_arm_qpos(self, data: mujoco.MjData, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        data.qpos[self.joint_qadr] = q
        data.qpos[self.mirror_qadr] = q[5]
        data.qvel[self.joint_dadr] = 0.0
        data.qvel[self.mirror_dadr] = 0.0
        data.ctrl[self.actuator_ids] = np.clip(q, self.ctrl_lo, self.ctrl_hi)

    def spawn_parts(self, data: mujoco.MjData, rng: np.random.Generator, n_distractors: int,
                    min_distance: float, max_layouts: int = 100, max_tries: int = 50) -> None:
        """Place the target and ``n_distractors`` distractors at random, non-overlapping tray positions.

        Whole layouts are re-sampled when a part does not fit, so an unlucky first placement can
        never make the remaining parts impossible to place.
        """
        tray = self.area(data, C.TRAY_SPAWN_SITE)
        names = [C.TARGET_PART] + list(C.DISTRACTOR_PARTS[:n_distractors])
        for _ in range(max_layouts):
            placed: list[np.ndarray] = []
            for _name in names:
                for _ in range(max_tries):
                    xy = tray.sample(rng)
                    if all(np.linalg.norm(xy - p) >= min_distance for p in placed):
                        placed.append(xy)
                        break
                else:
                    break  # this layout failed, start a new one
            if len(placed) == len(names):
                break
        else:
            raise RuntimeError("could not place parts without overlap; reduce n_distractors or min_part_distance")
        for name, xy in zip(names, placed):
            z = tray.z + self.part_half_height[name] + 0.002
            self.set_part_pose(data, name, [xy[0], xy[1], z], yaw_quat(rng.uniform(-np.pi, np.pi)))
        for name, park in zip(C.DISTRACTOR_PARTS[n_distractors:], C.PARK_POSITIONS[n_distractors:]):
            self.set_part_pose(data, name, park, [1.0, 0.0, 0.0, 0.0])

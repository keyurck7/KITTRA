"""KittraPlace-v0: precise placement of an already-grasped part into the kit box slot (RL sub-skill).

Start state
    The target cube is held in the gripper; the TCP is at a random pose 6-10 cm above the kit box,
    up to 4 cm off the slot horizontally, with a random yaw.

Observation (22,) float32
    joint_pos (6) | joint_vel (6) | slot - tcp (3) | part quaternion (4) | slot - part (3)

Action (6,) in [-1, 1]
    a[0:5] * delta_scale = arm joint target deltas [rad];  a[5] > 0 opens the gripper (release).

Release is a one-shot decision: when the policy opens the gripper the env simulates until the
part settles and ends the episode with success if the part rests in the kit box within
``success_tol`` of the slot.

Reward
    -|part - slot| - 0.01 |a|^2 per step, +10 on success, -5 on failure (bad release or dropped part).

Sensor and actuator models (joint noise/bias, latency, backlash) are applied inside the env on
*absolute* joint targets, with the same core classes the pick-and-place wrappers use.
"""
from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import mujoco
import numpy as np

from kittra_sim import constants as C
from kittra_sim.config import load_config
from kittra_sim.ik import DLSIK
from kittra_sim.randomization.core import DelayLine, PhysicsRandomizer, backlash_step
from kittra_sim.scene import SceneHandles
from kittra_sim.transforms import closest_symmetric_angle, topdown_gripper_mat, yaw_quat


class KittraPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": C.CONTROL_HZ}
    OBS_DIM = 22

    def __init__(
        self,
        config: str | Mapping[str, Any] | None = None,
        delta_scale: float = 0.05,
        max_episode_steps: int = 150,
        success_tol: float = 0.01,
        release_settle_steps: int = 20,
        render_mode: str | None = None,
    ):
        self.cfg = load_config(config)
        self.delta_scale = float(delta_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.success_tol = float(success_tol)
        self.release_settle_steps = int(release_settle_steps)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(C.SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.h = SceneHandles(self.model)
        self.ik = DLSIK(self.model)
        self.n_substeps = max(1, int(round(C.CONTROL_DT / self.model.opt.timestep)))
        self.physics_rand = PhysicsRandomizer.from_model(
            self.model, self.cfg["physics"],
            friction_geoms=list(self.h.part_geom.values()) + self.h.finger_geoms,
            mass_bodies=list(self.h.part_body.values()),
            damping_dofs=self.h.arm_dofs,
            kp_actuators=self.h.arm_actuators,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS_DIM,), np.float32)
        self._renderer = None
        self._step_count = 0
        self._cmd = np.array(C.HOME_QPOS, dtype=float)
        self._sent = self._cmd.copy()

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.physics_rand.apply(self.model, rng)
        mujoco.mj_setConst(self.model, self.data)

        for _attempt in range(10):
            if self._try_start_state(rng):
                break
        else:
            raise RuntimeError("could not create a stable grasped start state")

        self._sample_episode_noise(rng)
        self._step_count = 0
        return self._obs(), self._info()

    def _try_start_state(self, rng: np.random.Generator) -> bool:
        mujoco.mj_resetData(self.model, self.data)
        self.h.set_arm_qpos(self.data, np.array(C.HOME_QPOS))
        mujoco.mj_kinematics(self.model, self.data)
        slot = self.data.site_xpos[self.h.site_ids[C.KIT_SLOT_SITE]].copy()

        offset = np.array([*rng.uniform(-0.04, 0.04, 2), rng.uniform(0.06, 0.10)])
        tcp_target = slot + offset
        natural = np.arctan2(tcp_target[1], tcp_target[0]) + np.pi / 2
        yaw = closest_symmetric_angle(natural + rng.uniform(-np.pi / 4, np.pi / 4), natural, 2 * np.pi)
        res = self.ik.solve(np.array(C.HOME_QPOS[:5]), tcp_target, topdown_gripper_mat(yaw))
        if not res.success:
            return False

        q = np.concatenate([res.qpos, [C.GRIPPER_OPEN]])
        self.h.set_arm_qpos(self.data, q)
        mujoco.mj_kinematics(self.model, self.data)
        tcp = self.data.site_xpos[self.h.tcp_site].copy()
        part_quat = yaw_quat(yaw)  # cube faces aligned with the finger opening direction
        for name, park in zip(C.DISTRACTOR_PARTS, C.PARK_POSITIONS):
            self.h.set_part_pose(self.data, name, park, [1.0, 0.0, 0.0, 0.0])

        # Close the fingers while pinning the part at the TCP, then let go and let the grasp settle.
        self.data.ctrl[self.h.actuator_ids[5]] = C.GRIPPER_CLOSED
        part_start = tcp - np.array([0.0, 0.0, C.GRASP_OFFSET_Z])  # same grasp as the scripted expert
        for i in range(160):
            if i < 40:
                self.h.set_part_pose(self.data, C.TARGET_PART, part_start, part_quat)
            mujoco.mj_step(self.model, self.data)
        part_pos, _ = self.h.part_pose(self.data)
        tcp = self.data.site_xpos[self.h.tcp_site]
        if np.linalg.norm(part_pos - tcp) > 0.02:
            return False
        self._cmd = np.concatenate([res.qpos, [C.GRIPPER_CLOSED]])
        self._sent = self._cmd.copy()
        return True

    def _sample_episode_noise(self, rng: np.random.Generator) -> None:
        s, a = self.cfg["sensor"], self.cfg["actuator"]
        self._joint_sigma = float(s["joint_sigma"]) if s["enabled"] else 0.0
        bias = float(s["joint_bias"]) if s["enabled"] else 0.0
        self._bias = rng.uniform(-bias, bias, 6)
        delay = int(rng.integers(int(a["min_delay_steps"]), int(a["max_delay_steps"]) + 1)) if a["enabled"] else 0
        self._delay_line = DelayLine(delay, self._cmd)
        play = float(a["backlash"]) if a["enabled"] else 0.0
        self._play = np.concatenate([rng.uniform(0.0, play, 5), [0.0]])
        self._noise_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(6), -1.0, 1.0)
        release = bool(a[5] > 0.0)
        arm = np.clip(self._cmd[:5] + a[:5] * self.delta_scale, self.h.ctrl_lo[:5], self.h.ctrl_hi[:5])
        self._cmd = np.concatenate([arm, [C.GRIPPER_OPEN if release else C.GRIPPER_CLOSED]])
        self._apply_command(self._cmd)
        self._step_count += 1

        info = self._info()
        reward = -info["part_slot_dist"] - 0.01 * float(a @ a)
        terminated = False

        if release:
            # Hold the arm, open the gripper and let the part settle; then judge the placement.
            for _ in range(self.release_settle_steps):
                self._apply_command(self._cmd)
            info = self._info()
            success = info["placement_error"] < self.success_tol and info["in_kit"] and info["resting"]
            info["is_success"] = bool(success)
            reward += 10.0 if success else -5.0
            terminated = True
        elif info["part_dropped"]:
            reward -= 5.0
            terminated = True

        truncated = bool(not terminated and self._step_count >= self.max_episode_steps)
        return self._obs(), float(reward), terminated, truncated, info

    def _apply_command(self, cmd: np.ndarray) -> None:
        delayed = self._delay_line.push(cmd)
        self._sent = backlash_step(self._sent, delayed, self._play)
        self.data.ctrl[self.h.actuator_ids] = np.clip(self._sent, self.h.ctrl_lo, self.h.ctrl_hi)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self.h.joint_qadr] + self._bias + self._noise_rng.normal(0.0, self._joint_sigma, 6)
        qd = self.data.qvel[self.h.joint_dadr]
        tcp = self.data.site_xpos[self.h.tcp_site]
        slot = self.data.site_xpos[self.h.site_ids[C.KIT_SLOT_SITE]]
        part_pos, part_quat = self.h.part_pose(self.data)
        return np.concatenate([q, qd, slot - tcp, part_quat, slot - part_pos]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        part_pos, part_quat = self.h.part_pose(self.data)
        slot = self.data.site_xpos[self.h.site_ids[C.KIT_SLOT_SITE]].copy()
        kit = self.h.area(self.data, C.KIT_AREA_SITE)
        tcp = self.data.site_xpos[self.h.tcp_site].copy()
        held = np.linalg.norm(part_pos - tcp) < 0.03
        return {
            "is_success": False,
            "placement_error": float(np.linalg.norm(part_pos[:2] - slot[:2])),
            "part_slot_dist": float(np.linalg.norm(part_pos - (slot + [0, 0, C.TARGET_HALF_SIZE]))),
            "in_kit": bool(kit.contains(part_pos[:2], margin=0.5 * C.TARGET_HALF_SIZE) and part_pos[2] < kit.z + 3 * C.TARGET_HALF_SIZE),
            "resting": self.h.part_speed(self.data) < 0.03,
            "part_dropped": bool(not held and self._cmd[5] == C.GRIPPER_CLOSED),
            "target_pos": part_pos,
            "target_quat": part_quat,
            "slot_pos": slot,
            "tcp_pos": tcp,
            "n_steps": self._step_count,
        }

    def current_command(self) -> np.ndarray:
        return self._cmd.copy()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        self._renderer.update_scene(self.data, camera=C.OVERHEAD_CAMERA)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

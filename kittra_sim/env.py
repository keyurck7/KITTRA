"""KittraPickPlace-v0: pick the red cube from the tray and place it in the kit box.

Observation (dict)
    joint_pos     (6,)  float32  measured joint positions [rad, gripper in m]
    joint_vel     (6,)  float32
    ee_pos        (3,)  float32  tool-centre-point position (forward kinematics)
    overhead_rgb  (H, W, 3) uint8   (obs_mode="rgb" only)
    wrist_rgb     (H, W, 3) uint8   (obs_mode="rgb" only)

Privileged ground truth (target pose, kit position, ...) is returned in ``info`` only, never in
the observation, so a policy trained on observations cannot cheat. The scripted expert reads
``info``.

Action
    (6,) absolute joint position targets, clipped to the actuator ranges. This matches the action
    space LeRobot uses for the SO-101.
"""
from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import mujoco
import numpy as np

from kittra_sim import constants as C
from kittra_sim.config import load_config
from kittra_sim.randomization.core import PhysicsRandomizer, VisualRandomizer
from kittra_sim.scene import SceneHandles


class KittraPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": C.CONTROL_HZ}

    def __init__(
        self,
        config: str | Mapping[str, Any] | None = None,
        obs_mode: str = "rgb",
        image_size: int = 128,
        render_mode: str | None = None,
        scene_path: str | None = None,
    ):
        if obs_mode not in ("rgb", "state"):
            raise ValueError("obs_mode must be 'rgb' or 'state'")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render_mode {render_mode}")
        self.cfg = load_config(config)
        self.obs_mode = obs_mode
        self.image_size = int(image_size)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(scene_path or C.SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.h = SceneHandles(self.model)

        self.n_substeps = max(1, int(round(C.CONTROL_DT / self.model.opt.timestep)))
        self.control_dt = self.n_substeps * self.model.opt.timestep

        self.physics_rand = PhysicsRandomizer.from_model(
            self.model, self.cfg["physics"],
            friction_geoms=list(self.h.part_geom.values()) + self.h.finger_geoms,
            mass_bodies=list(self.h.part_body.values()),
            damping_dofs=self.h.arm_dofs,
            kp_actuators=self.h.arm_actuators,
        )
        self.visual_rand = VisualRandomizer.from_model(
            self.model, self.cfg["visual"],
            light_ids=self.h.light_ids,
            color_geoms=self.h.color_geoms,
            target_geoms=self.h.target_geoms,
            camera_ids=[self.h.camera_ids[C.OVERHEAD_CAMERA]],
        )

        self.action_space = gym.spaces.Box(self.h.ctrl_lo.astype(np.float32), self.h.ctrl_hi.astype(np.float32),
                                           dtype=np.float32)
        spaces = {
            "joint_pos": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "joint_vel": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "ee_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        }
        if obs_mode == "rgb":
            img = (self.image_size, self.image_size, 3)
            spaces["overhead_rgb"] = gym.spaces.Box(0, 255, img, np.uint8)
            spaces["wrist_rgb"] = gym.spaces.Box(0, 255, img, np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

        self._renderer: mujoco.Renderer | None = None
        self._render_renderer: mujoco.Renderer | None = None
        self._viewer = None
        self._step_count = 0
        self._cmd = np.array(C.HOME_QPOS, dtype=float)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        task = self.cfg["task"]

        # 1. Model randomization (always writes nominal * scale, so disabled = nominal).
        self.physics_rand.apply(self.model, rng)
        self.visual_rand.apply(self.model, rng)
        mujoco.mj_setConst(self.model, self.data)

        # 2. State.
        mujoco.mj_resetData(self.model, self.data)
        q0 = np.array(C.HOME_QPOS, dtype=float)
        q0[:5] += rng.uniform(-1, 1, 5) * float(task["init_qpos_noise"])
        q0 = np.clip(q0, self.h.ctrl_lo, self.h.ctrl_hi)
        self.h.set_arm_qpos(self.data, q0)
        self._cmd = q0.copy()
        mujoco.mj_kinematics(self.model, self.data)  # site positions are needed for spawning
        self.h.spawn_parts(self.data, rng, int(task["n_distractors"]), float(task["min_part_distance"]))
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(task["settle_steps"])):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        info = self._info()
        self._prev_shaping = self._shaping(info)
        return self._get_obs(), info

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(6)
        a = np.clip(a, self.h.ctrl_lo, self.h.ctrl_hi)
        self._cmd = a
        self.data.ctrl[self.h.actuator_ids] = a
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        if not np.all(np.isfinite(self.data.qpos)):
            # Simulation blew up: end the episode cleanly instead of propagating NaNs.
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            info = self._info()
            info["sim_unstable"] = True
            return self._get_obs(), 0.0, True, False, info

        info = self._info()
        success = info["is_success"]
        dropped = info["part_dropped"]
        terminated = bool(success or dropped)
        truncated = bool(not terminated and self._step_count >= int(self.cfg["task"]["max_episode_steps"]))

        if self.cfg["task"]["reward"] == "sparse":
            reward = float(success)
        else:
            shaping = self._shaping(info)
            reward = shaping - self._prev_shaping + 10.0 * float(success) - 5.0 * float(dropped)
            self._prev_shaping = shaping

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._render_renderer is None:
                self._render_renderer = mujoco.Renderer(self.model, 480, 480)
            self._render_renderer.update_scene(self.data, camera=C.OVERHEAD_CAMERA)
            return self._render_renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        for r in (self._renderer, self._render_renderer):
            if r is not None:
                r.close()
        self._renderer = self._render_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def current_command(self) -> np.ndarray:
        """Last joint command sent to the actuators (used by the actuator wrappers)."""
        return self._cmd.copy()

    def joint_positions(self) -> np.ndarray:
        q = self.data.qpos[self.h.joint_qadr].copy()
        return q

    def _render_camera(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {
            "joint_pos": self.data.qpos[self.h.joint_qadr].astype(np.float32),
            "joint_vel": self.data.qvel[self.h.joint_dadr].astype(np.float32),
            "ee_pos": self.data.site_xpos[self.h.tcp_site].astype(np.float32),
        }
        if self.obs_mode == "rgb":
            obs["overhead_rgb"] = self._render_camera(C.OVERHEAD_CAMERA)
            obs["wrist_rgb"] = self._render_camera(C.WRIST_CAMERA)
        return obs

    def _info(self) -> dict[str, Any]:
        pos, quat = self.h.part_pose(self.data)
        kit = self.h.area(self.data, C.KIT_AREA_SITE)
        slot = self.data.site_xpos[self.h.site_ids[C.KIT_SLOT_SITE]].copy()
        tcp = self.data.site_xpos[self.h.tcp_site].copy()
        gripper = float(self.data.qpos[self.h.joint_qadr[5]])
        half = C.TARGET_HALF_SIZE

        in_kit = kit.contains(pos[:2], margin=0.5 * half) and pos[2] < kit.z + 3.0 * half
        resting = self.h.part_speed(self.data) < 0.03
        released = gripper > 0.6 * C.GRIPPER_OPEN and np.linalg.norm(tcp - pos) > half
        success = bool(in_kit and resting and released)
        dropped = bool(pos[2] < -0.05)  # fell off the table

        return {
            "is_success": success,
            "part_dropped": dropped,
            "target_pos": pos,
            "target_quat": quat,
            "kit_center": np.array([kit.center[0], kit.center[1], kit.z]),
            "slot_pos": slot,
            "tcp_pos": tcp,
            "gripper_opening": gripper,
            "placement_error": float(np.linalg.norm(pos[:2] - slot[:2])),
            "n_steps": self._step_count,
        }

    def _shaping(self, info: Mapping[str, Any]) -> float:
        """Potential for the optional shaped reward (difference of potentials keeps the optimum)."""
        reach = np.linalg.norm(info["tcp_pos"] - info["target_pos"])
        place = np.linalg.norm(info["target_pos"][:2] - info["slot_pos"][:2])
        return -float(reach) - 2.0 * float(place)

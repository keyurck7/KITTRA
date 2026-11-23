# Kittra

**A language-instructed robotic kitting and inspection cell for high-mix, low-volume manufacturing.**

> Status: **Phase 1 ✅ Simulated cell · Phase 2 ✅ Learned skills in simulation · Phases 3–4 planned**

[![CI](https://github.com/keyurck7/KITTRA/actions/workflows/ci.yml/badge.svg)](https://github.com/keyurck7/KITTRA/actions/workflows/ci.yml)

---

## Why this project

Small and medium-sized manufacturers (the German *Mittelstand* in particular) face a shortage of skilled workers, and much of their work is high-mix, low-volume: many different parts in small batches. Classical industrial robots rarely pay off in that setting because every new task requires an integrator to reprogram them.

Kittra explores the alternative the industry is moving towards: a robot cell that a non-expert can instruct **in natural language** and teach **by demonstration**, without writing robot code.

**Target end-to-end behaviour (after Phase 4):**

> Operator: *"Prepare kit for order 17: two M4 screws, one bracket, one spacer. Reject anything damaged."*
>
> Kittra: inspects the tray → picks the correct parts into the kit box → puts defective parts in the reject bin → *"Kit 17 complete. 1 bracket rejected (bent)."*

---

## Project roadmap

| Phase | Focus | Status |
|---|---|---|
| **1** | Simulated cell: MuJoCo scene, Gymnasium env, IK-based scripted expert, ROS 2 bridge | ✅ Done |
| **2** | Learned skills in simulation: imitation learning (ACT, Diffusion Policy), RL (PPO) vs. behaviour cloning, sensor/actuator modeling, domain randomization, held-out evaluation | ✅ Done |
| 3 | Real hardware: SO-101 arms, teleoperated demos, VLA fine-tuning, sim-to-real, system identification | 🗓️ Planned |
| 4 | Language agent + visual inspection: VLM planner, open-vocabulary perception, defect classifier | 🗓️ Planned |

---

## Quick start

```bash
git clone https://github.com/keyurck7/KITTRA.git kittra && cd kittra
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
make install          # Phase 1 dependencies (laptop CPU is enough)
make test-fast        # unit + environment tests
make expert           # scripted expert, 50 episodes nominal + randomized
python scripts/view_scene.py
```

`make help` lists every workflow (data recording, training, evaluation, result tables).

---

## Tech stack

| Layer | Tool | Used in |
|---|---|---|
| OS | Ubuntu 24.04 LTS (native or WSL2) | 1–4 |
| Middleware | ROS 2 Jazzy Jalisco (LTS): rclpy, robot_state_publisher, tf2_ros, RViz2 | 1–4 |
| Simulator | MuJoCo 3.x (Python bindings), headless rendering via EGL/OSMesa | 1–3 |
| Environment API | Gymnasium 1.x | 1–2 |
| Inverse kinematics | Damped least squares on MuJoCo Jacobians (`kittra_sim/ik.py`) | 1–2 |
| Imitation learning | LeRobot (ACT, Diffusion Policy), LeRobot dataset format | 2–3 |
| Reinforcement learning | Stable-Baselines3 (PPO + VecNormalize) | 2 |
| Deep learning | PyTorch (LeRobot policies, behaviour-cloning MLP) | 2–4 |
| Experiment tracking | TensorBoard, Weights & Biases (optional) | 2–4 |
| Core libraries | NumPy, SciPy, OpenCV, PyYAML, imageio | 1–4 |
| Quality | pytest, GitHub Actions CI (MuJoCo tests + ROS 2 build and bridge smoke test) | 1–4 |
| Containers | Docker (`osrf/ros:jazzy-desktop` base) | 1–4 |

**Compute:** Phase 1 runs on a laptop CPU. In Phase 2, data recording, PPO, behaviour cloning and evaluation run on the laptop; ACT/Diffusion training runs on a free cloud GPU (Colab/Kaggle, `notebooks/train_lerobot_colab.ipynb`) or a university GPU server.

---

## Repository structure

```
kittra/
├── README.md  LICENSE  Makefile  pyproject.toml
├── requirements-sim.txt  requirements-train.txt  requirements.txt
├── .github/workflows/ci.yml           # MuJoCo tests + ROS 2 build/bridge smoke test on every push
├── docker/Dockerfile                  # Ubuntu 24.04 + ROS 2 Jazzy + MuJoCo (+ training deps)
├── assets/
│   ├── so101/so101_simplified.xml     # SO-101-like 5-DoF arm + parallel gripper (MJCF)
│   └── scenes/kitting_cell.xml        # table, tray, kit box, reject bin, parts, cameras, task sites
├── configs/
│   ├── env/                           # one YAML = one evaluation condition
│   │   ├── nominal.yaml  randomized.yaml  ood.yaml
│   │   └── ablation/abl_*.yaml        # nominal + exactly one perturbation (8 files)
│   └── train/act.yaml  diffusion.yaml  ppo_place.yaml  bc_place.yaml
├── kittra_sim/                        # simulation package, no ROS dependency
│   ├── constants.py  config.py  transforms.py
│   ├── scene.py                       # ids, task geometry from MJCF sites, part spawning
│   ├── env.py                         # KittraPickPlace-v0
│   ├── place_env.py                   # KittraPlace-v0 (state-based placement, RL)
│   ├── ik.py                          # damped-least-squares IK
│   ├── scripted_expert.py             # pick-and-place expert (demonstrations)
│   ├── place_expert.py                # placement expert (BC data, sanity baseline)
│   ├── factory.py                     # env + wrappers from a config
│   └── randomization/
│       ├── core.py                    # physics/visual randomizers, delay line, backlash, image noise
│       └── wrappers.py                # ActionLatency, Backlash, SensorNoise Gymnasium wrappers
├── kittra_learning/                   # Phase 2 learning code
│   ├── lerobot_compat.py              # single place that touches LeRobot APIs
│   ├── policies.py                    # one reset()/act() interface for every policy type
│   ├── bc.py                          # behaviour-cloning MLP
│   ├── evaluation.py  metrics.py  results.py
├── scripts/
│   ├── view_scene.py  run_expert.py  benchmark_sim.py  status.py          # Phase 1 + progress
│   ├── record_lerobot_dataset.py  train_lerobot.py  train_ppo.py          # Phase 2
│   └── train_bc_place.py  evaluate.py  make_results_table.py
├── notebooks/train_lerobot_colab.ipynb
├── ros2_ws/src/
│   ├── kittra_description/            # URDF (kinematics identical to the MJCF) + RViz config
│   ├── kittra_bridge/                 # mujoco_bridge node
│   ├── kittra_policy/                 # policy_runner node (trained policy -> /joint_commands)
│   └── kittra_bringup/                # sim.launch.py, sim_policy.launch.py
└── tests/                             # 67 tests; MuJoCo-dependent ones skip if MuJoCo is missing
```

---

## Setup

### 1. Operating system

- **Linux:** Ubuntu 24.04.
- **Windows:** WSL2 with Ubuntu 24.04 (PowerShell as administrator): `wsl --install -d Ubuntu-24.04`. WSLg opens the MuJoCo viewer and RViz2 as normal windows.
- **macOS:** MuJoCo runs natively; use the Docker image for the ROS 2 parts.

### 2. ROS 2 Jazzy

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools python3-colcon-common-extensions \
  ros-jazzy-joint-state-publisher-gui libegl1 libosmesa6
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

> If the apt repository instructions change, follow docs.ros.org (Jazzy → Installation → Ubuntu deb packages).

### 3. Python environment

ROS 2 Jazzy uses the system Python 3.12, so the venv reuses system packages (makes `rclpy` visible):

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -c constraints-ubuntu24.txt -r requirements-sim.txt && pip install -e .   # Phase 1
pip install torch --index-url https://download.pytorch.org/whl/cpu                    # CPU PyTorch (laptop)
pip install -c constraints-ubuntu24.txt "stable-baselines3>=2.3" tqdm rich tensorboard "pandas>=2.0" matplotlib
```

The constraints file keeps NumPy at 1.26: the venv also sees Ubuntu's system packages (needed for
`rclpy`), and those, like ROS 2 Jazzy, are built against NumPy 1.x. LeRobot (ACT/Diffusion) is installed
on the GPU server with `requirements-train.txt`, where no ROS constraint applies.

Headless machines (WSL2 without WSLg, servers, Colab): `export MUJOCO_GL=egl` (or `osmesa`).

> LeRobot changes quickly. All LeRobot calls go through `kittra_learning/lerobot_compat.py`, which handles the 0.3.x and 0.4.x APIs. Once your pipeline works, pin the exact version in `requirements-train.txt`.

### 4. ROS 2 workspace

Build with the venv active and run colcon through the venv's Python, so the nodes use the
interpreter that has `kittra_sim` and MuJoCo (plain `colcon` would use `/usr/bin/python3`):

```bash
source .venv/bin/activate
cd ros2_ws && python -m colcon build --symlink-install && source install/setup.bash
```

### 5. Docker (alternative)

```bash
docker build -f docker/Dockerfile -t kittra .
docker build -f docker/Dockerfile --build-arg INSTALL_TRAIN=1 -t kittra-train .
```

---

# Phase 1 – Simulated cell ✅

The goal of Phase 1 is a **complete simulated robot cell** that exposes the **same ROS 2 interface** the real robot will use in Phase 3, so every later phase can switch between simulation and hardware with one launch argument.

### Deliverables

1. **MuJoCo scene** (`assets/scenes/kitting_cell.xml`): SO-101-like 5-DoF arm with a parallel-jaw gripper, table, low-rim parts tray, kit box, reject bin, a red target cube, two distractor cylinders, overhead and wrist RGB cameras.
2. **Gymnasium environment** `KittraPickPlace-v0` with configurable randomization hooks.
3. **Damped-least-squares IK** and a **scripted expert** that completes the pick-and-place task.
4. **ROS 2 bridge** (`kittra_bridge`) publishing joint states, camera images, TF and scene markers, and accepting joint commands.
5. **URDF + RViz config + launch files**, including an `autopilot:=expert` demo mode.
6. **Tests** for configs, kinematics consistency (MJCF ↔ URDF), environment, IK and expert; **CI** on every push.

### Robot and scene model

| Item | Choice | Reason |
|---|---|---|
| Arm | 5 revolute joints named as in LeRobot's SO-101 (`shoulder_pan` … `wrist_roll`), position-controlled servos with gravity compensation | Same joint names, order and action semantics as the real robot |
| Gripper | Parallel jaw: two slide joints coupled by an equality constraint, value = finger travel per side (m) | Far more reliable grasps in simulation than the real rotating jaw |
| Link lengths | Approximations (2 × 14 cm links), not the exact SO-101 dimensions | Phase 3 swaps in the official SO-101 description and re-measures the cell |
| Physics | 480 Hz (`implicitfast`, elliptic friction cone, `noslip` iterations); 16 substeps = **30 Hz control** | Stable small-part contacts; control rate equals the LeRobot SO-101 recording rate |
| Collision groups | Links collide with table/parts but not with each other; finger pads and parts use `condim=4` | Avoids spurious self-contacts, adds torsional friction for grasping |
| Task geometry | Named MJCF sites (`tray_spawn`, `kit_area`, `kit_slot`, `reject_area`) | Python code reads geometry from the model; no duplicated numbers |
| Consistency | `tests/test_model_consistency.py` checks joint origins, axes and limits of MJCF vs. URDF | RViz shows exactly what MuJoCo simulates |

Workspace reachability was checked with an independent forward-kinematics model: every top-down grasp pose over the tray spawn area and every placement pose over the kit box is reachable within joint limits.

### Gymnasium environment – `KittraPickPlace-v0`

| Item | Definition |
|---|---|
| **Observation** | `dict`: `joint_pos` (6), `joint_vel` (6), `ee_pos` (3), `overhead_rgb` (128×128×3), `wrist_rgb` (128×128×3); `obs_mode="state"` drops the images |
| **Privileged info** | Target pose, slot position, TCP pose, placement error are returned in **`info` only**, never in the observation, so learned policies cannot cheat |
| **Action** | Absolute joint position targets (6), clipped to actuator ranges (LeRobot SO-101 action space) |
| **Reward** | Sparse `1` on success, or `shaped` (potential-based reach + place terms, +10 success, −5 drop) |
| **Success** | Cube rests inside the kit box, gripper open and away from the part |
| **Termination** | Success, cube fell off the table, or 300 steps (10 s); NaN guard ends unstable episodes cleanly |
| **Reset** | Home pose + noise; target and *n* distractors at random non-overlapping tray positions and yaws (seeded) |

```python
import gymnasium as gym
import kittra_sim

env = gym.make("KittraPickPlace-v0", config="randomized", obs_mode="rgb")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### Inverse kinematics

`DLSIK` solves `dq = Jᵀ(JJᵀ + λ²I)⁻¹e` on MuJoCo's site Jacobians with weighted position + orientation error, step limiting and joint-limit clipping, on its own `MjData` so it never disturbs the simulation. For a 5-DoF arm a top-down grasp is still fully reachable: `shoulder_pan` sets the azimuth and `wrist_roll` sets the gripper yaw.

### Scripted expert

Cartesian segments for the tool centre point, each with a minimum-jerk profile and IK at every control step (warm-started, so the elbow-up solution is kept):

| Phase | Steps | Motion |
|---|---|---|
| approach | 40 | to 7 cm above the cube, gripper yaw aligned with a cube face (closest of the 4 symmetric options to the natural wrist angle) |
| descend | 25 | to grasp height (TCP 4 mm above the cube centre, fingertips clear the tray rim) |
| close + settle | 12 + 6 | close the gripper, hold |
| lift | 25 | +10 cm |
| transport | 45 | above the kit slot, yaw re-chosen for the kit position |
| lower + release | 25 + 10 | 12 mm above the kit floor, open |
| retreat | 20 | +6 cm |

Approach and descend targets are refreshed from the latest part pose until the gripper closes, so a nudged part is still grasped. The expert uses privileged ground truth by design: it only generates demonstrations.

### ROS 2 bridge – `kittra_bridge`

| Direction | Interface | Type |
|---|---|---|
| Publishes | `/joint_states` (measured, includes sensor noise when configured) | `sensor_msgs/JointState` |
| Publishes | `/camera/overhead/image_raw`, `/camera/wrist/image_raw` (rgb8, sensor-data QoS) | `sensor_msgs/Image` |
| Publishes | `/tf`, `/tf_static` (world → base_link, world → part frames) | `tf2_msgs/TFMessage` |
| Publishes | `/kittra/scene_markers`, `/kittra/episode_info` (JSON events) | `MarkerArray`, `String` |
| Subscribes | `/joint_commands` (absolute targets by joint name) | `sensor_msgs/JointState` |
| Service | `/kittra/reset` | `std_srvs/Trigger` |
| Parameters | `env_config`, `image_size`, `publish_images`, `rate_hz`, `autopilot` (`none`/`expert`), `auto_reset`, `seed` | |

`robot_state_publisher` turns `/joint_states` into the TF tree from the URDF. The Phase 3 hardware driver will expose the identical topics, selected with `use_sim:=false`.

### Running Phase 1

```bash
python scripts/view_scene.py                                  # inspect the scene
python scripts/view_scene.py --config randomized --save-images out/
python scripts/run_expert.py --episodes 50                     # metrics -> results/phase1/expert_nominal.json
python scripts/run_expert.py --episodes 5 --render             # watch it
python scripts/run_expert.py --episodes 3 --video media/expert.mp4
python scripts/benchmark_sim.py                                # simulation speed

ros2 launch kittra_bringup sim.launch.py autopilot:=expert     # demo in RViz
ros2 launch kittra_bringup sim.launch.py                       # idle cell, drive it yourself:
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState \
  "{name: ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'], \
    position: [0.3, -1.2, 1.6, 1.2, 0.0, 0.024]}"
ros2 service call /kittra/reset std_srvs/srv/Trigger
ros2 topic hz /joint_states

pytest -m "not slow" -v && pytest -m slow -v
```

### Phase 1 results

> Produced by `make expert` and `make bench` (MuJoCo 3.17, laptop CPU under WSL2); raw data in `results/phase1/`.

| Metric | Value |
|---|---|
| Scripted expert success rate, nominal (50 episodes) | **100 %** (50/50), 95 % CI [92.9, 100] |
| Scripted expert success rate, randomized (50 episodes) | **100 %** (50/50), 95 % CI [92.9, 100] |
| Mean episode length (successful episodes) | 186 steps nominal / 187.4 steps randomized (≈ 6.2 s) |
| IK failures over 100 episodes | 0 |
| Simulation speed, state-only / with two 128 px cameras (laptop CPU) | 58.6 / 1.3 × real time |
| ROS 2 bridge `/joint_states` rate (`ros2 topic hz`, RViz and cameras running) | ≈ 29.7 Hz steady (target 30 Hz) |

Demo video: _add link_

---

# Phase 2 – Learned skills in simulation ✅

Phase 2 replaces the privileged expert with **learned policies that see only camera images and joint states**, and makes them **robust to the imperfections of real hardware** before any hardware is involved.

### Research questions

1. How do **ACT** and **Diffusion Policy** compare on vision-based pick-and-place, and how many demonstrations does each need?
2. For a precision sub-skill (placing a held part on a slot), does **RL (PPO)** beat **behaviour cloning** trained on expert demonstrations in the same environment?
3. How much do **domain randomization** and **sensor/actuator models** improve robustness on held-out perturbations, what does that cost under nominal conditions, and which single perturbation hurts most?

### Deliverables

1. Sensor and actuator models as composable Gymnasium wrappers.
2. Physics and visual domain randomization driven by YAML configs.
3. Evaluation conditions: nominal, randomized (training distribution), **held-out OOD**, and **8 single-perturbation ablations**.
4. LeRobot datasets from the scripted expert (nominal / randomized; 50 / 100 / 200 demos as exact prefixes).
5. ACT and Diffusion Policy training (`scripts/train_lerobot.py`, Colab notebook).
6. `KittraPlace-v0` with a PPO policy and a behaviour-cloning baseline.
7. One evaluation script for every policy type, fixed seeds, Wilson confidence intervals, automatic result tables.
8. `kittra_policy` ROS 2 node: the exact deployment path for Phase 3, run against the simulator.

## 2.1 Sensor and actuator modeling

Every effect is sampled **per episode** (fixed for the episode, like a real device) or **per step** (noise), from a private generator seeded by the environment, so `reset(seed=s)` reproduces it exactly.

| Effect | Model | `randomized.yaml` |
|---|---|---|
| Joint encoder noise | Gaussian per step + constant bias per episode | σ = 0.005 rad, bias ±0.01 rad |
| Image noise | Additive Gaussian, σ sampled per episode | σ ∈ [0, 8] (0–255 scale) |
| Motion blur | Horizontal or vertical box blur | p = 0.3, kernel 1–5 px |
| Exposure | Brightness / contrast factors per episode | ±20 % |
| Action latency | Commands delayed by k control steps (`DelayLine`) | k ∈ {0, 1, 2, 3} = 0–100 ms |
| Backlash | Play model: output moves only once the command leaves a ±b/2 band | b ∈ [0, 0.01] rad per arm joint |

Signal chain built by `kittra_sim.factory.make_pick_place_env`:

```
policy ──action──▶ ActionLatencyWrapper ──▶ BacklashWrapper ──▶ MuJoCo env
policy ◀──obs───── SensorNoiseWrapper ◀──────────────────────── MuJoCo env
```

```python
class ActionLatencyWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)          # inner reset first (it may re-seed)
        rng = _child_rng(self.env)
        self.delay = int(rng.integers(self.min_delay, self.max_delay + 1))
        self._line = DelayLine(self.delay, self.env.unwrapped.current_command())  # "hold still" prefill
        return obs, dict(info, action_delay=self.delay)

    def step(self, action):
        return self.env.step(self._line.push(np.asarray(action, dtype=float)))
```

## 2.2 Domain randomization

Applied in `env.reset()`. Randomizers cache the nominal model once and always write `nominal × scale`, so randomization never accumulates and a disabled section restores the nominal model exactly (both properties are unit-tested).

| Category | Parameter | Training range | OOD range |
|---|---|---|---|
| Physics | Sliding friction (parts, finger pads) | ×0.5 – ×1.5 | ×0.3 – ×2.0 |
| Physics | Part mass (inertia scaled with it) | ×0.7 – ×1.3 | ×0.5 – ×1.6 |
| Physics | Arm joint damping | ×0.8 – ×1.2 | ×0.6 – ×1.5 |
| Physics | Actuator gain kp (gain and bias kept consistent) | ×0.85 – ×1.15 | ×0.7 – ×1.3 |
| Visual | Light position / intensity | ±0.3 m / ×0.6–1.4 | ±0.45 m / ×0.4–1.7 |
| Visual | Table, container and distractor colours / target colour | ±0.2 / ±0.1 RGB | ±0.35 / ±0.2 RGB |
| Visual | Overhead camera pose | ±1.5 cm, ±3° | ±3 cm, ±6° |
| Sensor / actuator | see 2.1 | as above | noise ×1.75–2, latency 2–5 steps, backlash 0.02 rad |
| Task | Part positions and yaws in the tray | uniform | uniform |

```python
# kittra_sim/randomization/core.py (excerpt)
gain[self.kp_actuators, 0] *= k_scale     # MuJoCo position actuator: force = kp*ctrl - kp*qpos
bias[self.kp_actuators, 1] *= k_scale     # so gainprm[0] and -biasprm[1] must scale together
mass[self.mass_bodies] *= m_scale
inertia[self.mass_bodies] *= m_scale[:, None]
```

**Conditions** (`configs/env/`): `nominal`, `randomized` (training), `ood` (held out; never used for training or tuning), and `ablation/abl_{latency, backlash, camera, lighting, colors, friction, joint_noise, image_noise}` — nominal plus exactly one perturbation (enforced by a test).

## 2.3 Demonstration datasets (LeRobot format)

`scripts/record_lerobot_dataset.py` runs the expert in the configured environment and stores what a real robot would log: the **noisy** images and joint readings as observations and the expert's commands as actions. Failed episodes are discarded. Seeds are consumed in order, so the 50- and 100-demo datasets are **exact prefixes** of the 200-demo dataset, which makes the data-efficiency study clean.

| Dataset | Config | Demos |
|---|---|---|
| `kittra_nominal` | nominal | 200 |
| `kittra_randomized` | randomized | 200 |
| `kittra_randomized_50`, `kittra_randomized_100` | randomized | 50, 100 |

Features: `observation.state` (6), `observation.images.overhead` and `.wrist` (128×128 video), `action` (6), task string, 30 fps.

```bash
make data                         # all four datasets into data/lerobot/
python scripts/record_lerobot_dataset.py --config randomized --episodes 200 --push-to-hub --hf-user <you>
```

## 2.4 Imitation learning: ACT and Diffusion Policy

| Policy | Idea | Why it fits |
|---|---|---|
| **ACT** | Transformer + CVAE predicts a chunk of 50 future actions | Smooth motions, few demos needed, LeRobot default for SO-101 |
| **Diffusion Policy** | Generates action sequences by iterative denoising | Handles multi-modal demonstrations |

Hyperparameters live in `configs/train/{act,diffusion}.yaml`; `scripts/train_lerobot.py` builds the `lerobot-train` call and names runs `<policy>_<dataset>`:

```bash
python scripts/train_lerobot.py --config configs/train/act.yaml       --dataset kittra_randomized --device cuda
python scripts/train_lerobot.py --config configs/train/diffusion.yaml --dataset kittra_randomized --device cuda
make train-matrix                 # {ACT, Diffusion} x {nominal, randomized 50/100/200}
```

**On the GPU server (unattended):** `scripts/run_il_pipeline.sh` records the four datasets, trains the eight runs four at a time on one GPU, evaluates every checkpoint (plus the ablation) and builds the results table. Every stage skips finished work, so it can simply be restarted after an interruption.

```bash
python3 -m venv ~/venvs/kittra && source ~/venvs/kittra/bin/activate   # clean venv, no system packages
pip install -r requirements-train.txt && pip install -e .               # LeRobot 0.7.0 (pinned, verified)
nohup bash scripts/run_il_pipeline.sh > logs/pipeline.log 2>&1 &
```

ACT trains at about 20 steps/s (batch 8) and needs under 1 GB of GPU memory on an RTX A6000, so several runs share one GPU.

## 2.5 RL vs. behaviour cloning: the placement skill

Full pick-and-place from pixels is very sample-inefficient for RL, so RL is applied where it is strongest: **placing an already-grasped cube precisely on the kit slot**.

`KittraPlace-v0`:

| Item | Definition |
|---|---|
| **Start state** | Cube held in the gripper; TCP 6–10 cm above the kit box, up to 4 cm off the slot, random yaw |
| **Observation** (22) | joint positions (6, with sensor noise) · joint velocities (6) · slot − TCP (3) · cube quaternion (4) · slot − cube (3) |
| **Action** (6) | `a[:5] × 0.05 rad` arm joint deltas; `a[5] > 0` releases (one-shot decision) |
| **Reward** | −‖cube − slot‖ − 0.01‖a‖² per step; +10 success, −5 failed release or drop |
| **Success** | After release the cube rests in the kit box within **1 cm** of the slot centre |
| **Imperfections** | Same physics randomization, joint noise/bias, latency and backlash, applied on absolute joint targets with the same core classes |

**PPO** (Stable-Baselines3): 8 parallel environments, `VecNormalize`, 256×256 MLP, 3 M steps, best model and its normalization statistics saved on every evaluation improvement.

**Behaviour cloning baseline:** an MLP with the same observation/action spaces, trained on successful demonstrations from the scripted placement expert. Same env, same evaluation: a fair IL-vs-RL comparison.

```bash
make train-ppo          # CPU, ~3 h for 3 M steps; checkpoints every 100k steps
make resume-ppo         # continue after a shutdown or Ctrl+C
make status             # what has finished, what to run next
make train-bc
tensorboard --logdir runs/
```

## 2.6 Evaluation protocol

One script evaluates every policy type (`expert`, `act`, `diffusion`, `sb3`, `bc`, `place_expert`, `random`). Episode *i* of every condition uses seed `1000 + i` for all policies, so they face identical layouts and perturbation samples. **100 episodes per condition.**

| Metric | Definition |
|---|---|
| Success rate | with 95 % **Wilson** confidence interval |
| Episode length | mean, and mean over successful episodes |
| Drop rate | cube fell off the table / out of the gripper |
| Placement error | horizontal distance cube ↔ slot at the end (mm) |
| Inference time | mean policy time per control step (ms) |

```bash
python scripts/evaluate.py --policy-type act --policy outputs/act_randomized/checkpoints/last/pretrained_model \
  --name act_randomized_200 --train-data randomized --n-demos 200 --conditions nominal randomized ood --device cuda
python scripts/evaluate.py --policy-type sb3 --policy outputs/ppo_place/best_model.zip --name ppo_place
python scripts/evaluate.py --policy-type act --policy <ckpt> --name act_nominal_200 --conditions ablation
make eval-experts eval-place eval-il eval-ablation
make results                      # -> results/phase2_table.md + results/plots/data_efficiency.png
```

## 2.7 Deployment path: `kittra_policy` ROS 2 node

`policy_runner` subscribes to `/joint_states` and both camera topics, runs the LeRobot policy at 30 Hz and publishes `/joint_commands`. It never touches the simulator, so in Phase 3 the same node drives the real SO-101. It resets the policy (e.g. ACT's action queue) whenever the bridge announces a new episode, and can be paused via `/kittra_policy/enable`.

```bash
ros2 launch kittra_bringup sim_policy.launch.py policy_type:=act env_config:=randomized \
  policy_path:=$PWD/outputs/act_randomized/checkpoints/last/pretrained_model
```

### Phase 2 results

> Produced by `make results`; copy the tables from `results/phase2_table.md`. Success rate with 95 % CI, 100 episodes per cell.

**Imitation learning (pick-and-place, 200 demos)**

| Policy | Training data | Nominal | Randomized | Out-of-distribution |
|---|---|---|---|---|
| Scripted expert (privileged upper bound) | – | _TBD_ | _TBD_ | _TBD_ |
| ACT | Nominal | _TBD_ | _TBD_ | _TBD_ |
| ACT | Randomized | _TBD_ | _TBD_ | _TBD_ |
| Diffusion Policy | Nominal | _TBD_ | _TBD_ | _TBD_ |
| Diffusion Policy | Randomized | _TBD_ | _TBD_ | _TBD_ |

**Data efficiency (randomized data, randomized test)**

| Policy | 50 demos | 100 demos | 200 demos |
|---|---|---|---|
| ACT | _TBD_ | _TBD_ | _TBD_ |
| Diffusion Policy | _TBD_ | _TBD_ | _TBD_ |

**Placement skill** — success rate with 95 % CI, and mean final placement error; 100 fixed-seed episodes per cell (`results/phase2_ppo_training.md`)

![PPO learning curves and final evaluation](results/plots/ppo_training_summary.png)

| Policy | Nominal | Randomized | Out-of-distribution |
|---|---|---|---|
| Scripted placement expert (privileged, upper bound) | 100 % [96, 100] · 0.5 mm | 100 % [96, 100] · 0.8 mm | 99 % [95, 100] · 1.4 mm |
| PPO seed 0 (checkpoint at 1.38 M steps) | 87 % [79, 92] · 4.8 mm | 78 % [69, 85] · 5.1 mm | 49 % [39, 59] · 5.9 mm |
| PPO seed 1 (checkpoint at 2.38 M steps) | 97 % [92, 99] · 5.1 mm | 90 % [83, 94] · 4.8 mm | 57 % [47, 66] · 5.4 mm |
| **PPO, mean of 2 seeds** | **92 %** | **84 %** | **53 %** |
| Behaviour cloning (MLP, 300 randomized expert demos) | 87 % [79, 92] · 3.4 mm | 57 % [47, 66] · 4.2 mm | 14 % [9, 22] · 4.6 mm |

**Training cost and growth** (seed 1, complete 3 M-step run; 8 parallel environments on a laptop CPU)

| | Value |
|---|---|
| Environment steps | 3,000,000 |
| Training episodes (one episode = one placement attempt, ending at release) | 657,289 (≈ 4.6 steps each) |
| Wall-clock time | 5.4 h |
| 50 % / 80 % / 90 % success first reached | 525 k steps (194 k episodes) / 950 k (294 k) / 2.08 M (508 k) |
| Best in-training evaluation | 100 % (50 episodes) at 2.38 M steps |

Seed 0 was stopped at 1.875 M steps by a shutdown; it reached 50 % at 500 k and 80 % at 1.23 M steps and never sustained 90 %.

**What this shows**
- **Equal under nominal conditions, very different under perturbation.** BC matches PPO seed 0 at 87 % nominal but falls to 57 % on the training distribution and 14 % out of distribution; PPO averages 84 % and 53 %. BC only saw the expert's clean trajectories, so a small error from latency, backlash or noise puts it in states it never saw and errors compound (covariate shift). PPO explored those states during training.
- **Seed variance is real:** 87 % vs 97 % nominal between two PPO seeds with identical settings, which is why results are reported per seed and averaged.
- **Most of the learning happens early:** 80 % success after about 1 M steps; the last 2 M steps add about 10 points.
- **BC places more precisely when it succeeds** (3.4 vs ~5 mm): it imitates the expert's careful descent, while PPO learned to release from a few centimetres above the slot, which the reward allows. A release-height penalty is the obvious next reward change.
- **Out-of-distribution remains the weak point** (53 % for PPO): the held-out ranges are wider than anything trained on, which is exactly what Phase 3's measured (instead of guessed) randomization ranges should address.

**Ablation: which perturbation hurts most?** (ACT trained on nominal data)

| Perturbation | Success rate |
|---|---|
| None | _TBD_ |
| Action latency (100 ms) | _TBD_ |
| Backlash (up to 0.02 rad) | _TBD_ |
| Camera pose offset (±2 cm, ±4°) | _TBD_ |
| Lighting change | _TBD_ |
| Colour change | _TBD_ |
| Low friction (×0.4–0.6) | _TBD_ |
| Joint encoder noise + bias | _TBD_ |
| Image noise, blur, exposure | _TBD_ |

Training curves: _add W&B / TensorBoard link_ · Demo videos: _add link_

---

## Design decisions and known limitations

- **Simplified arm model.** Joint names, order and action semantics match the SO-101, but link lengths are approximate and the gripper is a parallel jaw. Policies trained here will not transfer zero-shot to the real arm without the Phase 3 model swap; the value of Phases 1–2 is the pipeline, the robustness methodology and the measured trends.
- **Privileged expert.** The scripted expert uses ground-truth poses; its demonstrations are cleaner and less varied than human teleoperation, so Phase 2 success rates are likely optimistic compared with Phase 3.
- **Noise models are assumptions.** The ranges in `randomized.yaml` are educated guesses for hobby-grade servos and USB cameras. Phase 3 replaces them with values measured on the real SO-101 (system identification).
- **Contact fidelity.** MuJoCo contact dynamics differ from reality, especially for small, light parts; the real sim-to-real gap is only measured in Phase 3.
- **Over-randomization trade-off.** Very wide ranges can lower nominal performance; the ablation table shows which perturbations matter before the ranges are widened.
- **Asynchronous ROS 2 loop.** In `sim_policy.launch.py` the bridge and the policy node run on independent 30 Hz timers, as they will on hardware; offline evaluation in `scripts/evaluate.py` is synchronous and is the number reported in the tables.
- **Headless rendering.** If the MuJoCo viewer or offscreen rendering fails (e.g. WSL2 without WSLg), set `MUJOCO_GL=egl` or `osmesa`.

---

# Phase 3 – Real hardware 🗓️ Planned

**Goal:** run the learned policies on a real robot and measure the sim-to-real gap.

**Planned hardware**

| Item | Purpose |
|---|---|
| 2× SO-101 arms (leader + follower) | Teleoperation and execution |
| USB RGB camera (overhead) + wrist camera | Same viewpoints as simulation |
| Optional: Intel RealSense / OAK-D depth camera | Depth for Phase 4 perception |
| 3D-printed parts, tray, kit box, reject bin | Physical replica of the simulated cell |
| Optional: NVIDIA Jetson Orin Nano | On-robot inference |

**Planned work**
- Replace the simplified MJCF/URDF with the official SO-101 description; re-run the consistency and reachability checks.
- Build and calibrate the arms; camera intrinsic and hand-eye calibration (OpenCV).
- `ros2_control` hardware interface for the Feetech servos exposing the same topics as `kittra_bridge` (`use_sim:=false`).
- Record 50–100 teleoperated demonstrations per skill with LeRobot (same dataset format as Phase 2).
- **System identification:** measure servo latency, backlash and camera noise; replace the guessed ranges in `randomized.yaml`.
- Evaluate sim-only (zero-shot), real-only and sim + real co-trained policies.
- Fine-tune a **vision-language-action model** (SmolVLA; already supported by `policies.py`) and compare it with ACT.
- Safety: joint limits, workspace bounds, software e-stop.

**Planned key result:** success rate of zero-shot sim-to-real vs. real-data vs. co-trained policies, and the effect of measured vs. guessed randomization ranges.

---

# Phase 4 – Language agent + visual inspection 🗓️ Planned

**Goal:** let an operator give orders in natural language and have the robot detect and reject defective parts.

**Planned components**
- **Open-vocabulary perception:** Grounding DINO + SAM 2 to locate parts by name ("M4 screw", "bracket").
- **Defect classifier:** a CNN trained on images of good vs. defective 3D-printed parts (bent, cracked, missing features).
- **VLM agent node:** a vision-language model with tool calling that plans the kit, calls ROS 2 skills, verifies results from the camera and replans on failure.
- **Order interface:** text (and optionally voice) input; kit report and order log as output.

**Planned skills as ROS 2 actions**

| Action | Description |
|---|---|
| `scan_tray()` | Lists detected parts with poses |
| `inspect(part_id)` | Returns part class + defect probability |
| `pick(part_id)` | Runs the learned grasp policy on a target |
| `place(location)` | Kit box slot or reject bin (PPO / ACT placement policy) |

**Planned key results:** order completion rate, defect detection precision/recall, agent recovery rate after failed grasps, end-to-end time per kit.

---

## Author

**Keyur Chaudhari** – M.Sc. Applied Data Science & AI, SRH University Heidelberg
GitHub: [keyurck7](https://github.com/keyurck7) · LinkedIn: [keyur-chaudhari-ck7](https://www.linkedin.com/in/keyur-chaudhari-ck7)

## License

MIT

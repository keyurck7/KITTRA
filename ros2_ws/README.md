# Kittra ROS 2 workspace (Jazzy)

| Package | Type | Purpose |
|---|---|---|
| `kittra_description` | ament_cmake | URDF of the simulated arm + RViz config |
| `kittra_bridge` | ament_python | MuJoCo ↔ ROS 2 bridge (`mujoco_bridge` node) |
| `kittra_policy` | ament_python | Runs a trained ACT/Diffusion policy (`policy_runner` node) |
| `kittra_bringup` | ament_cmake | `sim.launch.py`, `sim_policy.launch.py` |

Build **with the project virtual environment active, and run colcon through the venv's
Python** (`python -m colcon`). Otherwise colcon uses `/usr/bin/python3`, the node scripts get that
interpreter, and they fail with `ModuleNotFoundError: No module named 'kittra_sim'`:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/kittra && source .venv/bin/activate
cd ros2_ws
rm -rf build install log            # only needed if you built with plain `colcon` before
python -m colcon build --symlink-install
source install/setup.bash
head -1 install/kittra_bridge/lib/kittra_bridge/mujoco_bridge   # must point into .venv
```

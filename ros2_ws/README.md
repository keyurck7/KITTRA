# Kittra ROS 2 workspace (Jazzy)

| Package | Type | Purpose |
|---|---|---|
| `kittra_description` | ament_cmake | URDF of the simulated arm + RViz config |
| `kittra_bridge` | ament_python | MuJoCo ↔ ROS 2 bridge (`mujoco_bridge` node) |
| `kittra_policy` | ament_python | Runs a trained ACT/Diffusion policy (`policy_runner` node) |
| `kittra_bringup` | ament_cmake | `sim.launch.py`, `sim_policy.launch.py` |

Build **with the project virtual environment active**, so the nodes use the Python that has
`kittra_sim`, MuJoCo and (for the policy node) PyTorch/LeRobot:

```bash
source /opt/ros/jazzy/setup.bash
source ../.venv/bin/activate
colcon build --symlink-install
source install/setup.bash
```

If `ros2 run` picks up the wrong Python, run `python -m colcon build ...` from inside the venv,
or set `export PYTHONPATH=$PYTHONPATH:$(python -c "import site; print(site.getsitepackages()[0])")`.

"""Kittra simulation package: MuJoCo kitting cell, Gymnasium environments, scripted expert and
domain randomization. Has no ROS dependency, so it can be installed on Colab/Kaggle for training.
"""
from kittra_sim.constants import JOINT_NAMES, SCENE_PATH  # noqa: F401

__version__ = "0.2.0"

try:  # Gymnasium is optional for the pure-NumPy parts (configs, randomization core, transforms)
    from gymnasium.envs.registration import register, registry

    if "KittraPickPlace-v0" not in registry:
        register(id="KittraPickPlace-v0", entry_point="kittra_sim.factory:make_pick_place_env")
    if "KittraPlace-v0" not in registry:
        register(id="KittraPlace-v0", entry_point="kittra_sim.factory:make_place_env")
except ImportError:  # pragma: no cover
    pass

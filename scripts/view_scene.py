#!/usr/bin/env python3
"""Open the kitting cell in the interactive MuJoCo viewer.

    python scripts/view_scene.py                       # static scene at the home pose
    python scripts/view_scene.py --config randomized   # one randomized episode reset
    python scripts/view_scene.py --save-images out/    # also save overhead + wrist camera images
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

from kittra_sim.env import KittraPickPlaceEnv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-images", type=Path, default=None, help="directory for camera snapshots")
    args = ap.parse_args()

    env = KittraPickPlaceEnv(args.config, obs_mode="rgb" if args.save_images else "state")
    obs, info = env.reset(seed=args.seed)
    print(f"target part at {info['target_pos'].round(3)}, kit slot at {info['slot_pos'].round(3)}")

    if args.save_images:
        import cv2

        args.save_images.mkdir(parents=True, exist_ok=True)
        for key in ("overhead_rgb", "wrist_rgb"):
            path = args.save_images / f"{key}_seed{args.seed}.png"
            cv2.imwrite(str(path), cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR))
            print(f"saved {path}")

    # Passive viewer: the scene stays at the reset state while you inspect it.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()
    env.close()


if __name__ == "__main__":
    main()

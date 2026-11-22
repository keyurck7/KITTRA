#!/usr/bin/env python3
"""Run the scripted expert and report Phase 1 metrics (success rate, episode length, IK failures).

    python scripts/run_expert.py --episodes 50                       # headless, nominal
    python scripts/run_expert.py --episodes 5 --render               # watch it in the viewer
    python scripts/run_expert.py --episodes 50 --config randomized   # expert under perturbations
    python scripts/run_expert.py --episodes 3 --video out/expert.mp4 # save a demo video

Headless runs write results/phase1/expert_<config>.json; --render/--video demo runs do not.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from kittra_learning.metrics import wilson_interval
from kittra_sim.factory import make_pick_place_env
from kittra_sim.scripted_expert import ScriptedPickPlaceExpert


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--config", default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--video", type=Path, default=None, help="save an MP4 of all episodes (overhead camera)")
    ap.add_argument("--out", type=Path, default=Path("results/phase1"))
    ap.add_argument("--no-save", action="store_true", help="do not write the results JSON")
    args = ap.parse_args()

    render_mode = "human" if args.render else ("rgb_array" if args.video else None)
    env = make_pick_place_env(args.config, obs_mode="state", render_mode=render_mode)
    expert = ScriptedPickPlaceExpert(env)
    frames: list[np.ndarray] = []
    episodes = []
    t0 = time.perf_counter()
    sim_steps = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        expert.reset(obs, info)
        done, steps = False, 0
        while not done:
            obs, _, terminated, truncated, info = env.step(expert.act(obs, info))
            steps += 1
            done = terminated or truncated
            if args.video:
                frames.append(env.render())
            if args.render:
                time.sleep(1.0 / 30.0)  # play back in real time
        sim_steps += steps
        episodes.append({
            "seed": args.seed + ep,
            "success": bool(info["is_success"]),
            "length": steps,
            "dropped": bool(info["part_dropped"]),
            "placement_error_mm": 1000 * info["placement_error"],
            "ik_failures": expert.ik_failures,
            "unstable": bool(info.get("sim_unstable", False)),
        })
        print(f"episode {ep:3d}  success={episodes[-1]['success']!s:5}  steps={steps:3d}  "
              f"ik_failures={expert.ik_failures}")

    wall = time.perf_counter() - t0
    n_succ = sum(e["success"] for e in episodes)
    lo, hi = wilson_interval(n_succ, len(episodes))
    succ_lengths = [e["length"] for e in episodes if e["success"]]
    summary = {
        "config": args.config,
        "episodes": len(episodes),
        "success_rate": n_succ / len(episodes),
        "success_ci95": [lo, hi],
        "mean_length_successful": float(np.mean(succ_lengths)) if succ_lengths else None,
        "drop_rate": float(np.mean([e["dropped"] for e in episodes])),
        "total_ik_failures": int(sum(e["ik_failures"] for e in episodes)),
        "sim_speed_x_realtime": (sim_steps / 30.0) / wall if not args.render else None,
    }
    print(json.dumps(summary, indent=2))

    if args.render or args.video or args.no_save:
        print("demo run: results not saved (benchmark files are written by headless runs only)")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / f"expert_{Path(str(args.config)).stem}.json"
        out_file.write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2))
        print(f"wrote {out_file}")

    if args.video and frames:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=30, macro_block_size=1)
        print(f"wrote {args.video}")
    env.close()


if __name__ == "__main__":
    main()

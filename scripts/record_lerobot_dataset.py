#!/usr/bin/env python3
"""Record scripted-expert demonstrations as a LeRobot dataset.

Only successful episodes are kept. Seeds are consumed in order, so recording N episodes with the
same --seed always yields the same first N episodes: the 50- and 100-demo datasets used in the
data-efficiency study are exact prefixes of the 200-demo dataset.

    python scripts/record_lerobot_dataset.py --config nominal    --episodes 200
    python scripts/record_lerobot_dataset.py --config randomized --episodes 200
    python scripts/record_lerobot_dataset.py --config randomized --episodes 50  --variant randomized_50
    python scripts/record_lerobot_dataset.py --config randomized --episodes 200 --push-to-hub --hf-user <you>

Images come from the (possibly noisy) observation, states are the (possibly noisy) measured
joint positions and actions are the expert's commands, i.e. exactly what a real robot would log.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from kittra_learning import lerobot_compat as lc
from kittra_sim.constants import CONTROL_HZ, JOINT_NAMES, TASK_DESCRIPTION
from kittra_sim.factory import make_pick_place_env
from kittra_sim.scripted_expert import ScriptedPickPlaceExpert


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="randomized", help="environment config name or path")
    ap.add_argument("--episodes", type=int, default=200, help="number of SUCCESSFUL episodes to keep")
    ap.add_argument("--variant", default=None, help="dataset name suffix (default: config name, plus _N if N != 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--root", type=Path, default=Path("data/lerobot"))
    ap.add_argument("--hf-user", default="local", help="Hugging Face user/org for the repo id")
    ap.add_argument("--no-video", action="store_true", help="store PNG frames instead of MP4 video")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-attempts-factor", type=float, default=3.0,
                    help="give up after episodes * factor attempts (protects against a broken expert)")
    args = ap.parse_args()

    cfg_name = Path(str(args.config)).stem
    variant = args.variant or (cfg_name if args.episodes == 200 else f"{cfg_name}_{args.episodes}")
    repo_id = f"{args.hf_user}/kittra_{variant}"
    root = args.root / f"kittra_{variant}"
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; use --overwrite to replace it")
        shutil.rmtree(root)

    env = make_pick_place_env(args.config, obs_mode="rgb", image_size=args.image_size)
    expert = ScriptedPickPlaceExpert(env)
    use_videos = not args.no_video
    dataset = lc.create_dataset(repo_id, root, CONTROL_HZ, lc.build_features(JOINT_NAMES, args.image_size, use_videos),
                                use_videos=use_videos)

    kept, attempts, seed = 0, 0, args.seed
    max_attempts = int(args.episodes * args.max_attempts_factor)
    kept_seeds, t0 = [], time.perf_counter()
    while kept < args.episodes:
        if attempts >= max_attempts:
            raise SystemExit(f"only {kept}/{args.episodes} successful episodes after {attempts} attempts; "
                             "check the expert with scripts/run_expert.py --config " + str(args.config))
        obs, info = env.reset(seed=seed)
        expert.reset(obs, info)
        frames, done = [], False
        while not done:
            action = expert.act(obs, info)
            frames.append({
                "observation.state": obs["joint_pos"].astype("float32"),
                "observation.images.overhead": obs["overhead_rgb"],
                "observation.images.wrist": obs["wrist_rgb"],
                "action": action.astype("float32"),
            })
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        attempts += 1
        if info["is_success"]:
            for f in frames:
                lc.add_frame(dataset, f, TASK_DESCRIPTION)
            dataset.save_episode()
            kept += 1
            kept_seeds.append(seed)
            print(f"kept {kept:4d}/{args.episodes}  seed={seed}  frames={len(frames)}")
        else:
            clear = getattr(dataset, "clear_episode_buffer", None)
            if callable(clear):  # nothing was added, but keep the writer state clean
                clear()
            print(f"discarded seed={seed} (expert failed)")
        seed += 1

    lc.finish_dataset(dataset)
    meta = {
        "repo_id": repo_id,
        "root": str(root),
        "env_config": str(args.config),
        "episodes": kept,
        "attempts": attempts,
        "expert_success_rate": kept / attempts,
        "seeds": kept_seeds,
        "lerobot_version": lc.lerobot_version(),
        "minutes": (time.perf_counter() - t0) / 60,
    }
    (root / "kittra_recording.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "seeds"}, indent=2))

    if args.push_to_hub:
        dataset.push_to_hub()
        print(f"pushed {repo_id} to the Hugging Face Hub")
    env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Behaviour-cloning baseline for the placement skill: record scripted-expert demos in
KittraPlace-v0 and train an MLP on them.

    python scripts/train_bc_place.py --config configs/train/bc_place.yaml
    python scripts/train_bc_place.py --config configs/train/bc_place.yaml --demos 50    # data-efficiency point

Outputs: outputs/bc_place/bc_place.pt (+ demos.npz with the recorded data).
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from kittra_sim.factory import make_place_env
from kittra_sim.place_expert import ScriptedPlaceExpert

# Print the Python stack if a native library (MuJoCo, PyTorch, OpenCV) crashes.
faulthandler.enable()


def record(env_config: str, n_episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    env = make_place_env(env_config)
    expert = ScriptedPlaceExpert(env)
    obs_buf, act_buf, n_succ = [], [], 0
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        expert.reset(obs, info)
        ep_obs, ep_act, done = [], [], False
        while not done:
            a = expert.act(obs, info)
            ep_obs.append(obs)
            ep_act.append(a)
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        if info["is_success"]:  # keep only successful demonstrations
            obs_buf += ep_obs
            act_buf += ep_act
            n_succ += 1
        if (i + 1) % 25 == 0 or i + 1 == n_episodes:
            print(f"recorded {i + 1}/{n_episodes} episodes, {n_succ} successful", flush=True)
    env.close()
    print(f"expert success: {n_succ}/{n_episodes}; {len(obs_buf)} transitions kept")
    return np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32), n_succ / n_episodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/train/bc_place.yaml"))
    ap.add_argument("--demos", type=int, default=None, help="override demo_episodes")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--record-only", action="store_true", help="record demos.npz and stop (diagnostics)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    n_demos = args.demos or cfg["demo_episodes"]
    out = args.out or Path("outputs") / ("bc_place" if args.demos is None else f"bc_place_{n_demos}")
    out.mkdir(parents=True, exist_ok=True)

    demos = out / "demos.npz"
    if demos.exists() and not args.record_only:
        d = np.load(demos)
        obs, act = d["obs"], d["act"]
        expert_rate = float(json.loads((out / "train_info.json").read_text())["expert_success_rate"]) \
            if (out / "train_info.json").exists() else float("nan")
        print(f"reusing {demos} ({len(obs)} transitions); delete it to record again", flush=True)
    else:
        obs, act, expert_rate = record(cfg["env_config"], n_demos, cfg["seed"])
        np.savez_compressed(demos, obs=obs, act=act)
        (out / "train_info.json").write_text(json.dumps({"expert_success_rate": expert_rate}, indent=2))
    if args.record_only:
        print(f"saved {demos}")
        sys.exit(0)

    # PyTorch is imported only after the simulation work is done.
    import torch

    from kittra_learning.bc import BCConfig, train_bc

    print(f"training on {len(obs)} transitions ({args.device})", flush=True)
    bc_cfg = BCConfig(**{**cfg["bc"], "hidden": tuple(cfg["bc"]["hidden"]), "seed": cfg["seed"]})
    ckpt = train_bc(obs, act, bc_cfg, device=args.device)
    ckpt["expert_success_rate"] = expert_rate
    ckpt["n_demo_episodes"] = n_demos
    torch.save(ckpt, out / "bc_place.pt")
    (out / "train_info.json").write_text(json.dumps(
        {"expert_success_rate": expert_rate, "transitions": int(len(obs)),
         "best_val_loss": ckpt["best_val_loss"], "demo_episodes": n_demos}, indent=2))
    print(f"saved {out / 'bc_place.pt'}")


if __name__ == "__main__":
    main()

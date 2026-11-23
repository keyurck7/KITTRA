#!/usr/bin/env python3
"""Train ACT or Diffusion Policy on a recorded Kittra dataset with LeRobot.

Wraps the `lerobot-train` CLI so hyperparameters live in configs/train/*.yaml and runs are named
consistently (<policy>_<dataset variant>).

    python scripts/train_lerobot.py --config configs/train/act.yaml --dataset kittra_randomized
    python scripts/train_lerobot.py --config configs/train/diffusion.yaml --dataset kittra_nominal --device cuda
    python scripts/train_lerobot.py --config configs/train/act.yaml --dataset kittra_randomized_50 --dry-run

The dataset is read from data/lerobot/<dataset> (or from the Hub with --hf-user <you>).
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def has_flag(base_cmd: list[str], name: str) -> bool | None:
    """Whether `lerobot-train --help` mentions --<name> (None if help could not be read)."""
    try:
        out = subprocess.run(base_cmd + ["--help"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout + out.stderr
    return f"--{name}" in text if text.strip() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", required=True, help="dataset folder name, e.g. kittra_randomized")
    ap.add_argument("--data-root", type=Path, default=Path("data/lerobot"))
    ap.add_argument("--hf-user", default=None, help="load the dataset from the Hub as <hf-user>/<dataset>")
    ap.add_argument("--output-root", type=Path, default=Path("outputs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override the config's step count")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ptype = cfg["policy_type"]
    run_name = f"{ptype}_{args.dataset.removeprefix('kittra_')}"
    out_dir = args.output_root / run_name

    exe = shutil.which("lerobot-train")
    cmd = [exe] if exe else [sys.executable, "-m", "lerobot.scripts.train"]
    if args.hf_user:
        cmd += [f"--dataset.repo_id={args.hf_user}/{args.dataset}"]
    else:
        cmd += [f"--dataset.repo_id=local/{args.dataset}", f"--dataset.root={args.data_root / args.dataset}"]
    cmd += [
        f"--policy.type={ptype}",
        f"--policy.device={args.device}",
        f"--output_dir={out_dir}",
        f"--job_name={run_name}",
        f"--steps={args.steps or cfg['steps']}",
        f"--batch_size={cfg['batch_size']}",
        f"--save_freq={cfg['save_freq']}",
        f"--log_freq={cfg['log_freq']}",
        f"--num_workers={cfg['num_workers']}",
        f"--seed={cfg['seed']}",
    ]
    cmd += list(cfg.get("extra_args", []))
    # Experiment tracking: LeRobot <= 0.4 uses Weights & Biases (--wandb.enable), newer versions trackio.
    base = cmd[:1] if exe else cmd[:3]
    tracker = "wandb.enable" if has_flag(base, "wandb.enable") is not False else "trackio.enable"
    cmd.append(f"--{tracker}={'true' if args.wandb else 'false'}")
    print(" ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists (LeRobot refuses to overwrite); delete it or resume manually")
    subprocess.run(cmd, check=True)
    print(f"\ncheckpoint: {out_dir}/checkpoints/last/pretrained_model")


if __name__ == "__main__":
    main()

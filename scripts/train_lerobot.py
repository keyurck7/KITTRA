#!/usr/bin/env python3
"""Train ACT or Diffusion Policy on a recorded Kittra dataset with LeRobot.

Wraps the `lerobot-train` CLI so hyperparameters live in configs/train/*.yaml and runs are named
consistently (<policy>_<dataset variant>).

    python scripts/train_lerobot.py --config configs/train/act.yaml --dataset kittra_randomized
    python scripts/train_lerobot.py --config configs/train/diffusion.yaml --dataset kittra_nominal --device cuda
    python scripts/train_lerobot.py --config configs/train/act.yaml --dataset kittra_randomized_50 --dry-run

The dataset is read from data/lerobot/<dataset> (or from the Hub with --hf-user <you>).

Interruption-safe: if outputs/<run> already holds a checkpoint, the run is resumed from it
(`lerobot-train --config_path=... --resume=true`); a finished run is marked with outputs/<run>/DONE and skipped.
While training, older checkpoints are pruned (only `last` and the one being written are kept), and when the run
finishes the optimizer state is dropped (--keep-training-state keeps it), so a run needs ~1 checkpoint of disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
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


def step_dirs(ckdir: Path) -> list[Path]:
    if not ckdir.is_dir():
        return []
    return sorted((d for d in ckdir.iterdir() if d.name.isdigit() and d.is_dir() and not d.is_symlink()),
                  key=lambda d: int(d.name))


def last_step(ckdir: Path) -> str | None:
    last = ckdir / "last"
    return Path(os.readlink(last)).name if last.is_symlink() else None


def prune_checkpoints(ckdir: Path) -> None:
    """Delete every step checkpoint except the one `last` points to and the newest (possibly being written)."""
    dirs = step_dirs(ckdir)
    keep = {dirs[-1].name} if dirs else set()
    if (target := last_step(ckdir)) is not None:
        keep.add(target)
    for d in dirs:
        if d.name not in keep:
            shutil.rmtree(d, ignore_errors=True)


def drop_partial_checkpoints(ckdir: Path) -> None:
    """Before resuming: remove checkpoints newer than `last` (written when the server stopped mid-save)."""
    target = last_step(ckdir)
    if target is None:
        return
    for d in step_dirs(ckdir):
        if int(d.name) > int(target):
            print(f"removing incomplete checkpoint {d}")
            shutil.rmtree(d, ignore_errors=True)


def run_with_pruning(cmd: list[str], ckdir: Path, every_s: float = float(os.environ.get("KITTRA_PRUNE_S", 30))) -> None:
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_s):
            prune_checkpoints(ckdir)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        stop.set()
        t.join()


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
    ap.add_argument("--restart", action="store_true", help="delete an existing run and start from step 0")
    ap.add_argument("--keep-training-state", action="store_true",
                    help="keep the optimizer state of the final checkpoint (needed only to train further)")
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
    ckdir = out_dir / "checkpoints"
    done = out_dir / "DONE"
    resume_cfg = ckdir / "last" / "pretrained_model" / "train_config.json"
    if args.restart and out_dir.exists() and not args.dry_run:
        shutil.rmtree(out_dir)
    if done.exists():
        print(f"{run_name} already finished ({done}); use --restart to train again")
        return
    if resume_cfg.exists():
        cmd = cmd[:1] if exe else cmd[:3]
        cmd += [f"--config_path={resume_cfg}", "--resume=true"]
        print(f"resuming {run_name} from checkpoint {last_step(ckdir)}")
    elif out_dir.exists() and not args.dry_run:
        print(f"{out_dir} has no checkpoint yet, starting over")
        shutil.rmtree(out_dir)
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return
    if resume_cfg.exists():
        drop_partial_checkpoints(ckdir)
    t0 = time.time()
    run_with_pruning(cmd, ckdir)
    prune_checkpoints(ckdir)
    final = ckdir / "last"
    if not args.keep_training_state:
        shutil.rmtree(final / "training_state", ignore_errors=True)
    done.write_text(json.dumps({"run": run_name, "checkpoint": last_step(ckdir),
                                "wall_time_s_last_session": round(time.time() - t0)}, indent=2) + "\n")
    print(f"\ncheckpoint: {final}/pretrained_model")


if __name__ == "__main__":
    main()

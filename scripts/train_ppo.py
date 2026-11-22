#!/usr/bin/env python3
"""Train the PPO placement skill on KittraPlace-v0 (Stable-Baselines3, CPU is fine).

    python scripts/train_ppo.py                         # full run (configs/train/ppo_place.yaml)
    python scripts/train_ppo.py --timesteps 200000      # quick check
    python scripts/train_ppo.py --resume                # continue an interrupted run
    tensorboard --logdir runs/

Outputs (outputs/ppo_place/):
    ppo_place.zip, ppo_place_vecnormalize.pkl          final model + observation normalization
    best_model.zip, best_model_vecnormalize.pkl        best model on the evaluation env
    checkpoints/ppo_place_<N>_steps.zip (+ .pkl)       every `checkpoint_freq` steps, for --resume
    eval/evaluations.npz                               evaluation history (kept across resumes)

The run is safe to interrupt: Ctrl+C saves a checkpoint, and a shutdown loses at most
`checkpoint_freq` steps. `--resume` continues from the newest checkpoint (or, for runs started
before checkpoints existed, from best_model.zip) up to the configured total.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import kittra_sim  # noqa: F401  (registers KittraPlace-v0)
from kittra_sim.factory import make_place_env

TB_DIR = "runs/ppo_place"


class SaveVecNormalizeOnBest(BaseCallback):
    """Saves the observation statistics together with every new best model."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(self.path))
        return True


def find_resume_point(out: Path) -> tuple[Path, Path, str]:
    """Newest checkpoint, else best_model (runs started before checkpointing existed)."""
    ckpts = []
    for z in (out / "checkpoints").glob("ppo_place_*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", z.name)
        if m:
            stats = z.with_name(f"ppo_place_vecnormalize_{m.group(1)}_steps.pkl")
            if stats.exists():
                ckpts.append((int(m.group(1)), z, stats))
    if ckpts:
        steps, z, stats = max(ckpts)
        return z, stats, f"checkpoint at {steps:,} steps"
    best, best_stats = out / "best_model.zip", out / "best_model_vecnormalize.pkl"
    if best.exists() and best_stats.exists():
        return best, best_stats, "best_model.zip (no checkpoints found; steps after it are re-trained)"
    raise SystemExit(f"nothing to resume in {out}: no checkpoints and no best_model.zip + best_model_vecnormalize.pkl")


def restore_eval_history(eval_cb: EvalCallback, npz_path: Path, up_to_step: int) -> None:
    """Keep the evaluation history (and the best score) of the interrupted run."""
    if not npz_path.exists():
        return
    d = np.load(npz_path)
    keep = d["timesteps"] <= up_to_step
    eval_cb.evaluations_timesteps = d["timesteps"][keep].tolist()
    eval_cb.evaluations_results = d["results"][keep].tolist()
    eval_cb.evaluations_length = d["ep_lengths"][keep].tolist()
    if "successes" in d.files:
        eval_cb.evaluations_successes = d["successes"][keep].tolist()
    if keep.any():
        eval_cb.best_mean_reward = float(np.max(np.mean(d["results"][keep], axis=1)))
        print(f"restored {int(keep.sum())} evaluations; best mean reward so far {eval_cb.best_mean_reward:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/train/ppo_place.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ppo_place"))
    ap.add_argument("--timesteps", type=int, default=None, help="total timesteps (overrides the config)")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    n_envs = args.n_envs or cfg["n_envs"]
    total = args.timesteps or cfg["total_timesteps"]
    ckpt_freq = int(cfg.get("checkpoint_freq", 100_000))
    args.out.mkdir(parents=True, exist_ok=True)

    raw_venv = make_vec_env(make_place_env, n_envs=n_envs, seed=cfg["seed"],
                            vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
                            env_kwargs={"config": cfg["env_config"]})
    eval_env = make_vec_env(make_place_env, n_envs=1, seed=cfg["seed"] + 10_000,
                            env_kwargs={"config": cfg["eval_config"]})
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    if args.resume:
        model_path, stats_path, what = find_resume_point(args.out)
        venv = VecNormalize.load(str(stats_path), raw_venv)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(str(model_path), env=venv, device=args.device, tensorboard_log=TB_DIR)
        print(f"resuming from {what}: {model.num_timesteps:,} / {total:,} steps done")
    else:
        venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg["ppo"]["gamma"])
        model = PPO("MlpPolicy", venv, seed=cfg["seed"], device=args.device, verbose=1,
                    tensorboard_log=TB_DIR, policy_kwargs=cfg["policy_kwargs"], **cfg["ppo"])

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(cfg["eval"]["freq"] // n_envs, 1),
        n_eval_episodes=cfg["eval"]["n_episodes"],
        best_model_save_path=str(args.out),
        log_path=str(args.out / "eval"),
        deterministic=True,
        callback_on_new_best=SaveVecNormalizeOnBest(args.out / "best_model_vecnormalize.pkl"),
    )
    if args.resume:
        restore_eval_history(eval_cb, args.out / "eval" / "evaluations.npz", model.num_timesteps)
    ckpt_cb = CheckpointCallback(save_freq=max(ckpt_freq // n_envs, 1), save_path=str(args.out / "checkpoints"),
                                 name_prefix="ppo_place", save_vecnormalize=True)

    remaining = total - model.num_timesteps
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=CallbackList([eval_cb, ckpt_cb]),
                        reset_num_timesteps=not args.resume, progress_bar=False)
        else:
            print("target timesteps already reached; saving the final model only")
    except KeyboardInterrupt:
        steps = model.num_timesteps
        model.save(str(args.out / "checkpoints" / f"ppo_place_{steps}_steps"))
        venv.save(str(args.out / "checkpoints" / f"ppo_place_vecnormalize_{steps}_steps.pkl"))
        print(f"\ninterrupted at {steps:,} steps; checkpoint saved. Continue with: python scripts/train_ppo.py --resume")
        venv.close()
        eval_env.close()
        return

    model.save(str(args.out / "ppo_place"))
    venv.save(str(args.out / "ppo_place_vecnormalize.pkl"))
    print(f"finished at {model.num_timesteps:,} steps; saved {args.out / 'ppo_place.zip'} and {args.out / 'best_model.zip'}")
    venv.close()
    eval_env.close()


if __name__ == "__main__":
    main()

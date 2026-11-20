#!/usr/bin/env python3
"""Train the PPO placement skill on KittraPlace-v0 (Stable-Baselines3, CPU is fine).

    python scripts/train_ppo.py --config configs/train/ppo_place.yaml
    python scripts/train_ppo.py --config configs/train/ppo_place.yaml --timesteps 200000   # quick check
    tensorboard --logdir runs/

Outputs (outputs/ppo_place/):
    ppo_place.zip, ppo_place_vecnormalize.pkl         final model + observation normalization
    best_model.zip, best_model_vecnormalize.pkl       best model on the evaluation env
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import kittra_sim  # noqa: F401  (registers KittraPlace-v0)
from kittra_sim.factory import make_place_env


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/train/ppo_place.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ppo_place"))
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    n_envs = args.n_envs or cfg["n_envs"]
    args.out.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(make_place_env, n_envs=n_envs, seed=cfg["seed"],
                        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
                        env_kwargs={"config": cfg["env_config"]})
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg["ppo"]["gamma"])

    eval_env = make_vec_env(make_place_env, n_envs=1, seed=cfg["seed"] + 10_000,
                            env_kwargs={"config": cfg["eval_config"]})
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    model = PPO("MlpPolicy", venv, seed=cfg["seed"], device=args.device, verbose=1,
                tensorboard_log="runs/ppo_place", policy_kwargs=cfg["policy_kwargs"], **cfg["ppo"])
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(cfg["eval"]["freq"] // n_envs, 1),
        n_eval_episodes=cfg["eval"]["n_episodes"],
        best_model_save_path=str(args.out),
        log_path=str(args.out / "eval"),
        deterministic=True,
        callback_on_new_best=SaveVecNormalizeOnBest(args.out / "best_model_vecnormalize.pkl"),
    )
    model.learn(total_timesteps=args.timesteps or cfg["total_timesteps"], callback=eval_cb, progress_bar=False)

    model.save(str(args.out / "ppo_place"))
    venv.save(str(args.out / "ppo_place_vecnormalize.pkl"))
    print(f"saved {args.out / 'ppo_place.zip'} and {args.out / 'best_model.zip'}")
    venv.close()
    eval_env.close()


if __name__ == "__main__":
    main()

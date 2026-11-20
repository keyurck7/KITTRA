"""Unified evaluation: same seeds, same metrics, every policy, every condition.

Episode ``i`` of a condition always uses seed ``base_seed + i``, so different policies face exactly
the same part layouts and perturbation samples.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kittra_learning.metrics import summarize_episodes
from kittra_learning.policies import load_policy


def make_env(env_name: str, condition: str):
    from kittra_sim.factory import make_pick_place_env, make_place_env

    if env_name == "pick_place":
        return make_pick_place_env(condition, obs_mode="rgb")
    if env_name == "place":
        return make_place_env(condition)
    raise ValueError(f"unknown env '{env_name}'")


def run_episodes(env, policy, n_episodes: int, base_seed: int,
                 log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    episodes = []
    for i in range(n_episodes):
        seed = base_seed + i
        obs, info = env.reset(seed=seed)
        policy.reset(obs, info)
        done, steps, t_inf = False, 0, 0.0
        while not done:
            t0 = time.perf_counter()
            action = policy.act(obs, info)
            t_inf += time.perf_counter() - t0
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        ep = {
            "seed": seed,
            "success": bool(info.get("is_success", False)),
            "length": steps,
            "dropped": bool(info.get("part_dropped", False)),
            "placement_error_mm": 1000.0 * float(info["placement_error"]) if "placement_error" in info else None,
            "inference_ms": 1000.0 * t_inf / max(steps, 1),
            "unstable": bool(info.get("sim_unstable", False)),
        }
        episodes.append(ep)
        log(f"  [{i + 1:3d}/{n_episodes}] seed={seed} success={ep['success']} steps={steps}")
    return episodes


def evaluate(policy_type: str, policy_path: str | None, conditions: list[str], n_episodes: int, base_seed: int,
             out_dir: str | Path, policy_name: str, env_name: str, device: str = "cpu",
             metadata: dict | None = None, log: Callable[[str], None] = print) -> list[dict]:
    out_dir = Path(out_dir) / policy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    policy = None
    for cond in conditions:
        log(f"== {policy_name} | {env_name} | condition={cond} | {n_episodes} episodes ==")
        env = make_env(env_name, cond)
        # Scripted/random/SB3 policies hold a reference to the env; LeRobot/BC policies are env-independent.
        if policy is None or policy_type in ("expert", "place_expert", "random", "sb3"):
            policy = load_policy(policy_type, policy_path, env=env, device=device)
        episodes = run_episodes(env, policy, n_episodes, base_seed, log=log)
        env.close()
        summary = summarize_episodes(episodes)
        record = {
            "policy": policy_name,
            "policy_type": policy_type,
            "policy_path": str(policy_path) if policy_path else None,
            "env": env_name,
            "condition": cond,
            "base_seed": base_seed,
            **(metadata or {}),
            "summary": summary,
            "episodes": episodes,
        }
        path = out_dir / f"{cond}.json"
        path.write_text(json.dumps(record, indent=2, default=_json_default))
        log(f"   success {100 * summary['success_rate']:.1f} %  "
            f"[{100 * summary['success_ci95'][0]:.1f}, {100 * summary['success_ci95'][1]:.1f}]  -> {path}")
        results.append(record)
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))

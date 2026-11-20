#!/usr/bin/env python3
"""Measure simulation throughput on this machine (Phase 1 results table).

    python scripts/benchmark_sim.py               # state-only and RGB observation modes
    MUJOCO_GL=egl python scripts/benchmark_sim.py # headless rendering on Linux/WSL2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from kittra_sim.constants import CONTROL_DT
from kittra_sim.env import KittraPickPlaceEnv


def bench(obs_mode: str, steps: int) -> dict:
    env = KittraPickPlaceEnv("nominal", obs_mode=obs_mode)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    home = env.current_command()
    t0 = time.perf_counter()
    for i in range(steps):
        a = home + np.concatenate([rng.normal(0, 0.05, 5), [0.0]])
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset(seed=i)
    dt = time.perf_counter() - t0
    env.close()
    return {
        "obs_mode": obs_mode,
        "control_steps_per_s": steps / dt,
        "x_realtime": steps * CONTROL_DT / dt,
        "physics_hz": 1.0 / env.model.opt.timestep,
        "substeps": env.n_substeps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--modes", nargs="+", default=["state", "rgb"])
    args = ap.parse_args()
    results = [bench(m, args.steps) for m in args.modes]
    for r in results:
        print(f"{r['obs_mode']:5s}: {r['control_steps_per_s']:8.1f} steps/s  = {r['x_realtime']:6.1f} x real time")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

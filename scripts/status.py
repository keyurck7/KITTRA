#!/usr/bin/env python3
"""Show what has been produced so far (results, datasets, checkpoints) and what to run next.

    python scripts/status.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mark(ok: bool) -> str:
    return "✅" if ok else "⬜"


def section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def phase1() -> None:
    section("Phase 1 – scripted expert")
    for cfg in ("nominal", "randomized"):
        p = ROOT / "results" / "phase1" / f"expert_{cfg}.json"
        if p.exists():
            s = json.loads(p.read_text())["summary"]
            print(f"{mark(True)} expert {cfg:10s} {100 * s['success_rate']:.0f} % over {s['episodes']} episodes")
        else:
            print(f"{mark(False)} expert {cfg:10s} missing  -> make expert")


def ppo() -> list[str]:
    section("Phase 2 – PPO placement skill")
    todo = []
    out = ROOT / "outputs" / "ppo_place"
    final, best = out / "ppo_place.zip", out / "best_model.zip"
    npz = out / "eval" / "evaluations.npz"
    total = 3_000_000
    try:
        import yaml

        total = int(yaml.safe_load((ROOT / "configs/train/ppo_place.yaml").read_text())["total_timesteps"])
    except Exception:
        pass
    if npz.exists():
        d = np.load(npz)
        steps = int(d["timesteps"][-1])
        rewards = d["results"].mean(axis=1)
        line = f"   evaluations: {len(d['timesteps'])}, last at {steps:,} / {total:,} steps ({100 * steps / total:.0f} %)"
        if "successes" in d.files and len(d["successes"]):
            succ = d["successes"].mean(axis=1)
            i = int(np.argmax(rewards))
            line += (f"\n   success: last {100 * succ[-1]:.0f} %, at best reward {100 * succ[i]:.0f} %"
                     f" (step {int(d['timesteps'][i]):,}), max {100 * succ.max():.0f} %")
        print(line)
    ckpts = sorted((out / "checkpoints").glob("ppo_place_*_steps.zip")) if (out / "checkpoints").exists() else []
    print(f"{mark(final.exists())} final model ppo_place.zip")
    print(f"{mark(best.exists())} best model best_model.zip"
          + ("" if (out / "best_model_vecnormalize.pkl").exists() or not best.exists() else "  (⚠ normalization stats missing)"))
    print(f"   checkpoints: {len(ckpts)}" + (f", newest {ckpts[-1].name}" if ckpts else ""))
    if not final.exists():
        todo.append("python scripts/train_ppo.py --resume" if (ckpts or best.exists()) else "make train-ppo")
    return todo


def bc() -> list[str]:
    section("Phase 2 – behaviour-cloning baseline")
    p = ROOT / "outputs" / "bc_place" / "train_info.json"
    if (ROOT / "outputs" / "bc_place" / "bc_place.pt").exists():
        info = json.loads(p.read_text()) if p.exists() else {}
        print(f"{mark(True)} bc_place.pt  (expert success {100 * info.get('expert_success_rate', float('nan')):.0f} %, "
              f"{info.get('transitions', '?')} transitions, val loss {info.get('best_val_loss', float('nan')):.4f})")
        return []
    print(f"{mark(False)} bc_place.pt missing")
    return ["make train-bc"]


def datasets_and_il() -> None:
    section("Phase 2 – datasets and ACT / Diffusion (GPU server)")
    for ds in ("kittra_nominal", "kittra_randomized", "kittra_randomized_50", "kittra_randomized_100"):
        meta = ROOT / "data" / "lerobot" / ds / "kittra_recording.json"
        print(f"{mark(meta.exists())} dataset {ds}")
    for p in ("act", "diffusion"):
        for d in ("nominal", "randomized", "randomized_50", "randomized_100"):
            ck = ROOT / "outputs" / f"{p}_{d}" / "checkpoints" / "last" / "pretrained_model"
            if ck.exists():
                print(f"{mark(True)} checkpoint {p}_{d}")
    if not any((ROOT / "outputs").glob("*_*/checkpoints/last/pretrained_model")):
        print("⬜ no ACT / Diffusion checkpoints yet (trained on Data Lab)")


def evaluations() -> list[str]:
    section("Evaluations (results/runs)")
    runs = sorted((ROOT / "results" / "runs").glob("*/*.json")) if (ROOT / "results" / "runs").exists() else []
    if not runs:
        print("⬜ none yet")
    by_policy: dict[str, list[str]] = {}
    for r in runs:
        by_policy.setdefault(r.parent.name, []).append(r.stem)
    for pol, conds in by_policy.items():
        print(f"✅ {pol}: {', '.join(sorted(conds))}")
    todo = []
    if (ROOT / "outputs/ppo_place/best_model.zip").exists() and "ppo_place" not in by_policy:
        todo.append("make eval-place")
    if "place_expert" not in by_policy:
        todo.append("python scripts/evaluate.py --policy-type place_expert --name place_expert")
    return todo


def main() -> None:
    phase1()
    todo = ppo() + bc()
    datasets_and_il()
    todo += evaluations()
    section("Next on this laptop")
    if todo:
        for t in dict.fromkeys(todo):
            print(f"  {t}")
    else:
        print("  laptop work done; continue with Data Lab")


if __name__ == "__main__":
    main()

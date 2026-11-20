#!/usr/bin/env python3
"""Aggregate evaluation runs into markdown tables (and optional plots).

    python scripts/make_results_table.py --runs results/runs --out results/phase2_table.md
    python scripts/make_results_table.py --runs results/runs --out results/phase2_table.md --plots results/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kittra_learning.results import build_report, load_runs


def plot_data_efficiency(runs, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eff = [r for r in runs if r.get("env") == "pick_place" and r.get("n_demos")
           and r["condition"] == "randomized" and r.get("train_data") == "randomized"]
    if not eff:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for ptype in sorted({r["policy_type"] for r in eff}):
        pts = sorted((int(r["n_demos"]), r["summary"]) for r in eff if r["policy_type"] == ptype)
        xs = [n for n, _ in pts]
        ys = [100 * s["success_rate"] for _, s in pts]
        lo = [100 * (s["success_rate"] - s["success_ci95"][0]) for _, s in pts]
        hi = [100 * (s["success_ci95"][1] - s["success_rate"]) for _, s in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=ptype)
    ax.set_xlabel("demonstrations")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "data_efficiency.png", dpi=150)
    print(f"wrote {out_dir / 'data_efficiency.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/phase2_table.md"))
    ap.add_argument("--ablation-policy", default=None, help="only show this policy in the ablation table")
    ap.add_argument("--plots", type=Path, default=None)
    args = ap.parse_args()

    runs = load_runs(args.runs)
    print(f"loaded {len(runs)} runs from {args.runs}")
    report = build_report(runs, ablation_policy=args.ablation_policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    print(report)
    print(f"\nwrote {args.out}")
    if args.plots:
        plot_data_efficiency(runs, args.plots)


if __name__ == "__main__":
    main()

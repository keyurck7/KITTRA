#!/usr/bin/env python3
"""Learning curves and final results for the PPO placement skill, plus a numbers summary.

    python scripts/plot_ppo_training.py
    python scripts/plot_ppo_training.py --runs outputs/ppo_place outputs/ppo_place_seed1

Reads
    <run>/eval/evaluations.npz          in-training evaluations (50 episodes each, randomized config)
    runs/ppo_place/PPO_*/               TensorBoard logs: training episode lengths and wall-clock time
    results/runs/<name>/<cond>.json     final 100-episode evaluations (PPO seeds, BC, scripted expert)

Writes (results/plots/ and results/)
    ppo_learning_curve.png              success rate vs environment steps, one line per seed
    placement_final_eval.png            final evaluation: expert / PPO / BC under 3 conditions
    ppo_training_summary.png            both panels side by side (for sharing)
    phase2_ppo_training.md              steps, episodes, wall time, steps-to-threshold, final results
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reference palette (light mode): colour follows the entity in every chart.
SURFACE = "#fcfcfb"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
COLORS = {"ppo_place": "#2a78d6", "ppo_place_seed1": "#eb6834", "bc_place": "#1baf7a", "place_expert": "#8a8985"}
LABELS = {"ppo_place": "PPO seed 0", "ppo_place_seed1": "PPO seed 1",
          "bc_place": "Behaviour cloning", "place_expert": "Scripted expert"}
CONDITIONS = ("nominal", "randomized", "ood")
COND_LABELS = {"nominal": "Nominal", "randomized": "Randomized\n(training conditions)", "ood": "Out-of-distribution\n(never seen)"}
THRESHOLDS = (0.5, 0.7, 0.8, 0.9)


@dataclass
class Run:
    name: str
    steps: np.ndarray
    success: np.ndarray
    reward: np.ndarray
    episodes_at: np.ndarray | None = None      # cumulative training episodes at each eval step
    episodes_exact: bool = False
    wall_hours: float | None = None
    tb_dir: str | None = None
    extra: dict = field(default_factory=dict)


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    """Trailing mean over k evaluations (no look-ahead, so thresholds are not reached early)."""
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = y[max(0, i - k + 1): i + 1].mean()
    return out


def load_tb_runs(tb_root: Path) -> list[dict]:
    """All TensorBoard runs with their training episode lengths and wall-clock span."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed: training-episode counts will be estimated")
        return []
    runs = []
    for d in sorted(p for p in tb_root.glob("*") if p.is_dir()):
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "rollout/ep_len_mean" not in tags:
            continue
        ev = ea.Scalars("rollout/ep_len_mean")
        steps = np.array([e.step for e in ev], dtype=float)
        lens = np.array([e.value for e in ev], dtype=float)
        walls = [e.wall_time for e in ev]
        runs.append({"dir": str(d), "steps": steps, "ep_len": lens, "wall_hours": (max(walls) - min(walls)) / 3600})
    return runs


def attach_episodes(run: Run, tb_runs: list[dict]) -> None:
    """Training episodes = sum over rollouts of (steps in rollout / mean episode length)."""
    last = run.steps[-1]
    best = None
    for tb in tb_runs:
        if len(tb["steps"]) and abs(tb["steps"][-1] - last) <= 0.05 * last:
            if best is None or abs(tb["steps"][-1] - last) < abs(best["steps"][-1] - last):
                best = tb
    if best is not None:
        s, ln = best["steps"], np.maximum(best["ep_len"], 1.0)
        delta = np.diff(np.concatenate([[0.0], s]))
        cum = np.cumsum(delta / ln)
        run.episodes_at = np.interp(run.steps, s, cum)
        run.episodes_exact = True
        run.wall_hours = best["wall_hours"]
        run.tb_dir = best["dir"]


def load_run(path: Path) -> Run | None:
    npz = path / "eval" / "evaluations.npz"
    if not npz.exists():
        print(f"skip {path}: no {npz}")
        return None
    d = np.load(npz)
    success = d["successes"].mean(axis=1) if "successes" in d.files else np.full(len(d["timesteps"]), np.nan)
    run = Run(name=path.name, steps=d["timesteps"].astype(float), success=success, reward=d["results"].mean(axis=1))
    run.extra["eval_ep_len"] = d["ep_lengths"].mean(axis=1)
    return run


def estimate_episodes(run: Run) -> None:
    """Fallback without TensorBoard: deterministic eval episode length as a proxy (labelled as an estimate)."""
    ln = np.maximum(run.extra["eval_ep_len"], 1.0)
    delta = np.diff(np.concatenate([[0.0], run.steps]))
    run.episodes_at = np.cumsum(delta / ln)
    run.episodes_exact = False


def first_reach(run: Run, thr: float) -> tuple[float, float] | None:
    sm = smooth(run.success)
    idx = np.flatnonzero(sm >= thr)
    if not len(idx):
        return None
    i = int(idx[0])
    return float(run.steps[i]), float(run.episodes_at[i]) if run.episodes_at is not None else float("nan")


def load_final(results_root: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for p in sorted(results_root.glob("*/*.json")):
        rec = json.loads(p.read_text())
        if rec.get("env") == "place" and rec["condition"] in CONDITIONS:
            out.setdefault(rec["policy"], {})[rec["condition"]] = rec["summary"]
    return out


# --------------------------------------------------------------------------- plotting
def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_steps(x: float) -> str:
    return f"{x / 1e6:.2f} M" if x >= 1e6 else f"{x / 1e3:.0f} k"


def plot_curve(ax, runs: list[Run], final: dict) -> None:
    style(ax)
    for run in runs:
        c = COLORS.get(run.name, "#6250d6")
        ax.plot(run.steps / 1e6, 100 * run.success, color=c, alpha=0.25, linewidth=1)
        ax.plot(run.steps / 1e6, 100 * smooth(run.success), color=c, linewidth=2, label=LABELS.get(run.name, run.name))
        b = int(np.argmax(run.reward))
        ax.plot(run.steps[b] / 1e6, 100 * run.success[b], "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=5,
                label=f"selected checkpoint ({fmt_steps(run.steps[b])} steps)")
    refs = [("place_expert", "Scripted expert"), ("bc_place", "Behaviour cloning")]
    xmax = max(r.steps[-1] for r in runs) / 1e6
    for key, lab in refs:
        s = final.get(key, {}).get("randomized")
        if s:
            y = 100 * s["success_rate"]
            ax.axhline(y, color=COLORS[key], linestyle=(0, (4, 3)), linewidth=1.5)
            ax.text(xmax, y + 1.5, f"{lab}: {y:.0f} %", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Environment steps (millions)", color=INK_2, fontsize=9)
    ax.set_ylabel("Success rate (%)", color=INK_2, fontsize=9)
    ax.set_title("Learning curve: PPO placement skill\n"
                 "eval every 25k steps · 50 episodes · randomized · thick line = mean of last 3 evals",
                 fontsize=10, color=INK, loc="left")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)


def plot_final(ax, final: dict) -> None:
    style(ax)
    order = [k for k in ("place_expert", "ppo_place", "ppo_place_seed1", "bc_place") if k in final]
    n = len(order)
    width = 0.8 / max(n, 1)
    x = np.arange(len(CONDITIONS))
    for i, key in enumerate(order):
        vals, lo, hi = [], [], []
        for cond in CONDITIONS:
            s = final[key].get(cond)
            r = 100 * s["success_rate"] if s else np.nan
            vals.append(r)
            lo.append(max(0.0, r - 100 * s["success_ci95"][0]) if s else 0)
            hi.append(max(0.0, 100 * s["success_ci95"][1] - r) if s else 0)
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, vals, width * 0.92, color=COLORS[key], label=LABELS[key], edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5)
        for xi, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(xi, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5,
                        color="#ffffff" if key != "place_expert" else INK, fontweight="bold")
    ax.set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], fontsize=9, color=INK_2)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Success rate (%)", color=INK_2, fontsize=9)
    ax.set_title("Final evaluation: 100 fixed-seed episodes per bar\n(error bars = 95 % Wilson CI)",
                 fontsize=10, color=INK, loc="left")
    leg = ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=len(order))
    for t in leg.get_texts():
        t.set_color(INK)


# --------------------------------------------------------------------------- summary
def summary_md(runs: list[Run], final: dict) -> str:
    lines = ["# Kittra – PPO placement skill: training and results", "",
             "In-training evaluation: every 25,000 environment steps, 50 episodes, randomized conditions, "
             "deterministic policy. Thresholds use a trailing mean over 3 evaluations. "
             "Final evaluation: 100 episodes per condition with fixed seeds 1000–1099, shared by all policies.", "",
             "## Training", "",
             "| | " + " | ".join(LABELS.get(r.name, r.name) for r in runs) + " |",
             "|---|" + "---|" * len(runs)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(r) for r in runs) + " |")

    row("Environment steps trained", lambda r: f"{r.steps[-1]:,.0f}")
    row("Training episodes", lambda r: (f"{r.episodes_at[-1]:,.0f}" if r.episodes_exact else f"≈ {r.episodes_at[-1]:,.0f} (est.)")
        if r.episodes_at is not None else "–")
    row("Wall-clock time (laptop CPU, 8 envs)", lambda r: f"{r.wall_hours:.1f} h" if r.wall_hours else "–")
    row("Evaluations", lambda r: str(len(r.steps)))
    for thr in THRESHOLDS:
        def f(r, thr=thr):
            hit = first_reach(r, thr)
            if hit is None:
                return "not reached"
            ep = f" / {hit[1]:,.0f} ep." if not np.isnan(hit[1]) else ""
            return f"{fmt_steps(hit[0])} steps{ep}"
        row(f"First reaches {int(thr * 100)} % success", f)
    row("Best in-training eval (selected checkpoint)",
        lambda r: f"{100 * r.success[int(np.argmax(r.reward))]:.0f} % at {fmt_steps(r.steps[int(np.argmax(r.reward))])}")
    row("Mean of last 5 evaluations", lambda r: f"{100 * r.success[-5:].mean():.0f} %")

    lines += ["", "## Final evaluation (success rate, 95 % CI, mean placement error)", "",
              "| Policy | " + " | ".join(c.capitalize() if c != "ood" else "Out-of-distribution" for c in CONDITIONS) + " |",
              "|---|---|---|---|"]
    for key in ("place_expert", "ppo_place", "ppo_place_seed1", "bc_place"):
        if key not in final:
            continue
        cells = []
        for c in CONDITIONS:
            s = final[key].get(c)
            if not s:
                cells.append("–")
                continue
            lo, hi = (100 * v for v in s["success_ci95"])
            err = s.get("mean_placement_error_mm_successful")
            cells.append(f"{100 * s['success_rate']:.0f} % [{lo:.0f}, {hi:.0f}]" + (f" · {err:.1f} mm" if err else ""))
        lines.append(f"| {LABELS[key]} | " + " | ".join(cells) + " |")
    seeds = [k for k in ("ppo_place", "ppo_place_seed1") if k in final]
    if len(seeds) == 2:
        lines += ["", "**PPO mean over 2 seeds:** " + ", ".join(
            f"{c}: {100 * np.mean([final[k][c]['success_rate'] for k in seeds]):.0f} %"
            for c in CONDITIONS if all(c in final[k] for k in seeds))]
    if any(not r.episodes_exact for r in runs):
        lines += ["", "_Episode counts marked “est.” are estimated from evaluation episode lengths "
                      "(TensorBoard logs not found)._"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, default=[Path("outputs/ppo_place"), Path("outputs/ppo_place_seed1")])
    ap.add_argument("--tb", type=Path, default=Path("runs/ppo_place"))
    ap.add_argument("--results", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in (load_run(p) for p in args.runs) if r is not None]
    if not runs:
        raise SystemExit("no evaluations.npz found")
    tb_runs = load_tb_runs(args.tb) if args.tb.exists() else []
    for r in runs:
        attach_episodes(r, tb_runs)
        if r.episodes_at is None:
            estimate_episodes(r)
        print(f"{r.name}: {len(r.steps)} evals up to {r.steps[-1]:,.0f} steps; "
              f"episodes {'from ' + r.tb_dir if r.episodes_exact else 'estimated'}")
    final = load_final(args.results)

    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    plot_curve(ax, runs, final)
    fig.tight_layout()
    fig.savefig(plots / "ppo_learning_curve.png", dpi=200)
    plt.close(fig)

    if final:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        plot_final(ax, final)
        fig.tight_layout()
        fig.savefig(plots / "placement_final_eval.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    plot_curve(axes[0], runs, final)
    if final:
        plot_final(axes[1], final)
    fig.suptitle("Kittra – learned placement skill (MuJoCo, SO-101-class arm): PPO vs behaviour cloning",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(plots / "ppo_training_summary.png", dpi=200)
    plt.close(fig)

    md = summary_md(runs, final)
    (args.out / "phase2_ppo_training.md").write_text(md)
    print(md)
    print(f"wrote {plots}/ppo_learning_curve.png, placement_final_eval.png, ppo_training_summary.png "
          f"and {args.out}/phase2_ppo_training.md")


if __name__ == "__main__":
    main()

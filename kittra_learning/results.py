"""Collect evaluation run files (results/runs/**/*.json) and render the README result tables.

Each run file is written by :mod:`kittra_learning.evaluation` and looks like::

    {"policy": "act_randomized_200", "policy_type": "act", "env": "pick_place",
     "condition": "randomized", "train_data": "randomized", "n_demos": 200,
     "summary": {...}, "episodes": [...]}

Pure Python, so tables can be rebuilt anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from kittra_learning.metrics import format_rate

MAIN_CONDITIONS = ("nominal", "randomized", "ood")
ABLATION_CONDITIONS = (
    "nominal", "abl_latency", "abl_backlash", "abl_camera", "abl_lighting",
    "abl_colors", "abl_friction", "abl_joint_noise", "abl_image_noise",
)
ABLATION_LABELS = {
    "nominal": "None",
    "abl_latency": "Action latency (3 steps = 100 ms)",
    "abl_backlash": "Backlash (up to 0.02 rad)",
    "abl_camera": "Camera pose offset (±2 cm, ±4°)",
    "abl_lighting": "Lighting change",
    "abl_colors": "Colour change",
    "abl_friction": "Low friction (×0.4–0.6)",
    "abl_joint_noise": "Joint encoder noise + bias",
    "abl_image_noise": "Image noise, blur, exposure",
}


def load_runs(root: str | Path) -> list[dict]:
    runs = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and {"policy", "condition", "summary"} <= data.keys():
            data["_path"] = str(p)
            runs.append(data)
    return runs


def _index(runs: Iterable[Mapping]) -> dict[tuple[str, str], Mapping]:
    """(policy, condition) -> run; the most recently listed file wins."""
    return {(r["policy"], r["condition"]): r for r in runs}


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def pivot(
    runs: Sequence[Mapping],
    columns: Sequence[str],
    cell: Callable[[Mapping], str],
    row_filter: Callable[[Mapping], bool] = lambda r: True,
    row_label: Callable[[Mapping], str] = lambda r: r["policy"],
    first_header: str = "Policy",
    column_labels: Mapping[str, str] | None = None,
) -> str:
    idx = _index(runs)
    policies: list[str] = []
    labels: dict[str, str] = {}
    for r in runs:
        if row_filter(r) and r["policy"] not in policies:
            policies.append(r["policy"])
            labels[r["policy"]] = row_label(r)
    if not policies:
        return "_no runs found_"
    header = [first_header] + [(column_labels or {}).get(c, c) for c in columns]
    rows = [[labels[p]] + [cell(idx[(p, c)]["summary"]) if (p, c) in idx else "–" for c in columns] for p in policies]
    return markdown_table(header, rows)


def placement_cell(summary: Mapping) -> str:
    rate = format_rate(summary, with_ci=False)
    err = summary.get("mean_placement_error_mm_successful") or summary.get("mean_placement_error_mm")
    return f"{rate} / {err:.1f} mm" if err is not None and rate != "–" else rate


def build_report(runs: Sequence[Mapping], ablation_policy: str | None = None) -> str:
    pick = [r for r in runs if r.get("env") == "pick_place"]
    place = [r for r in runs if r.get("env") == "place"]
    out = ["# Kittra – Phase 2 results", "",
           "Success rate with 95 % Wilson confidence interval; 100 episodes per cell unless noted.", ""]

    out += ["## Imitation learning (pick-and-place)", "",
            pivot(pick, MAIN_CONDITIONS, format_rate, row_filter=lambda r: r["condition"] in MAIN_CONDITIONS), ""]

    eff = [r for r in pick if r.get("n_demos") and r["condition"] == "randomized" and r.get("train_data") == "randomized"]
    if eff:
        demo_counts = sorted({int(r["n_demos"]) for r in eff})
        idx: dict[tuple[str, int], Mapping] = {(r["policy_type"], int(r["n_demos"])): r for r in eff}
        types = sorted({r["policy_type"] for r in eff})
        rows = [[t] + [format_rate(idx[(t, n)]["summary"], with_ci=False) if (t, n) in idx else "–" for n in demo_counts]
                for t in types]
        out += ["## Data efficiency (randomized training data, randomized test)", "",
                markdown_table(["Policy"] + [f"{n} demos" for n in demo_counts], rows), ""]

    out += ["## Placement skill (success rate / mean placement error)", "",
            pivot(place, MAIN_CONDITIONS, placement_cell, row_filter=lambda r: r["condition"] in MAIN_CONDITIONS), ""]

    abl = [r for r in pick if r["condition"] in ABLATION_CONDITIONS and (ablation_policy is None or r["policy"] == ablation_policy)]
    abl_policies = {r["policy"] for r in abl if r["condition"].startswith("abl_")}
    if abl_policies:
        out += ["## Ablation: which perturbation hurts most?", ""]
        idx2 = _index(abl)
        for pol in sorted(abl_policies):
            rows = [[ABLATION_LABELS[c], format_rate(idx2[(pol, c)]["summary"]) if (pol, c) in idx2 else "–"]
                    for c in ABLATION_CONDITIONS]
            out += [f"**{pol}**", "", markdown_table(["Perturbation", "Success rate"], rows), ""]

    timing = [r for r in runs if r["summary"].get("mean_inference_ms") is not None and r["condition"] == "nominal"]
    if timing:
        rows = [[r["policy"], f"{r['summary']['mean_inference_ms']:.2f}"] for r in timing]
        out += ["## Inference time per control step (nominal condition)", "",
                markdown_table(["Policy", "ms / step"], rows), ""]
    return "\n".join(out)

"""Evaluation statistics. Pure Python/NumPy."""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion (well-behaved for small n and p near 0/1)."""
    if n <= 0:
        return (0.0, 0.0)
    if not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_episodes(episodes: Sequence[Mapping]) -> dict:
    """Aggregate per-episode records into the metrics reported in the README."""
    n = len(episodes)
    if n == 0:
        return {"episodes": 0}
    succ = [bool(e["success"]) for e in episodes]
    k = int(sum(succ))
    lo, hi = wilson_interval(k, n)
    out = {
        "episodes": n,
        "successes": k,
        "success_rate": k / n,
        "success_ci95": [lo, hi],
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "drop_rate": float(np.mean([bool(e.get("dropped", False)) for e in episodes])),
    }
    succ_len = [e["length"] for e in episodes if e["success"]]
    out["mean_length_successful"] = float(np.mean(succ_len)) if succ_len else None
    errs = [e["placement_error_mm"] for e in episodes if e.get("placement_error_mm") is not None]
    out["mean_placement_error_mm"] = float(np.mean(errs)) if errs else None
    succ_errs = [e["placement_error_mm"] for e, s in zip(episodes, succ) if s and e.get("placement_error_mm") is not None]
    out["mean_placement_error_mm_successful"] = float(np.mean(succ_errs)) if succ_errs else None
    inf = [e["inference_ms"] for e in episodes if e.get("inference_ms") is not None]
    out["mean_inference_ms"] = float(np.mean(inf)) if inf else None
    return out


def format_rate(summary: Mapping, with_ci: bool = True) -> str:
    """'73.0 % [63.6, 80.7]' style cell for markdown tables."""
    if not summary or not summary.get("episodes"):
        return "–"
    rate = 100 * summary["success_rate"]
    if not with_ci:
        return f"{rate:.1f} %"
    lo, hi = (100 * v for v in summary["success_ci95"])
    return f"{rate:.1f} % [{lo:.1f}, {hi:.1f}]"


def mean_or_none(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None

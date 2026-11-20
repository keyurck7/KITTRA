import json

import pytest

from kittra_learning.metrics import format_rate, summarize_episodes, wilson_interval
from kittra_learning.results import build_report, load_runs


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(73, 100)
    assert lo == pytest.approx(0.6358, abs=1e-3)
    assert hi == pytest.approx(0.8072, abs=1e-3)
    assert wilson_interval(0, 50)[0] == 0.0 and wilson_interval(0, 50)[1] > 0
    assert wilson_interval(50, 50)[1] == 1.0 and wilson_interval(50, 50)[0] < 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


def _episodes(n_succ, n):
    return [{"success": i < n_succ, "length": 200 + i, "dropped": False,
             "placement_error_mm": 3.0 + i % 3, "inference_ms": 2.0} for i in range(n)]


def test_summary_and_format():
    s = summarize_episodes(_episodes(7, 10))
    assert s["success_rate"] == pytest.approx(0.7)
    assert s["mean_length_successful"] == pytest.approx(203.0)
    assert format_rate(s).startswith("70.0 % [")
    assert format_rate({}) == "–"


def test_report_from_run_files(tmp_path):
    def write(policy, ptype, env, cond, k, n, **meta):
        rec = {"policy": policy, "policy_type": ptype, "env": env, "condition": cond, **meta,
               "summary": summarize_episodes(_episodes(k, n)), "episodes": []}
        d = tmp_path / policy
        d.mkdir(exist_ok=True)
        (d / f"{cond}.json").write_text(json.dumps(rec))

    for cond, k in (("nominal", 90), ("randomized", 80), ("ood", 50)):
        write("act_randomized_200", "act", "pick_place", cond, k, 100, train_data="randomized", n_demos=200)
        write("ppo_place", "sb3", "place", cond, k - 5, 100)
    write("act_randomized_50", "act", "pick_place", "randomized", 40, 100, train_data="randomized", n_demos=50)
    write("act_nominal_200", "act", "pick_place", "abl_latency", 30, 100, train_data="nominal", n_demos=200)
    write("act_nominal_200", "act", "pick_place", "nominal", 95, 100, train_data="nominal", n_demos=200)

    runs = load_runs(tmp_path)
    assert len(runs) == 9
    report = build_report(runs)
    assert "| act_randomized_200 | 90.0 % [" in report
    assert "50 demos" in report and "200 demos" in report
    assert "ppo_place" in report and "mm" in report
    assert "Action latency" in report

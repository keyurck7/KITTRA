#!/usr/bin/env python3
"""Evaluate any Kittra policy under any set of conditions with fixed seeds.

    # Imitation learning (pick-and-place, image-based)
    python scripts/evaluate.py --policy-type act \
        --policy outputs/act_randomized/checkpoints/last/pretrained_model \
        --name act_randomized_200 --train-data randomized --n-demos 200 \
        --conditions nominal randomized ood --episodes 100 --device cuda

    # Placement skill (state-based)
    python scripts/evaluate.py --policy-type sb3 --policy outputs/ppo_place/best_model.zip --name ppo_place
    python scripts/evaluate.py --policy-type bc  --policy outputs/bc_place/bc_place.pt     --name bc_place

    # Scripted experts (privileged upper bounds)
    python scripts/evaluate.py --policy-type expert --name expert
    python scripts/evaluate.py --policy-type place_expert --name place_expert

    # Single-perturbation ablation
    python scripts/evaluate.py --policy-type act --policy <ckpt> --name act_nominal_200 --conditions ablation

Writes results/runs/<name>/<condition>.json; build tables with scripts/make_results_table.py.
"""
from __future__ import annotations

import argparse

from kittra_learning.evaluation import evaluate
from kittra_learning.policies import POLICY_ENV
from kittra_learning.results import ABLATION_CONDITIONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-type", required=True, choices=sorted(set(POLICY_ENV) | {"random"}))
    ap.add_argument("--policy", default=None, help="checkpoint path (not needed for experts / random)")
    ap.add_argument("--name", required=True, help="run name used in the results tables")
    ap.add_argument("--env", default=None, choices=["pick_place", "place"],
                    help="defaults to the env that matches the policy type")
    ap.add_argument("--conditions", nargs="+", default=["nominal", "randomized", "ood"],
                    help="config names; 'ablation' expands to all single-perturbation configs")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000, help="base seed (keep it identical across policies)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--train-data", default=None, help="metadata: training data variant (nominal/randomized)")
    ap.add_argument("--n-demos", type=int, default=None, help="metadata: number of training demonstrations")
    args = ap.parse_args()

    env_name = args.env or POLICY_ENV.get(args.policy_type)
    if env_name is None:
        ap.error("--env is required for policy type 'random'")
    conditions: list[str] = []
    for c in args.conditions:
        conditions += list(ABLATION_CONDITIONS) if c == "ablation" else [c]

    evaluate(args.policy_type, args.policy, conditions, args.episodes, args.seed, args.out, args.name, env_name,
             device=args.device, metadata={"train_data": args.train_data, "n_demos": args.n_demos})


if __name__ == "__main__":
    main()

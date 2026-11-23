#!/usr/bin/env bash
# Phase 2 imitation-learning pipeline, unattended (GPU server).
#
#   nohup bash scripts/run_il_pipeline.sh > logs/pipeline.log 2>&1 &
#   tail -f logs/pipeline.log
#
# Stages (each skips work that already exists, so the script can simply be re-run after an interruption):
#   1. record the four LeRobot datasets          (nominal 200, randomized 200 / 100 / 50)
#   2. train ACT + Diffusion on the 200-demo datasets, 4 runs in parallel on one GPU
#   3. train ACT + Diffusion on the 50/100-demo datasets, 4 runs in parallel
#   4. evaluate every checkpoint (nominal / randomized / OOD, 100 episodes) + the ablation, in parallel
#   5. build results/phase2_table.md
#
# Environment overrides: STEPS (training steps, default from configs/train/*.yaml), EPISODES (default 100),
# STAGES (default "1 2 3 4 5"), e.g.  STAGES="4 5" bash scripts/run_il_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p logs
EPISODES=${EPISODES:-100}
STAGES=${STAGES:-"1 2 3 4 5"}
STEPS_ARG=${STEPS:+--steps $STEPS}
FAILED=0

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }

record() {  # config episodes dataset
  if [ -f "data/lerobot/$3/kittra_recording.json" ]; then stamp "dataset $3 exists, skip"; return; fi
  stamp "recording $3"
  python scripts/record_lerobot_dataset.py --config "$1" --episodes "$2" --overwrite > "logs/record_$3.log" 2>&1 \
    && stamp "recorded $3" || { stamp "FAILED recording $3 (logs/record_$3.log)"; FAILED=1; }
}

train() {  # policy dataset
  local run="$1_${2#kittra_}"
  if [ -d "outputs/$run/checkpoints/last/pretrained_model" ]; then stamp "$run trained, skip"; return; fi
  rm -rf "outputs/$run"   # a partial run without a final checkpoint is restarted
  stamp "training $run"
  python scripts/train_lerobot.py --config "configs/train/$1.yaml" --dataset "$2" $STEPS_ARG > "logs/train_$run.log" 2>&1 \
    && stamp "trained $run" || { stamp "FAILED training $run (logs/train_$run.log)"; FAILED=1; }
}

evaluate() {  # policy variant [conditions...]
  local p=$1 d=$2; shift 2
  local ck="outputs/${p}_$d/checkpoints/last/pretrained_model"
  [ -d "$ck" ] || { stamp "no checkpoint for ${p}_$d, skip eval"; return; }
  local data=${d%%_*} n
  n=$(echo "$d" | grep -oE '[0-9]+$' || echo 200)
  local name="${p}_${data}_${n}"
  stamp "evaluating $name ($*)"
  python scripts/evaluate.py --policy-type "$p" --policy "$ck" --name "$name" --train-data "$data" --n-demos "$n" \
    --conditions "$@" --episodes "$EPISODES" --device cuda > "logs/eval_${name}_$1.log" 2>&1 \
    && stamp "evaluated $name ($*)" || { stamp "FAILED eval $name (logs/eval_${name}_$1.log)"; FAILED=1; }
}

if want 1; then
  stamp "== stage 1: datasets =="
  # Sequential on purpose: ~2 s per episode, and parallel writers would share the Hugging Face cache.
  record nominal 200 kittra_nominal
  record randomized 200 kittra_randomized
  record randomized 100 kittra_randomized_100
  record randomized 50 kittra_randomized_50
fi

if want 2; then
  stamp "== stage 2: 200-demo runs =="
  for ds in kittra_randomized kittra_nominal; do for p in act diffusion; do train "$p" "$ds" & sleep 20; done; done
  wait
fi

if want 3; then
  stamp "== stage 3: data-efficiency runs =="
  for ds in kittra_randomized_50 kittra_randomized_100; do for p in act diffusion; do train "$p" "$ds" & sleep 20; done; done
  wait
fi

if want 4; then
  stamp "== stage 4: evaluation =="
  for p in act diffusion; do for d in randomized nominal randomized_50 randomized_100; do
    evaluate "$p" "$d" nominal randomized ood &
  done; done
  # single-perturbation ablation (the nominal condition is already covered above)
  evaluate act nominal abl_latency abl_backlash abl_camera abl_lighting abl_colors abl_friction abl_joint_noise abl_image_noise &
  wait
fi

if want 5; then
  stamp "== stage 5: results =="
  python scripts/make_results_table.py --runs results/runs --out results/phase2_table.md --plots results/plots \
    > logs/results.log 2>&1 && cat results/phase2_table.md
fi

[ "$FAILED" = 0 ] && stamp "PIPELINE DONE" || stamp "PIPELINE DONE WITH FAILURES (see FAILED lines above)"

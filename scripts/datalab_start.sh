#!/usr/bin/env bash
# Start (or continue) the Phase 2 imitation-learning pipeline on the SRH Data Lab JupyterHub.
#
# On Data Lab only ~/vault survives a server stop, so the repository (and with it data/, outputs/, logs/ and
# results/) lives in ~/vault/kittra. The venv lives outside vault and is rebuilt automatically after a stop.
#
#   first time:                git clone https://github.com/keyurck7/KITTRA.git ~/vault/kittra
#   after every server start:  bash ~/vault/kittra/scripts/datalab_start.sh
#
# The pipeline skips finished datasets, runs and evaluations and resumes interrupted training from its last
# checkpoint, so running this again after a stop loses at most the work since the last checkpoint
# (ACT: 20k steps ~ 20 min, Diffusion: 10k steps ~ 45 min).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$HOME/venvs/kittra}
cd "$REPO"
mkdir -p logs

case "$REPO/" in
  "$HOME"/vault/*) ;;
  *) echo "WARNING: $REPO is not inside ~/vault, so datasets and checkpoints are lost when the server stops" ;;
esac

if pgrep -f "scripts/run_il_pipeline.sh" > /dev/null; then
  echo "pipeline is already running:"; tail -5 logs/pipeline.log; exit 0
fi

free_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
echo "free space next to the repo: ${free_gb} GB"
if [ "${free_gb:-0}" -lt 25 ]; then
  echo "WARNING: less than 25 GB free. Peak use is ~20 GB (4 datasets + 4 runs in parallel with 2 checkpoints each);"
  echo "         finished runs shrink to the final weights only (~6 GB for all eight)."
fi

if ! "$VENV/bin/python" -c "import lerobot, transformers, diffusers, mujoco, kittra_sim" 2> /dev/null; then
  echo "building the venv in $VENV (~5-10 min, it is rebuilt after every server stop)"
  python3 -m venv "$VENV"          # clean venv: no --system-site-packages (see requirements-train.txt)
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements-train.txt
  "$VENV/bin/pip" install -q -e .
fi
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
print('GPU:', torch.cuda.get_device_name(0))"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export MUJOCO_GL=egl
nohup bash scripts/run_il_pipeline.sh >> logs/pipeline.log 2>&1 &
sleep 3
echo "pipeline started (pid $!). Follow it with:"
echo "  tail -f $REPO/logs/pipeline.log"

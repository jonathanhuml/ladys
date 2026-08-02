#!/usr/bin/env bash
set -euo pipefail

cd /home/jon/ladys

EPOCHS="${EPOCHS:-}"
EVAL_EVERY="${EVAL_EVERY:-500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
PATIENCE_EVALS="${PATIENCE_EVALS:-20}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-hal_full}"
OUT="${OUT:-runs/stndt_nlb_reproduction_hal_full}"
CONFIG_LIST="${CONFIG_LIST:-}"
DATASETS="${DATASETS:-area2_bump mc_rtt dmfc_rsg}"
LOG="$OUT/stndt_reproduction.log"

mkdir -p "$OUT"

args=(
  scripts/run_stndt_nlb_reproduction.py
  --device cuda
  --output-dir "$OUT"
  --eval-every "$EVAL_EVERY"
  --progress-every "$PROGRESS_EVERY"
  --patience-evals "$PATIENCE_EVALS"
  --run-name-suffix "$RUN_NAME_SUFFIX"
)

if [[ -n "$CONFIG_LIST" ]]; then
  args+=(--config-list "$CONFIG_LIST")
else
  # shellcheck disable=SC2206
  dataset_args=($DATASETS)
  args+=(--datasets "${dataset_args[@]}")
fi

if [[ -n "$EPOCHS" ]]; then
  args+=(--epochs "$EPOCHS")
fi

nohup env \
  PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX=/tmp/ladys_pycache \
  PYTHONPATH=/home/jon/ladys:/home/jon/ladys/src \
  /home/jon/torch-gpu/bin/python "${args[@]}" \
  > "$LOG" 2>&1 &

pid="$!"
printf '%s\n' "$pid" > "$OUT/pid.txt"
printf 'pid=%s\nout=%s\nlog=%s\n' "$pid" "$OUT" "$LOG"

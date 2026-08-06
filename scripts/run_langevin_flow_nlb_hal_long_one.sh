#!/usr/bin/env bash
set -u

DATASET="${1:?dataset required}"
cd /home/jon/ladys

OUT="${OUT:-runs/langevin_flow_long_20260802}"
mkdir -p "$OUT/logs"

case "$DATASET" in
  area2_bump)
    CONFIG="configs/experiment/real/area2_bump/langevin_flow/langevin_flow_area2_bump_nlb_5ms_canonical.yaml"
    EVAL_EVERY=75
    PROGRESS_EVERY=25
    ;;
  dmfc_rsg)
    CONFIG="configs/experiment/real/dmfc_rsg/langevin_flow/langevin_flow_dmfc_rsg_nlb_5ms_canonical.yaml"
    EVAL_EVERY=80
    PROGRESS_EVERY=25
    ;;
  mc_maze)
    CONFIG="configs/experiment/real/mc_maze/langevin_flow/langevin_flow_mc_maze_nlb_5ms_canonical.yaml"
    EVAL_EVERY=85
    PROGRESS_EVERY=25
    ;;
  mc_rtt)
    CONFIG="configs/experiment/real/mc_rtt/langevin_flow/langevin_flow_mc_rtt_nlb_5ms_canonical.yaml"
    EVAL_EVERY=110
    PROGRESS_EVERY=50
    ;;
  *)
    echo "unknown dataset: $DATASET" >&2
    exit 2
    ;;
esac

LOG="$OUT/logs/${DATASET}.log"
EXIT="$OUT/logs/${DATASET}.exit"
WRAPPER="$OUT/logs/${DATASET}.wrapper.log"

exec >> "$WRAPPER" 2>&1
date
echo "starting_langevin_flow_long dataset=$DATASET config=$CONFIG eval_every=$EVAL_EVERY"

env PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX=/tmp/ladys_pycache \
  PYTHONPATH=/home/jon/ladys:/home/jon/ladys/src \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  /home/jon/torch-gpu/bin/python /home/jon/ladys/scripts/run_langevin_flow_nlb_reproduction.py \
    --config "$CONFIG" \
    --device cuda \
    --output-dir "$OUT" \
    --eval-every "$EVAL_EVERY" \
    --progress-every "$PROGRESS_EVERY" \
    --patience-evals 0 \
    --stop-at-reported \
    --run-name-suffix long_20260802 > "$LOG" 2>&1
code=$?
date > "$EXIT"
echo "exit=$code" >> "$EXIT"
echo "finished_langevin_flow_long dataset=$DATASET exit=$code"
exit $code

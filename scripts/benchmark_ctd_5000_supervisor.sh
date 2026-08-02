#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/jon/ladys}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/ctd_500trials_5000neurons_benchmark_20260731}"
DATASET_CONFIG_DIR="${DATASET_CONFIG_DIR:-configs/dataset_500trials_5000neurons}"
DATASETS="${DATASETS:-lower}"
MODELS="${MODELS:-psth smoothing gpfa kalman cassm ndt stndt lfads mint bgpfa langevin_flow}"
EXPECTED_CASES="${EXPECTED_CASES:-44}"
MAX_CASE_SECONDS="${MAX_CASE_SECONDS:-14400}"
STALE_SECONDS="${STALE_SECONDS:-18000}"
PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"
WAIT_FOR_OUTPUT_DIR="${WAIT_FOR_OUTPUT_DIR:-$REPO_DIR/runs/ctd_rate_mse_lower_full_hal_mint_20260731_1250}"
WAIT_FOR_EXPECTED_CASES="${WAIT_FOR_EXPECTED_CASES:-4}"

PID_FILE="$OUTPUT_DIR/benchmark.pid"
LOG_PATH="$OUTPUT_DIR/benchmark.log"
SUPERVISOR_LOG="$OUTPUT_DIR/supervisor.log"
WATCHDOG="$REPO_DIR/scripts/benchmark_ctd_rate_mse_watchdog.sh"

mkdir -p "$OUTPUT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" >> "$SUPERVISOR_LOG"
}

pid_alive() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

benchmark_active_for_dir() {
  local dir="$1"
  pgrep -af 'benchmark_ctd_rate_mse.py' \
    | grep -F -- "$dir" \
    | grep -v grep >/dev/null 2>&1
}

row_count() {
  local path="$1"
  [[ -f "$path" ]] || {
    printf '0\n'
    return
  }
  awk -F, 'NR>1 {n++} END {print n+0}' "$path"
}

status_count() {
  local status="$1"
  local path="$OUTPUT_DIR/summary.csv"
  [[ -f "$path" ]] || {
    printf '0\n'
    return
  }
  awk -F, -v status="$status" 'NR>1 && $1==status {n++} END {print n+0}' "$path"
}

wait_target_done() {
  if benchmark_active_for_dir "$WAIT_FOR_OUTPUT_DIR"; then
    return 1
  fi
  local rows
  rows="$(row_count "$WAIT_FOR_OUTPUT_DIR/summary.csv")"
  [[ "$rows" -ge "$WAIT_FOR_EXPECTED_CASES" ]]
}

launch_benchmark() {
  cd "$REPO_DIR" || exit 1
  PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX=/tmp/ladys_pycache \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$PYTHON_BIN" scripts/benchmark_ctd_rate_mse.py \
    --datasets $DATASETS \
    --models $MODELS \
    --dataset-config-dir "$DATASET_CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --require-cuda \
    --trainer-config-source lorenz \
    --model-config-source lorenz \
    --bgpfa-infer-steps 500 \
    --bgpfa-infer-mc 20 \
    --max-case-seconds "$MAX_CASE_SECONDS" \
    --seed 1 \
    --retry-errors \
    >> "$LOG_PATH" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$PID_FILE"
  log "launched pid=$(cat "$PID_FILE") output_dir=$OUTPUT_DIR datasets=$DATASETS models=$MODELS dataset_config_dir=$DATASET_CONFIG_DIR"
}

if [[ -x "$WATCHDOG" ]]; then
  "$WATCHDOG" "$OUTPUT_DIR" "$PID_FILE" "$STALE_SECONDS"
fi

ok_rows="$(status_count ok)"
partial_rows="$(status_count partial)"
error_rows="$(status_count error)"
done_rows="$((ok_rows + partial_rows + error_rows))"

if pid_alive; then
  log "alive pid=$(cat "$PID_FILE") ok_rows=$ok_rows partial_rows=$partial_rows error_rows=$error_rows"
  exit 0
fi

if [[ "$done_rows" -ge "$EXPECTED_CASES" ]]; then
  log "complete done_rows=$done_rows ok_rows=$ok_rows partial_rows=$partial_rows error_rows=$error_rows expected=$EXPECTED_CASES"
  exit 0
fi

if ! wait_target_done; then
  wait_rows="$(row_count "$WAIT_FOR_OUTPUT_DIR/summary.csv")"
  log "waiting for dependency output_dir=$WAIT_FOR_OUTPUT_DIR rows=$wait_rows expected=$WAIT_FOR_EXPECTED_CASES"
  exit 0
fi

log "not running; launching done_rows=$done_rows ok_rows=$ok_rows partial_rows=$partial_rows error_rows=$error_rows expected=$EXPECTED_CASES"
launch_benchmark

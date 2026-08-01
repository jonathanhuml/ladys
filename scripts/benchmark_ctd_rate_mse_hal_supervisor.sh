#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/jon/ladys}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/ctd_rate_mse_lower_full_hal_lorenz_configs_20260728_2003}"
EXPECTED_CASES="${EXPECTED_CASES:-48}"
STALE_SECONDS="${STALE_SECONDS:-7200}"
PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"

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

count_rows() {
  local status="$1"
  local path="$OUTPUT_DIR/summary.csv"
  [[ -f "$path" ]] || {
    printf '0\n'
    return
  }
  awk -F, -v status="$status" 'NR>1 && $1==status {n++} END {print n+0}' "$path"
}

launch_benchmark() {
  cd "$REPO_DIR" || exit 1
  PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX=/tmp/ladys_pycache \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$PYTHON_BIN" scripts/benchmark_ctd_rate_mse.py \
    --datasets lower \
    --models psth smoothing gpfa kalman cassm ndt stndt lfads ilqr_vae mint bgpfa langevin_flow \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --require-cuda \
    --trainer-config-source lorenz \
    --model-config-source lorenz \
    --bgpfa-infer-steps 200 \
    --bgpfa-infer-mc 20 \
    --seed 1 \
    --retry-errors \
    >> "$LOG_PATH" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$PID_FILE"
  log "launched pid=$(cat "$PID_FILE") output_dir=$OUTPUT_DIR"
}

if [[ -x "$WATCHDOG" ]]; then
  "$WATCHDOG" "$OUTPUT_DIR" "$PID_FILE" "$STALE_SECONDS"
fi

ok_rows="$(count_rows ok)"
error_rows="$(count_rows error)"

if pid_alive; then
  log "alive pid=$(cat "$PID_FILE") ok_rows=$ok_rows error_rows=$error_rows"
  exit 0
fi

if [[ "$ok_rows" -ge "$EXPECTED_CASES" && "$error_rows" -eq 0 ]]; then
  log "complete ok_rows=$ok_rows error_rows=$error_rows"
  exit 0
fi

log "not running; restarting ok_rows=$ok_rows error_rows=$error_rows expected=$EXPECTED_CASES"
launch_benchmark

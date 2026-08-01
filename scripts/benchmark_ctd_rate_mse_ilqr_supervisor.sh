#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/jon/ladys}"
MASTER_OUTPUT_DIR="${MASTER_OUTPUT_DIR:-$REPO_DIR/runs/ctd_rate_mse_lower_full_hal_lorenz_configs_20260728_2003}"
EXTRA_OUTPUT_DIR="${EXTRA_OUTPUT_DIR:-$REPO_DIR/runs/ctd_rate_mse_lower_full_hal_ilqr_mint_20260730_2145}"
EXTRA_METHODS="${EXTRA_METHODS:-ilqr_vae mint}"
EXPECTED_CASES="${EXPECTED_CASES:-8}"
STALE_SECONDS="${STALE_SECONDS:-7200}"
MAX_CASE_SECONDS="${MAX_CASE_SECONDS:-14400}"
PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"

PID_FILE="$EXTRA_OUTPUT_DIR/benchmark.pid"
LOG_PATH="$EXTRA_OUTPUT_DIR/benchmark.log"
SUPERVISOR_LOG="$EXTRA_OUTPUT_DIR/supervisor.log"
WATCHDOG="$REPO_DIR/scripts/benchmark_ctd_rate_mse_watchdog.sh"

mkdir -p "$EXTRA_OUTPUT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" >> "$SUPERVISOR_LOG"
}

pid_alive() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

master_active() {
  pgrep -af 'benchmark_ctd_rate_mse.py' \
    | grep -F -- "$MASTER_OUTPUT_DIR" \
    | grep -v -F -- "$EXTRA_OUTPUT_DIR" \
    | grep -v grep >/dev/null 2>&1
}

count_rows() {
  local status="$1"
  local path="$EXTRA_OUTPUT_DIR/summary.csv"
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
    --models $EXTRA_METHODS \
    --output-dir "$EXTRA_OUTPUT_DIR" \
    --device cuda \
    --require-cuda \
    --trainer-config-source lorenz \
    --model-config-source lorenz \
    --max-case-seconds "$MAX_CASE_SECONDS" \
    --seed 1 \
    --retry-errors \
    >> "$LOG_PATH" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$PID_FILE"
  log "launched pid=$(cat "$PID_FILE") methods=$EXTRA_METHODS output_dir=$EXTRA_OUTPUT_DIR"
}

merge_into_master_if_safe() {
  if master_active; then
    log "extra methods complete but master sweep still active; waiting to merge"
    return
  fi

  MASTER_OUTPUT_DIR="$MASTER_OUTPUT_DIR" EXTRA_OUTPUT_DIR="$EXTRA_OUTPUT_DIR" python3 - <<'PY'
import csv
import os
import shutil
import time
from pathlib import Path

master = Path(os.environ["MASTER_OUTPUT_DIR"])
extra = Path(os.environ["EXTRA_OUTPUT_DIR"])
summary_path = master / "summary.csv"
history_path = master / "history.csv"
extra_summary_path = extra / "summary.csv"
extra_history_path = extra / "history.csv"

if not extra_summary_path.exists():
    raise SystemExit("missing extra-method summary")

stamp = time.strftime("%Y%m%d_%H%M%S")
for path in [summary_path, history_path]:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + f".pre_ilqr_merge_{stamp}"))

def read_rows(path):
    if not path.exists():
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])

def write_rows(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

summary_rows, summary_fields = read_rows(summary_path)
extra_summary_rows, extra_summary_fields = read_rows(extra_summary_path)
summary_fields = summary_fields or extra_summary_fields
summary_keys = {
    (row.get("dataset"), row.get("model"), row.get("seed"))
    for row in extra_summary_rows
}
summary_rows = [
    row for row in summary_rows
    if (row.get("dataset"), row.get("model"), row.get("seed")) not in summary_keys
]
summary_rows.extend(extra_summary_rows)
summary_rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", ""), row.get("seed", "")))
write_rows(summary_path, summary_rows, summary_fields)

history_rows, history_fields = read_rows(history_path)
extra_history_rows, extra_history_fields = read_rows(extra_history_path)
history_fields = history_fields or extra_history_fields
history_keys = {
    (row.get("dataset"), row.get("model"), row.get("seed"))
    for row in extra_history_rows
}
history_rows = [
    row for row in history_rows
    if (row.get("dataset"), row.get("model"), row.get("seed")) not in history_keys
]
history_rows.extend(extra_history_rows)
history_rows.sort(
    key=lambda row: (
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("seed", ""),
        int(float(row.get("epoch") or 0)),
    )
)
write_rows(history_path, history_rows, history_fields)

(master / "extra_methods_merge.done").write_text(
    f"merged {len(extra_summary_rows)} extra-method rows from {extra}\n"
)
PY
  log "merged extra-method rows into $MASTER_OUTPUT_DIR"
}

if [[ -x "$WATCHDOG" ]]; then
  "$WATCHDOG" "$EXTRA_OUTPUT_DIR" "$PID_FILE" "$STALE_SECONDS"
fi

ok_rows="$(count_rows ok)"
partial_rows="$(count_rows partial)"
done_rows="$((ok_rows + partial_rows))"
error_rows="$(count_rows error)"

if pid_alive; then
  log "alive pid=$(cat "$PID_FILE") ok_rows=$ok_rows partial_rows=$partial_rows error_rows=$error_rows"
  exit 0
fi

if [[ "$done_rows" -ge "$EXPECTED_CASES" && "$error_rows" -eq 0 ]]; then
  merge_into_master_if_safe
  exit 0
fi

log "not running; restarting done_rows=$done_rows ok_rows=$ok_rows partial_rows=$partial_rows error_rows=$error_rows expected=$EXPECTED_CASES"
launch_benchmark

#!/usr/bin/env bash
set -uo pipefail

OUTPUT_DIR="${1:?usage: benchmark_ctd_rate_mse_watchdog.sh OUTPUT_DIR [PID_FILE] [STALE_SECONDS]}"
PID_FILE="${2:-$OUTPUT_DIR/benchmark.pid}"
STALE_SECONDS="${3:-7200}"
LOG_PATH="$OUTPUT_DIR/watchdog.log"
SUMMARY_PATH="$OUTPUT_DIR/summary.csv"
HISTORY_PATH="$OUTPUT_DIR/history.csv"
HEARTBEAT_PATH="$OUTPUT_DIR/heartbeat.json"

mkdir -p "$OUTPUT_DIR"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] watchdog start output_dir=$OUTPUT_DIR"

  if [[ -f "$PID_FILE" ]]; then
    PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      echo "process=alive pid=$PID"
    else
      echo "process=dead pid=${PID:-missing}"
    fi
  else
    echo "process=unknown pid_file_missing=$PID_FILE"
  fi

  if [[ -f "$SUMMARY_PATH" ]]; then
    TOTAL_ROWS="$(tail -n +2 "$SUMMARY_PATH" | wc -l | tr -d ' ')"
    OK_ROWS="$(awk -F, 'NR>1 && $1=="ok" {n++} END {print n+0}' "$SUMMARY_PATH")"
    ERROR_ROWS="$(awk -F, 'NR>1 && $1=="error" {n++} END {print n+0}' "$SUMMARY_PATH")"
    echo "summary_rows=$TOTAL_ROWS ok_rows=$OK_ROWS error_rows=$ERROR_ROWS"
  else
    echo "summary_missing=$SUMMARY_PATH"
  fi

  if [[ -f "$HISTORY_PATH" ]]; then
    HISTORY_ROWS="$(tail -n +2 "$HISTORY_PATH" | wc -l | tr -d ' ')"
    echo "history_rows=$HISTORY_ROWS"
  else
    echo "history_missing=$HISTORY_PATH"
  fi

  if [[ -f "$HEARTBEAT_PATH" ]]; then
    python3 - "$HEARTBEAT_PATH" "$STALE_SECONDS" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
stale_seconds = float(sys.argv[2])
data = json.loads(path.read_text())
age = time.time() - float(data.get("timestamp", 0.0))
status = "stale" if age > stale_seconds else "fresh"
event = data.get("event", "")
dataset = data.get("dataset", "")
model = data.get("model", "")
epoch = data.get("epoch", "")
epochs = data.get("epochs", "")
case_index = data.get("case_index", "")
total_cases = data.get("total_cases", "")
print(
    "heartbeat="
    f"{status} age_seconds={age:.1f} event={event} "
    f"dataset={dataset} model={model} epoch={epoch}/{epochs} "
    f"case={case_index}/{total_cases}"
)
PY
  else
    echo "heartbeat_missing=$HEARTBEAT_PATH"
  fi

  echo "watchdog end"
} >> "$LOG_PATH" 2>&1

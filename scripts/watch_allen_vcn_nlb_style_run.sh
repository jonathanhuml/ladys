#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [PID]" >&2
  exit 2
fi

OUT="$1"
PID="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"

ROOT="${LADYS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

LOG="$OUT/overnight_watchdog.log"
SUMMARY="$OUT/summary.csv"
H5_OUT="$OUT/h5_predictions"
REPORT="$OUT/overnight_margin_report.txt"

mkdir -p "$OUT"
{
  echo "watchdog_start=$(date -Is)"
  echo "root=$ROOT"
  echo "out=$OUT"
} >> "$LOG"

if [[ -z "$PID" && -s "$OUT/pid.txt" ]]; then
  PID="$(tr -d '[:space:]' < "$OUT/pid.txt")"
fi

if [[ -n "$PID" ]]; then
  while kill -0 "$PID" 2>/dev/null; do
    {
      echo
      echo "still_running=$(date -Is) pid=$PID"
      if [[ -s "$SUMMARY" ]]; then
        tail -n 8 "$SUMMARY"
      else
        echo "summary_not_written_yet"
      fi
    } >> "$LOG"
    sleep "${WATCHDOG_SLEEP_SECONDS:-300}"
  done
fi

{
  echo
  echo "run_finished_or_pid_absent=$(date -Is)"
  if [[ -s "$OUT/sweep.log" ]]; then
    tail -n 80 "$OUT/sweep.log"
  fi
} >> "$LOG"

if [[ ! -s "$SUMMARY" ]]; then
  echo "summary_missing=$SUMMARY" >> "$LOG"
  exit 3
fi

PYTHONPYCACHEPREFIX=/tmp/ladys_pycache PYTHONPATH=src "$PYTHON_BIN" \
  scripts/export_allen_vcn_predictions_h5.py \
  --summary "$SUMMARY" \
  --output-dir "$H5_OUT" \
  --include-targets >> "$LOG" 2>&1

PYTHONPYCACHEPREFIX=/tmp/ladys_pycache PYTHONPATH=src "$PYTHON_BIN" - "$SUMMARY" "$REPORT" <<'PY' >> "$LOG" 2>&1
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

summary = Path(sys.argv[1])
report = Path(sys.argv[2])

rows = []
with summary.open(newline="") as f:
    for row in csv.DictReader(f):
        try:
            row["co_bps_float"] = float(row["co_bps"])
        except (KeyError, TypeError, ValueError):
            row["co_bps_float"] = math.nan
        rows.append(row)

by_group = defaultdict(list)
for row in rows:
    by_group[row.get("group", "")].append(row)

lines = []
for group in sorted(by_group):
    group_rows = by_group[group]
    smoothing = next((r["co_bps_float"] for r in group_rows if r.get("method") == "smoothing"), math.nan)
    lines.append(f"[{group}] smoothing={smoothing:.9f}")
    for row in sorted(group_rows, key=lambda r: (math.isnan(r["co_bps_float"]), -r["co_bps_float"], r.get("method", ""))):
        co_bps = row["co_bps_float"]
        margin = co_bps - smoothing if not math.isnan(smoothing) and not math.isnan(co_bps) else math.nan
        if math.isnan(margin):
            status = "missing"
        elif margin >= 0:
            status = "beats"
        elif margin >= -0.02:
            status = "slight_miss"
        else:
            status = "miss"
        lines.append(f"  {row.get('method',''):<14} co_bps={co_bps: .9f} margin={margin: .9f} {status}")
    lines.append("")

report.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote_report={report}")
PY

echo "watchdog_done=$(date -Is)" >> "$LOG"

#!/usr/bin/env bash
# Stable HAL entry point, invoked with the existing approved SSH prefix.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action=${1:-status}
shift || true
export PYTHONPATH="src:.:${PYTHONPATH:-}"
if [ -d test-deps ]; then
  export PYTHONPATH="test-deps:$PYTHONPATH"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
case "$action" in
  run)
    python=$1
    shift
    exec "$python" scripts/run_synthetic_readiness.py "$@"
    ;;
  status)
    python3 - "$@" <<'PY'
import json
from pathlib import Path
import sys
for value in sys.argv[1:]:
    path = Path(value) / "status.json"
    for row in json.loads(path.read_text()):
        print(row["dataset"], row["model"], row["status"],
              f"{row.get('seconds', 0):.1f}s", row.get("completed_points", ""),
              row.get("final_rate_mse", ""), row.get("error", ""))
PY
    ;;
  *)
    printf 'Usage: hal_synthetic.sh {run <python> [args...]|status <output directories...>}\n' >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Run on HAL through the already-approved SSH prefix; no nested login shell.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
action=${1:-status}
if [ "$#" -gt 0 ]; then
  shift
fi

case "$action" in
  status)
    python3 - "$@" <<'PY'
import json
from pathlib import Path
import sys

roots = [Path(value) for value in sys.argv[1:]] or sorted(Path("runs").glob("*/summary.json"))
for root in roots:
    summary = root if root.is_file() else root / "summary.json"
    if not summary.exists():
        print(f"No summary yet: {summary}")
        continue
    print(f"\n{summary.parent}")
    print("Dataset       Model     Status    Epoch   Best co-BPS  Minutes  Phase")
    for row in json.loads(summary.read_text()):
        folder = summary.parent / row["dataset"] / row["model"]
        status = folder / "status.json"
        if status.exists():
            row = json.loads(status.read_text())
        score = row.get("best_co_bps")
        score_text = "" if score is None else f"{score:.5f}"
        print(f"{row['dataset']:13} {row['model']:9} {row['status']:9} "
              f"{row.get('epoch', '')!s:>5} {score_text:>13} "
              f"{row.get('seconds', 0) / 60:8.1f}  {row.get('phase', '')}")
        if row.get("status") == "failed":
            print(f"  {row.get('error', 'Worker failed; inspect run.log')}")
PY
    ;;
  gpu)
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
    ;;
  processes)
    pgrep -af 'run_nlb_model_comparison.py|profile_ilqr.py' || true
    ;;
  start)
    if [ "$#" -lt 2 ]; then
      printf 'Usage: hal_nlb.sh start <log directory> <python executable> [run arguments...]\n' >&2
      exit 2
    fi
    python3 - "$@" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
pid_file = folder / "launcher.json"
if pid_file.exists():
    previous = json.loads(pid_file.read_text())
    try:
        os.kill(previous["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit(f"Launcher PID {previous['pid']} is still alive; refusing duplicate launch")
command = ["bash", str(Path("scripts/hal_nlb.sh").resolve()), "run", *sys.argv[2:]]
with (folder / "launcher.log").open("a") as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=process.pid, command=command, log=str(folder / "launcher.log"))
pid_file.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
PY
    ;;
  stop)
    python3 - "$@" <<'PY'
import json
import os
from pathlib import Path
import signal
import sys

for value in sys.argv[1:]:
    pid = int(value)
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        print(f"PID {pid} has already exited")
        continue
    command = (proc / "cmdline").read_bytes().split(b"\0")
    if (proc / "cwd").resolve() != Path.cwd() or not any(
        arg.endswith(b"/run_nlb_model_comparison.py") for arg in command
    ):
        raise SystemExit(f"Refusing to stop PID {pid}: not this checkout's NLB runner")
    os.kill(pid, signal.SIGTERM)
    arguments = [arg.decode() for arg in command if arg]
    if "--worker" in arguments and "--output-dir" in arguments:
        worker = arguments.index("--worker")
        method, dataset = arguments[worker + 1:worker + 3]
        output = Path(arguments[arguments.index("--output-dir") + 1])
        status = output / dataset / method / "status.json"
        if status.exists():
            state = json.loads(status.read_text())
            state.update(status="interrupted", phase="stopped", stopped_pid=pid)
            temporary = status.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(status)
    print(f"Stopped NLB runner PID {pid}")
PY
    ;;
  run|test)
    if [ "$#" -eq 0 ]; then
      printf 'Usage: hal_nlb.sh %s <python executable> [arguments...]\n' "$action" >&2
      exit 2
    fi
    python=$1
    shift
    export PYTHONPATH="src:.:${PYTHONPATH:-}"
    if [ -d test-deps ]; then
      export PYTHONPATH="test-deps:$PYTHONPATH"
    fi
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export PYTHONUNBUFFERED=1
    if [ "$action" = run ]; then
      exec "$python" scripts/run_nlb_model_comparison.py "$@"
    else
      exec "$python" -m pytest "$@"
    fi
    ;;
  *)
    printf 'Usage: hal_nlb.sh {status [run directories...]|gpu|processes|run <python> [args...]|start <log directory> <python> [args...]|stop <runner PIDs...>|test <python> [args...]}\n' >&2
    exit 2
    ;;
esac

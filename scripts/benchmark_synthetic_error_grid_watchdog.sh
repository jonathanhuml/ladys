#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/jon/ladys}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jon/ladys/runs/synthetic_error_grid_psth_smoothing_20260725_15x10}"
PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"
EXPECTED_ROWS="${EXPECTED_ROWS:-1800}"
DATASET="${DATASET:-lorenz}"
GRID_Y="${GRID_Y:-neurons}"
FIXED_NEURONS="${FIXED_NEURONS:-500}"
WORKERS="${WORKERS:-8}"
PLOT_EVERY="${PLOT_EVERY:-25}"
NUM_NEURON_POINTS="${NUM_NEURON_POINTS:-15}"
NUM_TRIAL_POINTS="${NUM_TRIAL_POINTS:-10}"
MIN_TIME_POINTS="${MIN_TIME_POINTS:-10}"
MAX_TIME_POINTS="${MAX_TIME_POINTS:-1000}"
NUM_TIME_POINTS="${NUM_TIME_POINTS:-15}"
NUM_STEPS="${NUM_STEPS:-100}"
BURN_STEPS="${BURN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
MAX_DATASET_GB="${MAX_DATASET_GB:-}"
HEATMAP_TITLE="${HEATMAP_TITLE:-}"
MODELS="${MODELS:-bgpfa cassm gpfa ilqr_vae kalman langevin_flow lfads mint ndt stndt psth smoothing}"
PANEL_MODELS="${PANEL_MODELS:-all}"

if [[ -z "$HEATMAP_TITLE" && "$GRID_Y" == "time" ]]; then
    HEATMAP_TITLE="Error as a function of trials and time"
fi

PID_FILE="$OUTPUT_DIR/remaining_pid.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"
WATCHDOG_LOG="$OUTPUT_DIR/watchdog.log"
RUN_LOG="$OUTPUT_DIR/hal_remaining_methods_watchdog.log"
LOCK_FILE="$OUTPUT_DIR/watchdog.lock"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/errors" "$OUTPUT_DIR/plots"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0
fi

timestamp() {
    date "+%Y-%m-%d %H:%M:%S %z"
}

log() {
    printf "[%s] %s\n" "$(timestamp)" "$*" >> "$WATCHDOG_LOG"
}

summary_rows() {
    if [[ ! -f "$SUMMARY_FILE" ]]; then
        echo 0
        return
    fi
    local lines
    lines="$(wc -l < "$SUMMARY_FILE" | tr -d " ")"
    if [[ "$lines" -le 0 ]]; then
        echo 0
    else
        echo $((lines - 1))
    fi
}

cmdline_is_target_benchmark() {
    local pid="$1"
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    local cmdline
    cmdline="$(tr "\0" " " < "/proc/$pid/cmdline")"
    [[ "$cmdline" == *"benchmark_synthetic_error_grid.py"* && "$cmdline" == *"$OUTPUT_DIR"* ]]
}

running_pid() {
    local pid
    if [[ -s "$PID_FILE" ]]; then
        pid="$(tr -dc "0-9" < "$PID_FILE")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && cmdline_is_target_benchmark "$pid"; then
            echo "$pid"
            return 0
        fi
    fi

    while read -r pid; do
        [[ -n "$pid" ]] || continue
        if cmdline_is_target_benchmark "$pid"; then
            echo "$pid"
            return 0
        fi
    done < <(pgrep -u "$(id -u)" -f "benchmark_synthetic_error_grid.py" || true)

    return 1
}

cleanup_orphan_workers() {
    local pid ppid cwd
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        ppid="$(ps -o ppid= -p "$pid" | tr -d " " || true)"
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
        if [[ "$ppid" == "1" && "$cwd" == "$REPO_DIR" ]]; then
            log "terminating orphan multiprocessing worker pid=$pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done < <(pgrep -u "$(id -u)" -f "multiprocessing.spawn.*spawn_main" || true)
}

rows="$(summary_rows)"
if [[ "$rows" -ge "$EXPECTED_ROWS" ]]; then
    log "complete rows=$rows expected=$EXPECTED_ROWS; no action"
    exit 0
fi

if pid="$(running_pid)"; then
    log "running pid=$pid rows=$rows expected=$EXPECTED_ROWS; no action"
    exit 0
fi

cleanup_orphan_workers
rows="$(summary_rows)"
log "benchmark not running; restarting rows=$rows expected=$EXPECTED_ROWS"

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/tmp/ladys_pycache
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r -a model_args <<< "$MODELS"
read -r -a panel_model_args <<< "$PANEL_MODELS"
cmd=(
    "$PYTHON_BIN" scripts/benchmark_synthetic_error_grid.py
    --dataset "$DATASET"
    --grid-y "$GRID_Y"
    --models "${model_args[@]}"
    --panel-models "${panel_model_args[@]}"
    --num-neuron-points "$NUM_NEURON_POINTS"
    --num-trial-points "$NUM_TRIAL_POINTS"
    --num-steps "$NUM_STEPS"
    --burn-steps "$BURN_STEPS"
    --batch-size "$BATCH_SIZE"
    --device cuda \
    --require-cuda \
    --workers "$WORKERS" \
    --plot-every "$PLOT_EVERY" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS"
)
if [[ "$GRID_Y" == "time" ]]; then
    cmd+=(
        --fixed-neurons "$FIXED_NEURONS"
        --min-time-points "$MIN_TIME_POINTS"
        --max-time-points "$MAX_TIME_POINTS"
        --num-time-points "$NUM_TIME_POINTS"
    )
fi
if [[ -n "$MAX_DATASET_GB" ]]; then
    cmd+=(--max-dataset-gb "$MAX_DATASET_GB")
fi
if [[ -n "$HEATMAP_TITLE" ]]; then
    cmd+=(--heatmap-title "$HEATMAP_TITLE")
fi

nohup "${cmd[@]}" >> "$RUN_LOG" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"
log "launched pid=$pid dataset=$DATASET grid_y=$GRID_Y workers=$WORKERS plot_every=$PLOT_EVERY run_log=$RUN_LOG"

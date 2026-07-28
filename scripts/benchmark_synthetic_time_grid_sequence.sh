#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/jon/ladys}"
PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d)}"
FIXED_NEURONS="${FIXED_NEURONS:-500}"
NUM_TIME_POINTS="${NUM_TIME_POINTS:-15}"
NUM_TRIAL_POINTS="${NUM_TRIAL_POINTS:-10}"
MIN_TIME_POINTS="${MIN_TIME_POINTS:-10}"
MAX_TIME_POINTS="${MAX_TIME_POINTS:-1000}"
NUM_STEPS="${NUM_STEPS:-100}"
BURN_STEPS="${BURN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
WORKERS="${WORKERS:-8}"
PLOT_EVERY="${PLOT_EVERY:-25}"
EXPECTED_ROWS="${EXPECTED_ROWS:-1800}"
LORENZ_MAX_DATASET_GB="${LORENZ_MAX_DATASET_GB:-4.5}"
CHAOTIC_MAX_DATASET_GB="${CHAOTIC_MAX_DATASET_GB:-4.5}"
MODELS="${MODELS:-bgpfa cassm gpfa ilqr_vae kalman langevin_flow lfads mint ndt stndt psth smoothing}"
PANEL_MODELS="${PANEL_MODELS:-all}"
HEATMAP_TITLE="${HEATMAP_TITLE:-Error as a function of trials and time}"

RUNS_DIR="$REPO_DIR/runs"
LORENZ_OUTPUT_DIR="${LORENZ_OUTPUT_DIR:-$RUNS_DIR/synthetic_time_grid_lorenz_${RUN_STAMP}_${FIXED_NEURONS}n_${NUM_TIME_POINTS}x${NUM_TRIAL_POINTS}}"
CHAOTIC_OUTPUT_DIR="${CHAOTIC_OUTPUT_DIR:-$RUNS_DIR/synthetic_time_grid_chaotic_rnn_${RUN_STAMP}_${FIXED_NEURONS}n_${NUM_TIME_POINTS}x${NUM_TRIAL_POINTS}}"
SEQUENCE_LOG="${SEQUENCE_LOG:-$RUNS_DIR/synthetic_time_grid_sequence_${RUN_STAMP}.log}"
LOCK_FILE="${LOCK_FILE:-$RUNS_DIR/synthetic_time_grid_sequence_${RUN_STAMP}.lock}"

mkdir -p "$RUNS_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0
fi

timestamp() {
    date "+%Y-%m-%d %H:%M:%S %z"
}

log() {
    printf "[%s] %s\n" "$(timestamp)" "$*" >> "$SEQUENCE_LOG"
}

summary_rows() {
    local output_dir="$1"
    local summary_file="$output_dir/summary.csv"
    if [[ ! -f "$summary_file" ]]; then
        echo 0
        return
    fi
    local lines
    lines="$(wc -l < "$summary_file" | tr -d " ")"
    if [[ "$lines" -le 0 ]]; then
        echo 0
    else
        echo $((lines - 1))
    fi
}

run_watchdog() {
    local dataset="$1"
    local output_dir="$2"
    local max_dataset_gb="$3"

    log "checking dataset=$dataset output_dir=$output_dir"
    REPO_DIR="$REPO_DIR" \
    OUTPUT_DIR="$output_dir" \
    PYTHON_BIN="$PYTHON_BIN" \
    EXPECTED_ROWS="$EXPECTED_ROWS" \
    DATASET="$dataset" \
    GRID_Y=time \
    FIXED_NEURONS="$FIXED_NEURONS" \
    WORKERS="$WORKERS" \
    PLOT_EVERY="$PLOT_EVERY" \
    NUM_NEURON_POINTS=1 \
    NUM_TRIAL_POINTS="$NUM_TRIAL_POINTS" \
    MIN_TIME_POINTS="$MIN_TIME_POINTS" \
    MAX_TIME_POINTS="$MAX_TIME_POINTS" \
    NUM_TIME_POINTS="$NUM_TIME_POINTS" \
    NUM_STEPS="$NUM_STEPS" \
    BURN_STEPS="$BURN_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" \
    EPOCHS="$EPOCHS" \
    MAX_DATASET_GB="$max_dataset_gb" \
    HEATMAP_TITLE="$HEATMAP_TITLE" \
    MODELS="$MODELS" \
    PANEL_MODELS="$PANEL_MODELS" \
    "$REPO_DIR/scripts/benchmark_synthetic_error_grid_watchdog.sh"
}

lorenz_rows="$(summary_rows "$LORENZ_OUTPUT_DIR")"
if [[ "$lorenz_rows" -lt "$EXPECTED_ROWS" ]]; then
    log "phase=lorenz rows=$lorenz_rows expected=$EXPECTED_ROWS"
    run_watchdog lorenz "$LORENZ_OUTPUT_DIR" "$LORENZ_MAX_DATASET_GB"
    exit 0
fi

log "lorenz complete rows=$lorenz_rows expected=$EXPECTED_ROWS"

chaotic_rows="$(summary_rows "$CHAOTIC_OUTPUT_DIR")"
if [[ "$chaotic_rows" -lt "$EXPECTED_ROWS" ]]; then
    log "phase=chaotic_rnn rows=$chaotic_rows expected=$EXPECTED_ROWS"
    run_watchdog chaotic_rnn "$CHAOTIC_OUTPUT_DIR" "$CHAOTIC_MAX_DATASET_GB"
    exit 0
fi

log "all complete lorenz_rows=$lorenz_rows chaotic_rows=$chaotic_rows expected=$EXPECTED_ROWS"

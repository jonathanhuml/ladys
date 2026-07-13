#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jon/torch-gpu/bin/python}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-runs/ilqr_vae_hal_logs}"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PYTHONPATH:-src}"

run_job() {
  local label="$1"
  shift
  local log_path="${LOG_DIR}/${label}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date -Is)] starting ${label}" | tee "${log_path}"
  "${PYTHON_BIN}" -m ladys.cli run "$@" --device "${DEVICE}" 2>&1 | tee -a "${log_path}"
  echo "[$(date -Is)] finished ${label}" | tee -a "${log_path}"
}

should_run() {
  local label="$1"
  [[ -z "${RUN_ONLY:-}" || "${RUN_ONLY}" == "${label}" ]]
}

if should_run "mc_maze"; then
  run_job "mc_maze_checkpoint" \
    -c configs/experiment/real/mc_maze/ilqr_vae/ilqr_vae_mc_maze_nlb_5ms.yaml \
    --run-name ilqr_vae_mc_maze_nlb_5ms_checkpoint_gpu
fi

if should_run "area2_bump"; then
  run_job "area2_bump_train" \
    -c configs/experiment/real/area2_bump/ilqr_vae/ilqr_vae_area2_bump_nlb_5ms_train.yaml \
    --epochs "${EPOCHS_AREA2_BUMP:-25}" \
    --run-name ilqr_vae_area2_bump_nlb_5ms_train_gpu
fi

if should_run "dmfc_rsg"; then
  run_job "dmfc_rsg_train" \
    -c configs/experiment/real/dmfc_rsg/ilqr_vae/ilqr_vae_dmfc_rsg_nlb_5ms_train.yaml \
    --epochs "${EPOCHS_DMFC_RSG:-25}" \
    --run-name ilqr_vae_dmfc_rsg_nlb_5ms_train_gpu
fi

if should_run "mc_rtt"; then
  run_job "mc_rtt_train" \
    -c configs/experiment/real/mc_rtt/ilqr_vae/ilqr_vae_mc_rtt_nlb_5ms_train.yaml \
    --epochs "${EPOCHS_MC_RTT:-25}" \
    --run-name ilqr_vae_mc_rtt_nlb_5ms_train_gpu
fi

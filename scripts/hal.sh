#!/usr/bin/env bash
# One stable entry point for remote commands through the user's HAL SSH alias.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf 'Usage: bash scripts/hal.sh <remote command>\n' >&2
  exit 2
fi

exec ssh -o BatchMode=yes -o ConnectTimeout=10 HAL "$@"

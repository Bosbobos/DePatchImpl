#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
source "$VENV_DIR/bin/activate"

python scripts/prepare_celeb_fbi_640.py \
  --output-dir datasets/celeb_fbi_640 \
  --no-metadata \
  "$@"

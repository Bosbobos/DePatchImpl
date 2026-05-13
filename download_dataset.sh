#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
source "$VENV_DIR/bin/activate"

if [ -z "${HF_TOKEN:-}" ] && [ -t 0 ]; then
    read -r -s -p "Hugging Face token (press Enter to skip): " HF_TOKEN
    echo
    export HF_TOKEN
fi

python scripts/prepare_celeb_fbi_640.py \
  --output-dir datasets/celeb_fbi_640 \
  --no-metadata \
  "$@"

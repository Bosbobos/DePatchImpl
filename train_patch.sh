#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-2000}"

source "$VENV_DIR/bin/activate"

python depatch.py \
  --train-dir datasets/celeb_fbi_640/images \
  --val-dir datasets/celeb_fbi_640/images \
  --weights yolo11s.pt \
  --device auto \
  --output-dir outputs/depatch_celeb_fbi \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --cleanup-interval 0 \
  --cleanup-batch-interval 100 \
  --eval-interval 100 \
  --log-interval 50 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./setup_venv.sh
./download_dataset.sh
./train_patch.sh "$@"

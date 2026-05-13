#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt
python -m ipykernel install --user --name portable-depatch --display-name "Python (portable-depatch)"

echo "Venv is ready: $PWD/$VENV_DIR"

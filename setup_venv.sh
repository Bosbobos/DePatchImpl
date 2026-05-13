#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

install_python311_with_apt() {
    if ! command -v apt-get >/dev/null 2>&1; then
        return 1
    fi

    local sudo_cmd=()
    if [ "$(id -u)" -ne 0 ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            return 1
        fi
        sudo_cmd=(sudo)
    fi

    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y python3.11 python3.11-venv
}

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="python3.11"
    elif install_python311_with_apt && command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="python3.11"
    else
        PYTHON_BIN="python3"
    fi
fi

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.5.1 torchvision==0.20.1}"

"$PYTHON_BIN" - <<'PY'
import os
import sys

version = sys.version_info
if version < (3, 10):
    sys.exit("Python 3.10 or newer is required.")

if version >= (3, 13) and "TORCH_PACKAGES" not in os.environ:
    sys.exit(
        "The default CUDA 12.1 PyTorch pins are intended for Python 3.10-3.12. "
        "Use PYTHON_BIN=python3.11 ./setup_venv.sh, or set TORCH_PACKAGES and "
        "TORCH_INDEX_URL for a Python 3.13-compatible PyTorch build."
    )
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install $TORCH_PACKAGES --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt
python -m ipykernel install --user --name portable-depatch --display-name "Python (portable-depatch)"

echo "Venv is ready: $PWD/$VENV_DIR"

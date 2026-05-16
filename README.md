# Portable DePatch Celeb-FBI

Portable DePatch package for a CUDA server. It creates a local `venv`, downloads
Celeb-FBI from Hugging Face, resizes every image to `640x640` with letterbox
padding, and starts DePatch training on YOLO11s.

## One-command run on a fresh server

```bash
cd portable_depatch
./run_all.sh
```

Defaults:

- dataset: `alecccdd/celeb-fbi`
- output images: `datasets/celeb_fbi_640/images`
- patch outputs: `outputs/depatch_celeb_fbi`
- epochs: `2000`
- batch size: `64`
- memory cleanup: every `100` optimizer steps

## Change batch size

Either use an environment variable:

```bash
BATCH_SIZE=32 ./train_patch.sh
```

or pass the Python argument directly:

```bash
./train_patch.sh --batch-size 32
```

The same works with `run_all.sh`:

```bash
BATCH_SIZE=32 ./run_all.sh
```

Change epochs:

```bash
EPOCHS=100 ./train_patch.sh
```

## Separate steps

Install dependencies only:

```bash
cd portable_depatch
./setup_venv.sh
```

Download and prepare dataset only:

```bash
./download_dataset.sh
```

If the Hugging Face dataset requires authentication, enter an HF token when
prompted. Press Enter to continue without a token. You can also provide it
non-interactively:

```bash
HF_TOKEN=hf_... ./download_dataset.sh
```

Start training:

```bash
./train_patch.sh
```

If `outputs/depatch_celeb_fbi/latest_patch.pt` exists, training resumes from it
automatically. Use a different `--output-dir` or remove the checkpoint to start
from a fresh random patch.

## Backup training artifacts to git

In a separate terminal, run:

```bash
./scripts/git_backup_training_artifacts.sh
```

It commits and pushes the latest patch, best patch, history, and training
progress every 20 minutes to a separate `training-backup` branch using a
separate `.git-training-backup-worktree/` worktree, so backup commits do not
mix with code commits. To run one backup immediately:

```bash
./scripts/git_backup_training_artifacts.sh --once
```

Override the destination branch if needed:

```bash
BACKUP_BRANCH=my-training-backup ./scripts/git_backup_training_artifacts.sh
```

## Jupyter workflow

After `./setup_venv.sh` and `./download_dataset.sh`, start Jupyter:

```bash
source .venv/bin/activate
jupyter lab
```

Open `notebooks/DePatch_CelebFBI.ipynb` and select the
`Python (portable-depatch)` kernel.

## CUDA wheel selection

`setup_venv.sh` installs matching PyTorch and TorchVision CUDA 12.1 wheels by
default. It automatically uses `python3.11` when available, then `python3.12`,
then `python3`. On Ubuntu/Debian, it tries to install the matching `venv`
package through `apt-get` if it is missing:

```bash
./setup_venv.sh
```

You can still choose a specific Python 3.10-3.12 interpreter explicitly:

```bash
PYTHON_BIN=python3.12 ./setup_venv.sh
```

For a different CUDA/PyTorch wheel index or Python version, override
`TORCH_INDEX_URL` and `TORCH_PACKAGES`:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
TORCH_PACKAGES="torch torchvision" \
PYTHON_BIN=python3.12 \
./setup_venv.sh
```

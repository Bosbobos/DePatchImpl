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

Start training:

```bash
./train_patch.sh
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

`setup_venv.sh` installs PyTorch from CUDA 12.1 wheels by default:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./setup_venv.sh
```

For a different CUDA/PyTorch wheel index, override `TORCH_INDEX_URL`.

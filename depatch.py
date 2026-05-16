from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from ultralytics import YOLO

try:
    from IPython import get_ipython
except Exception:
    get_ipython = None

shell = get_ipython() if get_ipython is not None else None
if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


DEFAULT_TRAIN_DIR = "datasets/celeb_fbi_640/images"
DEFAULT_VAL_DIR = "datasets/celeb_fbi_640/images"
DEFAULT_FALLBACK_WEIGHTS = "yolo11s.pt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
BOX_CACHE_VERSION = 1


@dataclass
class PatchTrainerConfig:
    train_dir: str = DEFAULT_TRAIN_DIR
    val_dir: str = DEFAULT_VAL_DIR
    train_labels_dir: str | None = None
    val_labels_dir: str | None = None
    weights: str = "yolo11s.pt"
    device: str = "auto"
    output_dir: str = "outputs/depatch_celeb_fbi"
    iterations: int | None = None
    epochs: int = 2000
    batch_size: int = 64
    lr: float = 0.03
    patch_size: int = 300
    bbox_patch_scale: float = 0.15
    placement_mode: str = "object"
    fixed_patch_size: int = 160
    bbox_jitter_center_prob: float = 0.70
    bbox_jitter_upper_prob: float = 0.22
    bbox_jitter_center_x: float = 0.12
    bbox_jitter_center_y: float = 0.10
    bbox_jitter_upper_x: float = 0.18
    bbox_jitter_upper_y: float = 0.18
    bbox_jitter_lower_x: float = 0.14
    bbox_jitter_lower_y: float = 0.12
    max_boxes_per_image: int = 14
    conf_thres: float = 0.25
    iou_match_thres: float = 0.30
    acc_iou_weight: float = 3.0
    class_id: int = 0
    nps_weight: float = 0.001
    tv_weight: float = 0.25
    printable_colors: str | None = None
    topk: int = 200
    temperature: float = 0.10
    decoupling: bool = True
    pds_min_n: int = 2
    pds_max_n: int = 6
    pds_min_r: float = 0.20
    pds_max_r: float = 0.50
    min_contrast: float = 0.8
    max_contrast: float = 1.2
    min_brightness: float = -0.1
    max_brightness: float = 0.1
    noise_factor: float = 0.10
    enable_tc: bool = False
    enable_tps: bool = True
    tps_max_warp: float = 0.08
    num_workers: int = 4
    use_separate_predict_model: bool = False
    seed: int = 7
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    use_train_as_val: bool = True
    train_eval_samples: int | None = 256
    val_eval_samples: int | None = 256
    eval_interval: int = 100
    resume_patch: str | None = None
    start_epoch: int = 1
    cleanup_interval: int = 0
    cleanup_batch_interval: int = 100
    log_interval: int = 50


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, labels_dir: str | Path | None = None):
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.labels_dir = Path(labels_dir).expanduser() if labels_dir is not None else self._infer_labels_dir()
        self.paths = sorted(
            p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")

    def _infer_labels_dir(self) -> Path | None:
        parts = self.root.parts
        if "images" in parts:
            idx = parts.index("images")
            candidate = Path(*parts[:idx], "labels", *parts[idx + 1 :])
            if candidate.exists():
                return candidate
        sibling = self.root.parent / "labels" / self.root.name
        if sibling.exists():
            return sibling
        return None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (640, 640):
                raise ValueError(f"Expected 640x640 image, got {image.size}: {path}")
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, str(path)

    def label_path_for_image(self, image_path: str | Path) -> Path | None:
        if self.labels_dir is None:
            return None
        return self.labels_dir / f"{Path(image_path).stem}.txt"


def resolve_weights(weights: str) -> str:
    candidate = Path(weights).expanduser()
    if candidate.exists():
        return str(candidate)
    if weights == "yolo11s.pt":
        local = Path("yolo11s.pt")
        if local.exists():
            return str(local)
        fallback = Path(DEFAULT_FALLBACK_WEIGHTS)
        if fallback.exists():
            return str(fallback)
    return weights


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("Requested MPS is not available; falling back to CPU.")
        return torch.device("cpu")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_printable_colors(device: torch.device) -> Tensor:
    # Coarse printable palette over RGB. A file palette can be supplied for a specific printer.
    levels = torch.linspace(0.0, 1.0, steps=6, device=device)
    colors = torch.stack(torch.meshgrid(levels, levels, levels, indexing="ij"), dim=-1)
    return colors.reshape(-1, 3)


def load_printable_colors(path: str | None, device: torch.device) -> Tensor:
    if path is None:
        return default_printable_colors(device)
    rows: list[list[float]] = []
    with open(path, "r", newline="") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p for p in raw.replace(",", " ").split() if p]
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"Printable color file is empty or invalid: {path}")
    colors = torch.tensor(rows, dtype=torch.float32, device=device)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0)


def non_printability_loss(patch: Tensor, printable_colors: Tensor) -> Tensor:
    pixels = patch.permute(1, 2, 0).reshape(-1, 3)
    distances = torch.sqrt(((pixels[:, None, :] - printable_colors[None, :, :]) ** 2).sum(dim=-1) + 1e-12)
    return distances.min(dim=1).values.mean()


def total_variation_loss(patch: Tensor) -> Tensor:
    horizontal = torch.abs(patch[:, :, 1:] - patch[:, :, :-1]).mean()
    vertical = torch.abs(patch[:, 1:, :] - patch[:, :-1, :]).mean()
    return horizontal + vertical


def save_patch_png(patch: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    patch_np = patch.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    image = Image.fromarray((patch_np * 255).round().astype(np.uint8))
    image.save(path)


def save_patch_pt(patch: Tensor, path: Path, config: PatchTrainerConfig, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "patch": patch.detach().clamp(0, 1).cpu(),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.50) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = bbox_iou_matrix(boxes[current : current + 1], boxes[rest]).reshape(-1)
        order = rest[ious <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


class DePatchTrainer:
    def __init__(self, config: PatchTrainerConfig):
        self.config = config
        seed_everything(config.seed)
        self.device = select_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weights = resolve_weights(config.weights)
        print(f"Loading YOLO weights: {self.weights}")
        print(f"Using device: {self.device}")
        self.yolo = YOLO(self.weights)
        self.model: nn.Module = self.yolo.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        if config.use_separate_predict_model:
            self.predict_yolo = YOLO(self.weights)
            self.predict_model: nn.Module = self.predict_yolo.model.to(self.device)
            self.predict_model.eval()
            for param in self.predict_model.parameters():
                param.requires_grad_(False)
        else:
            self.predict_yolo = self.yolo
            self.predict_model = self.model

        patch_size = max(16, int(config.patch_size))
        initial_patch = torch.rand(3, patch_size, patch_size, device=self.device) * 0.10 + 0.45
        initial_patch = initial_patch.clamp(1e-4, 1 - 1e-4)
        self.patch_logits = nn.Parameter(torch.logit(initial_patch))
        self.load_resume_patch(self.resolve_resume_patch(config.resume_patch))
        self.printable_colors = load_printable_colors(config.printable_colors, self.device)
        self.history: list[dict] = []
        self.best_asr = -1.0
        self.best_loss = math.inf
        self.current_epoch = 1
        self.optimizer_steps = 0

        self.train_dataset = self._make_dataset(config.train_dir, config.max_train_samples, config.train_labels_dir)
        if config.use_train_as_val:
            self.val_dataset = self.train_dataset
        else:
            self.val_dataset = self._make_dataset(config.val_dir, config.max_val_samples, config.val_labels_dir)
        self.train_eval_dataset = self._limit_dataset(self.train_dataset, config.train_eval_samples)
        self.val_eval_dataset = self._limit_dataset(self.val_dataset, config.val_eval_samples)
        self.clean_cache: dict[str, np.ndarray] = {}
        self.clean_cache_dirty = False
        self.clean_cache_path = self.output_dir / "clean_boxes_cache.json"
        self.load_clean_box_cache()
        self.load_existing_history()

    def resolve_resume_patch(self, patch_path: str | None) -> str | None:
        if patch_path is not None:
            return patch_path
        latest = self.output_dir / "latest_patch.pt"
        if latest.exists():
            return str(latest)
        return None

    def load_resume_patch(self, patch_path: str | None) -> None:
        if patch_path is None:
            return
        path = Path(patch_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Resume patch does not exist: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        patch = checkpoint["patch"] if isinstance(checkpoint, dict) and "patch" in checkpoint else checkpoint
        if not torch.is_tensor(patch):
            raise TypeError(f"Resume patch must be a tensor or checkpoint with tensor key 'patch': {path}")
        patch = patch.detach().float()
        if patch.ndim == 4 and patch.shape[0] == 1:
            patch = patch.squeeze(0)
        if patch.ndim != 3 or patch.shape[0] != 3:
            raise ValueError(f"Expected resume patch shape [3,H,W], got {tuple(patch.shape)}: {path}")
        if patch.shape[-2:] != self.patch_logits.shape[-2:]:
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=self.patch_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        patch = patch.clamp(1e-4, 1.0 - 1e-4).to(self.device)
        with torch.no_grad():
            self.patch_logits.copy_(torch.logit(patch))
        print(f"Resumed patch from: {path}")

    def load_existing_history(self) -> None:
        path = self.output_dir / "history.csv"
        if not path.exists():
            return

        loaded: list[dict] = []
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {key: self._parse_history_value(value) for key, value in row.items()}
                epoch = int(parsed.get("epoch", 0))
                if self.config.start_epoch <= 1 or epoch < self.config.start_epoch:
                    loaded.append(parsed)

        self.history = loaded
        for metrics in self.history:
            val_asr = float(metrics.get("val_asr", 0.0))
            train_loss = float(metrics.get("train_loss", math.inf))
            if val_asr > self.best_asr or (val_asr == self.best_asr and train_loss < self.best_loss):
                self.best_asr = val_asr
                self.best_loss = train_loss
        if self.history:
            print(f"Loaded {len(self.history)} history rows from: {path}")

    def next_history_step(self, key: str, requested_start: int) -> int:
        start = max(1, requested_start)
        if requested_start > 1 or not self.history:
            return start
        last_step = max(int(row.get(key, row.get("epoch", 0))) for row in self.history)
        return max(start, last_step + 1)

    def clean_box_cache_metadata(self) -> dict:
        return {
            "version": BOX_CACHE_VERSION,
            "weights": str(self.weights),
            "class_id": self.config.class_id,
            "conf_thres": self.config.conf_thres,
            "max_boxes_per_image": self.config.max_boxes_per_image,
        }

    def load_clean_box_cache(self) -> None:
        if not self.clean_cache_path.exists():
            return
        try:
            with self.clean_cache_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            tqdm.write(f"Ignoring unreadable clean box cache: {self.clean_cache_path} ({exc})")
            return

        if payload.get("metadata") != self.clean_box_cache_metadata():
            tqdm.write(f"Ignoring stale clean box cache: {self.clean_cache_path}")
            return

        entries = payload.get("boxes", {})
        if not isinstance(entries, dict):
            tqdm.write(f"Ignoring invalid clean box cache: {self.clean_cache_path}")
            return

        loaded = 0
        for image_path, boxes in entries.items():
            array = np.asarray(boxes, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 4:
                continue
            self.clean_cache[image_path] = array
            loaded += 1
        if loaded:
            tqdm.write(f"Loaded {loaded} clean box cache entries from: {self.clean_cache_path}")

    def save_clean_box_cache(self, force: bool = False) -> None:
        if not self.clean_cache_dirty and not force:
            return
        payload = {
            "metadata": self.clean_box_cache_metadata(),
            "boxes": {path: boxes.astype(float).tolist() for path, boxes in sorted(self.clean_cache.items())},
        }
        tmp_path = self.clean_cache_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        tmp_path.replace(self.clean_cache_path)
        self.clean_cache_dirty = False

    @staticmethod
    def _parse_history_value(value: str) -> int | float | str:
        if value is None:
            return value
        try:
            number = float(value)
        except ValueError:
            return value
        if number.is_integer():
            return int(number)
        return number

    @staticmethod
    def release_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (
            hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()

    @staticmethod
    def _make_dataset(root: str, limit: int | None, labels_dir: str | None = None) -> Dataset:
        dataset = ImageFolderDataset(root, labels_dir=labels_dir)
        if limit is not None:
            return Subset(dataset, range(min(limit, len(dataset))))
        return dataset

    @staticmethod
    def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
        if limit is None:
            return dataset
        return Subset(dataset, range(min(limit, len(dataset))))

    @staticmethod
    def _dataset_paths(dataset: Dataset) -> list[str] | None:
        if isinstance(dataset, Subset):
            parent_paths = DePatchTrainer._dataset_paths(dataset.dataset)
            if parent_paths is None:
                return None
            return [parent_paths[int(index)] for index in dataset.indices]
        if isinstance(dataset, ImageFolderDataset):
            return [str(path) for path in dataset.paths]
        return None

    @staticmethod
    def _dataset_label_path(dataset: Dataset, image_path: str) -> Path | None:
        current = dataset
        while isinstance(current, Subset):
            current = current.dataset
        if isinstance(current, ImageFolderDataset):
            return current.label_path_for_image(image_path)
        return None

    def read_label_boxes(self, dataset: Dataset, image_path: str) -> np.ndarray | None:
        label_path = self._dataset_label_path(dataset, image_path)
        if label_path is None or not label_path.exists():
            return None
        boxes: list[list[float]] = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id != self.config.class_id:
                continue
            x, y, w, h = map(float, parts[1:5])
            cx = x * 640.0
            cy = y * 640.0
            bw = w * 640.0
            bh = h * 640.0
            boxes.append([
                max(0.0, cx - bw / 2),
                max(0.0, cy - bh / 2),
                min(640.0, cx + bw / 2),
                min(640.0, cy + bh / 2),
            ])
        return np.asarray(boxes, dtype=np.float32)

    def get_or_detect_boxes(self, dataset: Dataset, image: Tensor, path: str) -> np.ndarray:
        if path in self.clean_cache:
            return self.clean_cache[path]
        label_boxes = self.read_label_boxes(dataset, path)
        if label_boxes is not None:
            self.clean_cache[path] = label_boxes
            self.clean_cache_dirty = True
            return label_boxes
        boxes = self.detect_person_boxes([image])[0]
        self.clean_cache[path] = boxes
        self.clean_cache_dirty = True
        return boxes

    def get_or_detect_boxes_batch(self, dataset: Dataset, images: Tensor, paths: Sequence[str]) -> list[np.ndarray]:
        results: list[np.ndarray | None] = []
        detect_indices: list[int] = []
        detect_images: list[Tensor] = []

        for index, path in enumerate(paths):
            if path in self.clean_cache:
                results.append(self.clean_cache[path])
                continue

            label_boxes = self.read_label_boxes(dataset, path)
            if label_boxes is not None:
                self.clean_cache[path] = label_boxes
                self.clean_cache_dirty = True
                results.append(label_boxes)
                continue

            results.append(None)
            detect_indices.append(index)
            detect_images.append(images[index])

        if detect_images:
            detected_boxes = self.detect_person_boxes(detect_images)
            for index, boxes in zip(detect_indices, detected_boxes):
                self.clean_cache[paths[index]] = boxes
                results[index] = boxes
            self.clean_cache_dirty = True
            self.save_clean_box_cache()

        return [boxes if boxes is not None else np.zeros((0, 4), dtype=np.float32) for boxes in results]

    @property
    def patch(self) -> Tensor:
        return torch.sigmoid(self.patch_logits)

    def pds_params(self, epoch: int | None = None) -> tuple[int, float]:
        epoch = self.current_epoch if epoch is None else epoch
        stages = max(1, self.config.pds_max_n - self.config.pds_min_n + 1)
        schedule_len = self.config.iterations if self.config.iterations is not None else self.config.epochs
        stage_len = max(1, math.ceil(schedule_len / stages))
        stage = min(stages - 1, max(0, (epoch - 1) // stage_len))
        n = self.config.pds_min_n + stage

        local_epoch = (epoch - 1) % stage_len
        progress = local_epoch / max(1, stage_len - 1)
        r = self.config.pds_min_r + progress * (self.config.pds_max_r - self.config.pds_min_r)
        return n, r

    def decoupling_mask(self, patch: Tensor) -> Tensor:
        if not self.config.decoupling:
            return torch.ones(1, patch.shape[-2], patch.shape[-1], device=patch.device, dtype=patch.dtype)

        n, ratio = self.pds_params()
        height, width = patch.shape[-2:]
        block_keep = (torch.rand(n, n, device=patch.device) >= ratio).to(patch.dtype)
        mask = F.interpolate(
            block_keep.view(1, 1, n, n),
            size=(height, width),
            mode="nearest",
        ).squeeze(0)
        shift_h = random.randrange(height)
        shift_w = random.randrange(width)
        return torch.roll(mask, shifts=(shift_h, shift_w), dims=(-2, -1))

    def transform_patch_for_training(self) -> tuple[Tensor, Tensor]:
        patch = self.patch
        if self.config.enable_tc:
            patch = torch.roll(
                patch,
                shifts=(random.randrange(patch.shape[-2]), random.randrange(patch.shape[-1])),
                dims=(-2, -1),
            )
        alpha = self.decoupling_mask(patch)
        transformed = patch * alpha

        contrast = torch.empty(1, 1, 1, device=patch.device).uniform_(
            self.config.min_contrast, self.config.max_contrast
        )
        brightness = torch.empty(1, 1, 1, device=patch.device).uniform_(
            self.config.min_brightness, self.config.max_brightness
        )
        noise = torch.empty_like(transformed).uniform_(-1.0, 1.0) * self.config.noise_factor
        transformed = (transformed * contrast + brightness + noise).clamp(0.0, 1.0)
        transformed = transformed * alpha
        if self.config.enable_tps:
            transformed, alpha = self._smooth_cloth_deform(transformed, alpha, self.config.tps_max_warp)
        return transformed, alpha

    def make_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def place_patch(self, images: Tensor, deterministic: bool = False) -> Tensor:
        batch, _, height, width = images.shape
        output = images.clone()
        base_ratio = self.config.patch_size / width

        for index in range(batch):
            if deterministic:
                scale = base_ratio
                angle = 0.0
                cx = 0.50 * width
                cy = 0.58 * height
            else:
                scale = random.uniform(0.22, 0.32)
                angle = random.uniform(-20.0, 20.0)
                cx = (0.50 + random.uniform(-0.08, 0.08)) * width
                cy = (0.58 + random.uniform(-0.08, 0.08)) * height

            target_size = max(8, int(round(scale * width)))
            patch_tensor, alpha_mask = self.transform_patch_for_training() if not deterministic else (self.patch, None)
            self._overlay_patch_at_(output, index, cx, cy, target_size, angle, patch_tensor, alpha_mask)
        return output.clamp(0.0, 1.0)

    def place_patch_on_boxes(
        self,
        images: Tensor,
        boxes_batch: list[np.ndarray],
        deterministic: bool = True,
    ) -> Tensor:
        """Attach one patch to every target bbox, scaled from bbox diagonal as in NAP."""
        batch, _, height, width = images.shape
        output = images.clone()

        for index in range(batch):
            boxes = boxes_batch[index] if index < len(boxes_batch) else np.zeros((0, 4), dtype=np.float32)
            if len(boxes) == 0:
                if not deterministic:
                    cx = (0.50 + random.uniform(-0.08, 0.08)) * width
                    cy = (0.58 + random.uniform(-0.08, 0.08)) * height
                    target_size = max(8, int(round(random.uniform(0.22, 0.32) * width)))
                    angle = random.uniform(-20.0, 20.0)
                    patch_tensor, alpha_mask = self.transform_patch_for_training()
                    self._overlay_patch_at_(output, index, cx, cy, target_size, angle, patch_tensor, alpha_mask)
                continue
            for box in boxes[: self.config.max_boxes_per_image]:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                box_w = max(1.0, x2 - x1)
                box_h = max(1.0, y2 - y1)
                target_size = max(
                    8,
                    int(round(math.sqrt((box_w * self.config.bbox_patch_scale) ** 2 + (box_h * self.config.bbox_patch_scale) ** 2))),
                )
                if deterministic:
                    target_size = min(target_size, int(box_w), int(box_h))

                if not deterministic:
                    cx, cy = self.sample_bbox_patch_center(x1, y1, x2, y2, target_size)
                    angle = random.uniform(-20.0, 20.0)
                    patch_tensor, alpha_mask = self.transform_patch_for_training()
                else:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    angle = 0.0
                    patch_tensor, alpha_mask = self.patch, None
                self._overlay_patch_at_(output, index, cx, cy, target_size, angle, patch_tensor, alpha_mask)

        return output.clamp(0.0, 1.0)

    def sample_bbox_patch_center(self, x1: float, y1: float, x2: float, y2: float, target_size: int) -> tuple[float, float]:
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        roll = random.random()
        center_prob = min(1.0, max(0.0, self.config.bbox_jitter_center_prob))
        upper_prob = min(1.0, max(0.0, self.config.bbox_jitter_upper_prob))

        if roll < center_prob:
            base_x, base_y = 0.50, 0.50
            jitter_x = self.config.bbox_jitter_center_x
            jitter_y = self.config.bbox_jitter_center_y
        elif roll < center_prob + upper_prob:
            base_x, base_y = 0.50, 0.34
            jitter_x = self.config.bbox_jitter_upper_x
            jitter_y = self.config.bbox_jitter_upper_y
        else:
            base_x, base_y = 0.50, 0.68
            jitter_x = self.config.bbox_jitter_lower_x
            jitter_y = self.config.bbox_jitter_lower_y

        cx = x1 + (base_x + random.uniform(-jitter_x, jitter_x)) * box_w
        cy = y1 + (base_y + random.uniform(-jitter_y, jitter_y)) * box_h
        half = target_size / 2.0
        return (
            min(max(cx, x1 + half), x2 - half) if box_w > target_size else (x1 + x2) / 2.0,
            min(max(cy, y1 + half), y2 - half) if box_h > target_size else (y1 + y2) / 2.0,
        )

    def place_random_image_patch(self, images: Tensor, deterministic: bool = False) -> Tensor:
        batch, _, height, width = images.shape
        output = images.clone()
        target_size = min(self.config.fixed_patch_size, height, width)
        for index in range(batch):
            if deterministic:
                cx = width / 2.0
                cy = height / 2.0
                angle = 0.0
                patch_tensor, alpha_mask = self.patch, None
            else:
                half = target_size / 2.0
                cx = random.uniform(half, width - half)
                cy = random.uniform(half, height - half)
                angle = random.uniform(-20.0, 20.0)
                patch_tensor, alpha_mask = self.transform_patch_for_training()
            self._overlay_patch_at_(output, index, cx, cy, target_size, angle, patch_tensor, alpha_mask)
        return output.clamp(0.0, 1.0)

    def _overlay_patch_at_(
        self,
        images: Tensor,
        index: int,
        cx: float,
        cy: float,
        target_size: int,
        angle_degrees: float,
        patch_tensor: Tensor | None = None,
        alpha_mask: Tensor | None = None,
    ) -> Tensor:
        _, _, height, width = images.shape
        patch = self.patch if patch_tensor is None else patch_tensor
        resized_patch = F.interpolate(
            patch.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
        )
        if alpha_mask is None:
            alpha = torch.ones(1, patch.shape[-2], patch.shape[-1], device=images.device, dtype=images.dtype)
        else:
            alpha = alpha_mask.to(device=images.device, dtype=images.dtype)
        mask = F.interpolate(alpha.unsqueeze(0), size=(target_size, target_size), mode="nearest")
        rotated_patch, rotated_mask = self._rotate_patch(resized_patch, mask, angle_degrees)

        top = int(round(cy - target_size / 2))
        left = int(round(cx - target_size / 2))
        y0, x0 = max(0, top), max(0, left)
        y1, x1 = min(height, top + target_size), min(width, left + target_size)
        if y1 <= y0 or x1 <= x0:
            return

        py0, px0 = y0 - top, x0 - left
        py1, px1 = py0 + (y1 - y0), px0 + (x1 - x0)
        patch_crop = rotated_patch[:, :, py0:py1, px0:px1].squeeze(0)
        mask_crop = rotated_mask[:, :, py0:py1, px0:px1].squeeze(0)
        region = images[index, :, y0:y1, x0:x1]
        images[index, :, y0:y1, x0:x1] = region * (1.0 - mask_crop) + patch_crop * mask_crop

    @staticmethod
    def _rotate_patch(patch: Tensor, mask: Tensor, angle_degrees: float) -> tuple[Tensor, Tensor]:
        if abs(angle_degrees) < 1e-6:
            return patch, mask
        rotated_patch = DePatchTrainer._rotate_square_bilinear(patch, angle_degrees)
        rotated_mask = DePatchTrainer._rotate_square_bilinear(mask, angle_degrees)
        return rotated_patch, rotated_mask

    @staticmethod
    def _rotate_square_bilinear(tensor: Tensor, angle_degrees: float) -> Tensor:
        """Rotate a square tensor without grid_sample.

        PyTorch MPS currently has no backward for aten::grid_sampler_2d_backward.
        This sampler keeps the transform coordinates constant and uses gather-based
        bilinear sampling, so gradients still flow to the patch pixels on MPS.
        """
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError(f"Expected [1, C, S, S] square tensor, got {tuple(tensor.shape)}")

        _, channels, size, _ = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        center = (size - 1) / 2.0

        coords = torch.arange(size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        x = xx - center
        y = yy - center

        src_x = cos_a * x - sin_a * y + center
        src_y = sin_a * x + cos_a * y + center
        return DePatchTrainer._sample_square_bilinear(tensor, src_x, src_y)

    @staticmethod
    def _smooth_cloth_deform(patch: Tensor, alpha: Tensor, max_warp_ratio: float) -> tuple[Tensor, Tensor]:
        size = patch.shape[-1]
        device = patch.device
        dtype = patch.dtype
        coords = torch.arange(size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        max_warp = size * max_warp_ratio
        phase_x = torch.rand((), device=device, dtype=dtype) * (2 * math.pi)
        phase_y = torch.rand((), device=device, dtype=dtype) * (2 * math.pi)
        amp_x = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * max_warp
        amp_y = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * max_warp
        src_x = xx + amp_x * torch.sin((2 * math.pi * yy / max(1, size - 1)) + phase_x)
        src_y = yy + amp_y * torch.sin((2 * math.pi * xx / max(1, size - 1)) + phase_y)
        return (
            DePatchTrainer._sample_square_bilinear(patch.unsqueeze(0), src_x, src_y).squeeze(0),
            DePatchTrainer._sample_square_bilinear(alpha.unsqueeze(0), src_x, src_y).squeeze(0).clamp(0, 1),
        )

    @staticmethod
    def _sample_square_bilinear(tensor: Tensor, src_x: Tensor, src_y: Tensor) -> Tensor:
        _, channels, size, _ = tensor.shape
        dtype = tensor.dtype

        valid = (src_x >= 0) & (src_x <= size - 1) & (src_y >= 0) & (src_y <= size - 1)

        x0 = src_x.floor().clamp(0, size - 1).long()
        y0 = src_y.floor().clamp(0, size - 1).long()
        x1 = (x0 + 1).clamp(0, size - 1)
        y1 = (y0 + 1).clamp(0, size - 1)

        wx = (src_x - x0.to(dtype)).clamp(0, 1)
        wy = (src_y - y0.to(dtype)).clamp(0, 1)
        w00 = (1 - wx) * (1 - wy) * valid
        w01 = wx * (1 - wy) * valid
        w10 = (1 - wx) * wy * valid
        w11 = wx * wy * valid

        flat = tensor.reshape(channels, size * size)
        idx00 = (y0 * size + x0).reshape(-1)
        idx01 = (y0 * size + x1).reshape(-1)
        idx10 = (y1 * size + x0).reshape(-1)
        idx11 = (y1 * size + x1).reshape(-1)

        sampled = (
            flat[:, idx00] * w00.reshape(1, -1)
            + flat[:, idx01] * w01.reshape(1, -1)
            + flat[:, idx10] * w10.reshape(1, -1)
            + flat[:, idx11] * w11.reshape(1, -1)
        )
        return sampled.reshape(1, channels, size, size)

    @staticmethod
    def predictions_xyxy(predictions: Tensor) -> Tensor:
        xywh = predictions[:, :4, :].permute(0, 2, 1)
        cx, cy, width, height = xywh.unbind(dim=-1)
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

    @staticmethod
    def torch_bbox_iou_matrix(a: Tensor, b: Tensor) -> Tensor:
        if a.numel() == 0 or b.numel() == 0:
            return a.new_zeros((a.shape[0], b.shape[0]))
        lt = torch.maximum(a[:, None, :2], b[None, :, :2])
        rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))
        area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / union.clamp(min=1e-6)

    def suppression_loss(self, patched_images: Tensor, target_boxes: list[np.ndarray] | None = None) -> Tensor:
        output = self.model(patched_images)
        predictions = output[0] if isinstance(output, (tuple, list)) else output
        if predictions.ndim != 3:
            raise RuntimeError(f"Unexpected YOLO output shape: {tuple(predictions.shape)}")
        class_channel = 4 + self.config.class_id
        scores = predictions[:, class_channel, :]
        if target_boxes is None:
            topk = min(self.config.topk, scores.shape[-1])
            top_scores = scores.topk(topk, dim=-1).values
            return self.config.temperature * torch.logsumexp(top_scores / self.config.temperature, dim=-1).mean()

        pred_boxes = self.predictions_xyxy(predictions)
        losses: list[Tensor] = []
        for batch_index, boxes_np in enumerate(target_boxes):
            image_scores = scores[batch_index]
            if len(boxes_np) == 0:
                topk = min(self.config.topk, image_scores.shape[-1])
                top_scores = image_scores.topk(topk, dim=-1).values
                losses.append(self.config.temperature * torch.logsumexp(top_scores / self.config.temperature, dim=-1))
                continue

            gt = torch.as_tensor(boxes_np[:, :4], dtype=pred_boxes.dtype, device=pred_boxes.device)
            ious = self.torch_bbox_iou_matrix(pred_boxes[batch_index], gt)
            best_ious = ious.max(dim=1).values
            accuracy_score = self.config.acc_iou_weight * best_ious.detach() + image_scores
            selected = accuracy_score.argmax()
            losses.append(image_scores[selected])
        return torch.stack(losses).mean()

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> dict:
        total_loss = 0.0
        total_attack = 0.0
        total_nps = 0.0
        total_tv = 0.0
        batches = 0

        progress = tqdm(loader, desc=f"Training epoch {self.current_epoch}", unit="batch", leave=False)
        for images, paths in progress:
            optimizer.zero_grad(set_to_none=True)
            clean_boxes = self.get_or_detect_boxes_batch(self.train_dataset, images, paths)

            images = images.to(self.device, non_blocking=True)
            if self.config.placement_mode == "random_image":
                patched = self.place_random_image_patch(images, deterministic=False)
            else:
                patched = self.place_patch_on_boxes(images, clean_boxes, deterministic=False)
            attack_loss = self.suppression_loss(patched, clean_boxes)
            if self.config.nps_weight:
                nps = non_printability_loss(self.patch, self.printable_colors)
            else:
                nps = self.patch.new_tensor(0.0)
            tv = total_variation_loss(self.patch)
            loss = attack_loss + self.config.nps_weight * nps + self.config.tv_weight * tv

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_attack += float(attack_loss.detach().cpu())
            total_nps += float(nps.detach().cpu())
            total_tv += float(tv.detach().cpu())
            batches += 1
            self.optimizer_steps += 1
            progress.set_postfix(
                loss=total_loss / batches,
                attack=total_attack / batches,
                refresh=False,
            )
            del images, patched, attack_loss, nps, tv, loss
            if (
                self.config.cleanup_batch_interval
                and self.optimizer_steps % max(1, self.config.cleanup_batch_interval) == 0
            ):
                self.release_memory()

        return {
            "train_loss": total_loss / max(1, batches),
            "attack_loss": total_attack / max(1, batches),
            "nps_loss": total_nps / max(1, batches),
            "tv_loss": total_tv / max(1, batches),
        }

    def train_batch(self, batch: tuple[Tensor, tuple[str, ...]], optimizer: torch.optim.Optimizer) -> dict:
        images, paths = batch
        optimizer.zero_grad(set_to_none=True)
        clean_boxes = self.get_or_detect_boxes_batch(self.train_dataset, images, paths)

        images = images.to(self.device, non_blocking=True)
        if self.config.placement_mode == "random_image":
            patched = self.place_random_image_patch(images, deterministic=False)
        else:
            patched = self.place_patch_on_boxes(images, clean_boxes, deterministic=False)
        attack_loss = self.suppression_loss(patched, clean_boxes)
        if self.config.nps_weight:
            nps = non_printability_loss(self.patch, self.printable_colors)
        else:
            nps = self.patch.new_tensor(0.0)
        tv = total_variation_loss(self.patch)
        loss = attack_loss + self.config.nps_weight * nps + self.config.tv_weight * tv

        loss.backward()
        optimizer.step()
        self.optimizer_steps += 1

        metrics = {
            "train_loss": float(loss.detach().cpu()),
            "attack_loss": float(attack_loss.detach().cpu()),
            "nps_loss": float(nps.detach().cpu()),
            "tv_loss": float(tv.detach().cpu()),
        }
        del images, patched, attack_loss, nps, tv, loss
        if (
            self.config.cleanup_batch_interval
            and self.optimizer_steps % max(1, self.config.cleanup_batch_interval) == 0
        ):
            self.release_memory()
        return metrics

    def detect_person_boxes(self, images: Iterable[np.ndarray | Tensor]) -> list[np.ndarray]:
        tensors: list[Tensor] = []
        for image in images:
            if isinstance(image, Tensor):
                tensor = image.detach().to(dtype=torch.float32)
                if tensor.ndim != 3:
                    raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
            else:
                tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            tensors.append(tensor)
        if not tensors:
            return []

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            output = self.predict_model(batch)
            predictions = output[0] if isinstance(output, (tuple, list)) else output
        if predictions.ndim != 3:
            raise RuntimeError(f"Unexpected YOLO output shape: {tuple(predictions.shape)}")

        pred_boxes = self.predictions_xyxy(predictions).detach().cpu().numpy().astype(np.float32)
        pred_scores = predictions[:, 4 + self.config.class_id, :].detach().cpu().numpy().astype(np.float32)
        boxes: list[np.ndarray] = []
        for image_boxes, image_scores in zip(pred_boxes, pred_scores):
            keep = image_scores >= self.config.conf_thres
            if not np.any(keep):
                boxes.append(np.zeros((0, 4), dtype=np.float32))
                continue
            candidate_boxes = image_boxes[keep]
            candidate_scores = image_scores[keep]
            order = np.argsort(-candidate_scores)[:1000]
            candidate_boxes = candidate_boxes[order]
            candidate_scores = candidate_scores[order]
            keep_nms = nms_numpy(candidate_boxes, candidate_scores, iou_thres=0.50)
            keep_nms = keep_nms[: self.config.max_boxes_per_image]
            boxes.append(candidate_boxes[keep_nms].astype(np.float32))
        del batch, output, predictions
        return boxes

    def evaluate_asr(self, dataset: Dataset, prefix: str) -> dict:
        if len(dataset) == 0:
            return {
                f"{prefix}_asr": float("nan"),
                f"{prefix}_positive_images": 0,
                f"{prefix}_successes": 0,
            }
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        positives = 0
        successes = 0

        with torch.no_grad():
            for images, paths in tqdm(loader, desc=f"Evaluating {prefix}", unit="batch", leave=False):
                clean_boxes = self.get_or_detect_boxes_batch(dataset, images, paths)
                images_device = images.to(self.device)
                if self.config.placement_mode == "random_image":
                    patched = self.place_random_image_patch(images_device, deterministic=True).detach().cpu()
                else:
                    patched = self.place_patch_on_boxes(images_device, clean_boxes, deterministic=True).detach().cpu()
                patched_boxes = self.detect_person_boxes(patched)

                for clean, patched_detected in zip(clean_boxes, patched_boxes):
                    if len(clean) == 0:
                        continue
                    positives += 1
                    ious = bbox_iou_matrix(clean, patched_detected)
                    matched = bool((ious >= self.config.iou_match_thres).any()) if len(patched_detected) else False
                    if not matched:
                        successes += 1
                del images, images_device, patched, patched_boxes
                self.release_memory()

        asr = successes / positives if positives else 0.0
        return {
            f"{prefix}_asr": asr,
            f"{prefix}_positive_images": positives,
            f"{prefix}_successes": successes,
        }

    def validate(self) -> dict:
        return self.evaluate_asr(self.val_eval_dataset, "val")

    def cache_clean_boxes(self, dataset: Dataset, label: str) -> None:
        dataset_paths = self._dataset_paths(dataset)
        new_total = 0
        if dataset_paths is not None:
            missing_indices = []
            for index, path in enumerate(dataset_paths):
                if path in self.clean_cache:
                    continue
                label_boxes = self.read_label_boxes(dataset, path)
                if label_boxes is not None:
                    self.clean_cache[path] = label_boxes
                    self.clean_cache_dirty = True
                    new_total += 1
                    continue
                missing_indices.append(index)

            if not missing_indices:
                self.save_clean_box_cache()
                tqdm.write(f"Cached clean boxes for {label}: {new_total} new / {len(dataset_paths)} total")
                return

            dataset = Subset(dataset, missing_indices)

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        total = 0
        missing_total = 0
        for images, paths in tqdm(loader, desc=f"Caching boxes ({label})", unit="batch", leave=False):
            missing_total += sum(1 for path in paths if path not in self.clean_cache)
            self.get_or_detect_boxes_batch(dataset, images, paths)
            total += len(paths)
            del images
            self.release_memory()
        self.save_clean_box_cache()
        total_count = len(dataset_paths) if dataset_paths is not None else total
        tqdm.write(f"Cached clean boxes for {label}: {new_total + missing_total} new / {total_count} total")

    @staticmethod
    def tensor_to_uint8(image: Tensor) -> np.ndarray:
        array = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (array * 255).round().astype(np.uint8)

    def save_history(self) -> None:
        if not self.history:
            return
        path = self.output_dir / "history.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def save_latest_and_maybe_best(self, epoch: int, metrics: dict) -> None:
        patch = self.patch
        save_patch_png(patch, self.output_dir / "latest_patch.png")
        save_patch_pt(patch, self.output_dir / "latest_patch.pt", self.config, epoch, metrics)

        val_asr = float(metrics.get("val_asr", 0.0))
        train_loss = float(metrics.get("train_loss", math.inf))
        is_better = val_asr > self.best_asr or (val_asr == self.best_asr and train_loss < self.best_loss)
        if is_better:
            self.best_asr = val_asr
            self.best_loss = train_loss
            save_patch_png(patch, self.output_dir / "best_patch.png")
            save_patch_pt(patch, self.output_dir / "best_patch.pt", self.config, epoch, metrics)

    def fit(self, callback: Callable[[dict], None] | None = None) -> list[dict]:
        loader = self.make_train_loader()
        optimizer = torch.optim.Adam([self.patch_logits], lr=self.config.lr)
        self.cache_clean_boxes(self.train_dataset, "train")
        self.cache_clean_boxes(self.val_eval_dataset, "val_eval")
        if self.train_eval_dataset is not self.train_dataset:
            self.cache_clean_boxes(self.train_eval_dataset, "train_eval")

        if self.config.iterations is not None:
            iterator = iter(loader)
            start_iteration = self.next_history_step("iteration", self.config.start_epoch)
            iteration_progress = tqdm(
                range(start_iteration, self.config.iterations + 1),
                desc="Training",
                unit="iter",
            )
            for iteration in iteration_progress:
                self.current_epoch = iteration
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)

                train_metrics = self.train_batch(batch, optimizer)
                iteration_progress.set_postfix(
                    loss=train_metrics["train_loss"],
                    attack=train_metrics["attack_loss"],
                    refresh=False,
                )
                should_eval = (
                    iteration == 1
                    or iteration == self.config.iterations
                    or iteration % max(1, self.config.eval_interval) == 0
                )
                if should_eval:
                    train_asr_metrics = self.evaluate_asr(self.train_eval_dataset, "train")
                    val_metrics = self.validate()
                elif self.history:
                    train_asr_metrics = {
                        "train_asr": self.history[-1]["train_asr"],
                        "train_positive_images": self.history[-1]["train_positive_images"],
                        "train_successes": self.history[-1]["train_successes"],
                    }
                    val_metrics = {
                        "val_asr": self.history[-1]["val_asr"],
                        "val_positive_images": self.history[-1]["val_positive_images"],
                        "val_successes": self.history[-1]["val_successes"],
                    }
                else:
                    train_asr_metrics = self.evaluate_asr(self.train_eval_dataset, "train")
                    val_metrics = self.validate()

                decouple_n, decouple_r = self.pds_params(iteration)
                metrics = {"epoch": iteration, "iteration": iteration, **train_metrics, **train_asr_metrics, **val_metrics}
                metrics["decouple_n"] = decouple_n
                metrics["decouple_r"] = decouple_r
                self.history.append(metrics)
                self.save_latest_and_maybe_best(iteration, metrics)
                self.save_history()
                self.save_clean_box_cache()

                should_log = (
                    iteration == start_iteration
                    or iteration == self.config.iterations
                    or iteration % max(1, self.config.log_interval) == 0
                )
                if should_log:
                    tqdm.write(
                        f"iter={iteration:05d} train_loss={metrics['train_loss']:.5f} "
                        f"attack_loss={metrics['attack_loss']:.5f} nps_loss={metrics['nps_loss']:.5f} "
                        f"tv_loss={metrics['tv_loss']:.5f} decouple=({decouple_n},{decouple_r:.2f}) "
                        f"train_asr={metrics['train_asr']:.4f} "
                        f"val_asr={metrics['val_asr']:.4f} "
                        f"({metrics['val_successes']}/{metrics['val_positive_images']})"
                    )
                if callback is not None:
                    callback(metrics)
                if self.config.cleanup_interval and iteration % max(1, self.config.cleanup_interval) == 0:
                    self.release_memory()

            return self.history

        start_epoch = self.next_history_step("epoch", self.config.start_epoch)
        epoch_progress = tqdm(
            range(start_epoch, self.config.epochs + 1),
            desc="Training",
            unit="epoch",
        )
        for epoch in epoch_progress:
            self.current_epoch = epoch
            train_metrics = self.train_epoch(loader, optimizer)
            epoch_progress.set_postfix(
                loss=train_metrics["train_loss"],
                attack=train_metrics["attack_loss"],
                refresh=False,
            )
            should_eval = epoch == 1 or epoch == self.config.epochs or epoch % max(1, self.config.eval_interval) == 0
            if should_eval:
                train_asr_metrics = self.evaluate_asr(self.train_eval_dataset, "train")
                val_metrics = self.validate()
            elif self.history:
                train_asr_metrics = {
                    "train_asr": self.history[-1]["train_asr"],
                    "train_positive_images": self.history[-1]["train_positive_images"],
                    "train_successes": self.history[-1]["train_successes"],
                }
                val_metrics = {
                    "val_asr": self.history[-1]["val_asr"],
                    "val_positive_images": self.history[-1]["val_positive_images"],
                    "val_successes": self.history[-1]["val_successes"],
                }
            else:
                train_asr_metrics = self.evaluate_asr(self.train_eval_dataset, "train")
                val_metrics = self.validate()
            decouple_n, decouple_r = self.pds_params(epoch)
            metrics = {"epoch": epoch, **train_metrics, **train_asr_metrics, **val_metrics}
            metrics["decouple_n"] = decouple_n
            metrics["decouple_r"] = decouple_r
            self.history.append(metrics)
            self.save_latest_and_maybe_best(epoch, metrics)
            self.save_history()
            self.save_clean_box_cache()
            should_log = (
                epoch == start_epoch
                or epoch == self.config.epochs
                or epoch % max(1, self.config.log_interval) == 0
            )
            if should_log:
                tqdm.write(
                    f"epoch={epoch:03d} train_loss={metrics['train_loss']:.5f} "
                    f"attack_loss={metrics['attack_loss']:.5f} nps_loss={metrics['nps_loss']:.5f} "
                    f"tv_loss={metrics['tv_loss']:.5f} decouple=({decouple_n},{decouple_r:.2f}) "
                    f"train_asr={metrics['train_asr']:.4f} "
                    f"val_asr={metrics['val_asr']:.4f} "
                    f"({metrics['val_successes']}/{metrics['val_positive_images']})"
                )
            if callback is not None:
                callback(metrics)
            if self.config.cleanup_interval and epoch % max(1, self.config.cleanup_interval) == 0:
                self.release_memory()

        return self.history


def parse_args() -> PatchTrainerConfig:
    parser = argparse.ArgumentParser(description="Train a DePatch-style adversarial patch for YOLO11s.")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    parser.add_argument("--train-labels-dir", default=None)
    parser.add_argument("--val-labels-dir", default=None)
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", default="outputs/depatch_celeb_fbi")
    parser.add_argument("--iterations", type=int, default=None, help="Optional optimizer-step limit. Omit for epoch-based training.")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--patch-size", type=int, default=300)
    parser.add_argument("--bbox-patch-scale", type=float, default=0.15)
    parser.add_argument("--placement-mode", default="object", choices=["object", "random_image"])
    parser.add_argument("--fixed-patch-size", type=int, default=160)
    parser.add_argument("--bbox-jitter-center-prob", type=float, default=0.70)
    parser.add_argument("--bbox-jitter-upper-prob", type=float, default=0.22)
    parser.add_argument("--bbox-jitter-center-x", type=float, default=0.12)
    parser.add_argument("--bbox-jitter-center-y", type=float, default=0.10)
    parser.add_argument("--bbox-jitter-upper-x", type=float, default=0.18)
    parser.add_argument("--bbox-jitter-upper-y", type=float, default=0.18)
    parser.add_argument("--bbox-jitter-lower-x", type=float, default=0.14)
    parser.add_argument("--bbox-jitter-lower-y", type=float, default=0.12)
    parser.add_argument("--max-boxes-per-image", type=int, default=14)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-match-thres", type=float, default=0.30)
    parser.add_argument("--acc-iou-weight", type=float, default=3.0)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--nps-weight", type=float, default=0.001)
    parser.add_argument("--tv-weight", type=float, default=0.25)
    parser.add_argument("--printable-colors", default=None)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--no-decoupling", action="store_false", dest="decoupling")
    parser.add_argument("--pds-min-n", type=int, default=2)
    parser.add_argument("--pds-max-n", type=int, default=6)
    parser.add_argument("--pds-min-r", type=float, default=0.20)
    parser.add_argument("--pds-max-r", type=float, default=0.50)
    parser.add_argument("--min-contrast", type=float, default=0.8)
    parser.add_argument("--max-contrast", type=float, default=1.2)
    parser.add_argument("--min-brightness", type=float, default=-0.1)
    parser.add_argument("--max-brightness", type=float, default=0.1)
    parser.add_argument("--noise-factor", type=float, default=0.10)
    parser.add_argument("--enable-tc", action="store_true")
    parser.add_argument("--no-tps", action="store_false", dest="enable_tps")
    parser.add_argument("--tps-max-warp", type=float, default=0.08)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-separate-predict-model", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--use-train-as-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-eval-samples", type=int, default=256)
    parser.add_argument("--val-eval-samples", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--resume-patch", default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--cleanup-interval", type=int, default=0)
    parser.add_argument("--cleanup-batch-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    return PatchTrainerConfig(**vars(args))


def main() -> None:
    config = parse_args()
    trainer = DePatchTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose which YOLO detection head dominates at different scene scales."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_SCALES = (1.0, 0.7, 0.45, 0.35)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_paths(root: str | Path, limit: int, seed: int) -> list[Path]:
    paths = sorted(
        path for path in Path(root).expanduser().iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit and len(paths) > limit:
        rng = random.Random(seed)
        paths = rng.sample(paths, limit)
        paths.sort()
    return paths


def scale_scene(image: Image.Image, scale: float, size: int = 640, fill: int = 127) -> Image.Image:
    image = image.convert("RGB")
    scaled_size = max(1, round(size * scale))
    resized = image.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    offset = ((size - scaled_size) // 2, (size - scaled_size) // 2)
    canvas.paste(resized, offset)
    return canvas


def images_to_tensor(images: Iterable[Image.Image], device: torch.device) -> torch.Tensor:
    arrays = [np.asarray(image, dtype=np.float32) / 255.0 for image in images]
    batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    return batch.to(device)


def model_pre_nms_output(model: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with torch.no_grad():
        output = model(batch)
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[0], normalize_raw_heads(output[1])

    detect_head = model.model[-1] if hasattr(model, "model") else None
    if detect_head is None:
        raise RuntimeError("Could not locate YOLO detect head.")
    previous_training = detect_head.training
    try:
        detect_head.train()
        with torch.no_grad():
            raw = normalize_raw_heads(model(batch))
        detect_head.eval()
        with torch.no_grad():
            decoded = model(batch)
        if isinstance(decoded, (tuple, list)):
            decoded = decoded[0]
        return decoded, raw
    finally:
        detect_head.training = previous_training


def normalize_raw_heads(raw) -> list[torch.Tensor]:
    if isinstance(raw, dict):
        if "one2one" in raw:
            raw = raw["one2one"]
        elif "one2many" in raw:
            raw = raw["one2many"]
        else:
            raw = next(iter(raw.values()))
    if isinstance(raw, tuple):
        raw = list(raw)
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected raw YOLO heads as a list, got {type(raw).__name__}")
    return raw


def head_candidate_ranges(raw_heads: list[torch.Tensor]) -> list[tuple[int, int]]:
    ranges = []
    offset = 0
    for head in raw_heads:
        if head.ndim == 4:
            count = int(head.shape[-2] * head.shape[-1])
        elif head.ndim == 3:
            count = int(head.shape[-1])
        else:
            raise RuntimeError(f"Unexpected raw head shape: {tuple(head.shape)}")
        ranges.append((offset, offset + count))
        offset += count
    return ranges


def candidate_heads(raw_heads: list[torch.Tensor], total_candidates: int, device: torch.device) -> torch.Tensor:
    ranges = head_candidate_ranges(raw_heads)
    if ranges and ranges[-1][1] != total_candidates:
        raise RuntimeError(
            f"Raw head candidate count ({ranges[-1][1]}) does not match decoded output ({total_candidates})."
        )
    heads = torch.empty(total_candidates, dtype=torch.long, device=device)
    for head_index, (start, end) in enumerate(ranges):
        heads[start:end] = head_index
    return heads


def dominant_decoded_heads(decoded: torch.Tensor, raw_heads: list[torch.Tensor], class_id: int, conf_thres: float) -> tuple[np.ndarray, np.ndarray]:
    if decoded.ndim != 3:
        raise RuntimeError(f"Unexpected decoded prediction shape: {tuple(decoded.shape)}")
    scores = decoded[:, 4 + class_id, :]
    best_scores, best_indices = scores.max(dim=1)
    heads_by_candidate = candidate_heads(raw_heads, decoded.shape[-1], decoded.device)
    best_heads = heads_by_candidate[best_indices]
    best_heads = torch.where(best_scores >= conf_thres, best_heads, torch.full_like(best_heads, -1))
    return best_heads.detach().cpu().numpy(), best_scores.detach().cpu().numpy()


def head_labels(model: torch.nn.Module, count: int) -> list[str]:
    stride = getattr(model, "stride", None)
    if stride is not None:
        values = [int(v) for v in stride.detach().cpu().tolist()]
        return [f"P{i + 3} stride {values[i]}" if i < len(values) else f"head {i}" for i in range(count)]
    return [f"head {i}" for i in range(count)]


def best_heads_for_images(
    model: torch.nn.Module,
    images: list[Image.Image],
    scales: tuple[float, ...],
    device: torch.device,
    class_id: int,
    conf_thres: float,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    batch_images = [scale_scene(image, scale) for scale in scales for image in images]
    batch = images_to_tensor(batch_images, device)
    decoded, raw_heads = model_pre_nms_output(model, batch)
    heads_np, scores_np = dominant_decoded_heads(decoded, raw_heads, class_id=class_id, conf_thres=conf_thres)

    result = {}
    offset = 0
    for scale in scales:
        result[scale] = (
            heads_np[offset : offset + len(images)],
            scores_np[offset : offset + len(images)],
        )
        offset += len(images)
    return result


def compute_statistics(
    model: torch.nn.Module,
    paths: list[Path],
    scales: tuple[float, ...],
    device: torch.device,
    class_id: int,
    conf_thres: float,
    batch_size: int,
) -> tuple[dict[float, Counter], list[str]]:
    head_count = None
    counters: dict[float, Counter] = {scale: Counter() for scale in scales}
    for start in tqdm(range(0, len(paths), batch_size), desc="Head statistics", unit="batch"):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        result = best_heads_for_images(model, images, scales, device, class_id, conf_thres)
        for scale, (heads, _) in result.items():
            counters[scale].update(int(head) for head in heads)
        if head_count is None:
            head_count = max(max(heads, default=-1) for heads, _ in result.values()) + 1
    labels = head_labels(model, max(0, head_count or 0))
    return counters, labels


def plot_statistics(counters: dict[float, Counter], labels: list[str], output_path: Path) -> None:
    head_keys = list(range(len(labels)))
    names = labels + ["no detection"]
    x = np.arange(len(counters))
    width = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    totals = {scale: sum(counter.values()) for scale, counter in counters.items()}
    for index, key in enumerate(head_keys + [-1]):
        values = []
        for scale, counter in counters.items():
            total = max(1, totals[scale])
            values.append(counter.get(key, 0) / total * 100.0)
        ax.bar(x + (index - (len(names) - 1) / 2) * width, values, width=width, label=names[index])
    ax.set_xticks(x)
    ax.set_xticklabels([str(scale) for scale in counters])
    ax.set_xlabel("scene scale")
    ax.set_ylabel("images, %")
    ax.set_ylim(0, 100)
    ax.set_title("Dominant decoded pre-NMS person head by scene scale")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_statistics_csv(counters: dict[float, Counter], labels: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["scale", "total", *labels, "no_detection"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scale, counter in counters.items():
            total = sum(counter.values())
            row = {"scale": scale, "total": total, "no_detection": counter.get(-1, 0)}
            for index, label in enumerate(labels):
                row[label] = counter.get(index, 0)
            writer.writerow(row)


def show_scale_grid(
    model: torch.nn.Module,
    image_path: str | Path,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    device: torch.device | str = "mps",
    class_id: int = 0,
    conf_thres: float = 0.25,
    output_path: str | Path | None = None,
) -> Path:
    device = select_device(device) if isinstance(device, str) else device
    image = Image.open(image_path).convert("RGB")
    result = best_heads_for_images(model, [image], scales, device, class_id, conf_thres)
    detected_heads = [int(heads[0]) for heads, _ in result.values() if int(heads[0]) >= 0]
    labels = head_labels(model, max(detected_heads) + 1 if detected_heads else 0)

    fig, axes = plt.subplots(1, len(scales), figsize=(4 * len(scales), 4))
    if len(scales) == 1:
        axes = [axes]
    for ax, scale in zip(axes, scales):
        shown = scale_scene(image, scale)
        head = int(result[scale][0][0])
        score = float(result[scale][1][0])
        label = "no detection" if head < 0 else labels[head]
        ax.imshow(shown)
        ax.set_title(f"scale={scale}\\n{label}, score={score:.3f}")
        ax.axis("off")
    fig.tight_layout()
    output = Path(output_path) if output_path is not None else Path("outputs/head_scale_examples") / f"{Path(image_path).stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_visual_examples(
    model: torch.nn.Module,
    paths: list[Path],
    scales: tuple[float, ...],
    device: torch.device,
    class_id: int,
    conf_thres: float,
    output_dir: Path,
    count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths[:count]:
        show_scale_grid(
            model=model,
            image_path=path,
            scales=scales,
            device=device,
            class_id=class_id,
            conf_thres=conf_thres,
            output_path=output_dir / f"{path.stem}_scales.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose YOLO head usage across whole-scene scales.")
    parser.add_argument("--dataset-dir", default="datasets/celeb_fbi_640/images")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--output-dir", default="outputs/head_scale_diagnostics")
    parser.add_argument("--device", default="mps", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--scales", default="1.0,0.7,0.45,0.35")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--visual-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = tuple(float(part) for part in args.scales.split(","))
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    paths = image_paths(args.dataset_dir, args.limit, args.seed)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.dataset_dir}")

    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()
    counters, labels = compute_statistics(
        model=model,
        paths=paths,
        scales=scales,
        device=device,
        class_id=args.class_id,
        conf_thres=args.conf_thres,
        batch_size=args.batch_size,
    )
    plot_statistics(counters, labels, output_dir / "head_usage_by_scale.png")
    save_statistics_csv(counters, labels, output_dir / "head_usage_by_scale.csv")
    save_visual_examples(
        model=model,
        paths=paths,
        scales=scales,
        device=device,
        class_id=args.class_id,
        conf_thres=args.conf_thres,
        output_dir=output_dir / "examples",
        count=args.visual_examples,
    )

    print(f"Saved statistics: {output_dir / 'head_usage_by_scale.png'}")
    print(f"Saved statistics CSV: {output_dir / 'head_usage_by_scale.csv'}")
    print(f"Saved examples: {output_dir / 'examples'}")
    for scale, counter in counters.items():
        total = max(1, sum(counter.values()))
        parts = []
        for head_index, label in enumerate(labels):
            parts.append(f"{label}: {counter.get(head_index, 0) / total * 100:.1f}%")
        parts.append(f"no detection: {counter.get(-1, 0) / total * 100:.1f}%")
        print(f"scale={scale}: " + ", ".join(parts))


if __name__ == "__main__":
    main()

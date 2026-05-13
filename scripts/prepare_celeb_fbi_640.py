#!/usr/bin/env python3
"""Download Celeb-FBI from Hugging Face and export 640x640 letterboxed images."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageOps


def letterbox(image: Image.Image, size: int, fill: tuple[int, int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized_size = (round(width * scale), round(height * scale))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Celeb-FBI and export square letterboxed images."
    )
    parser.add_argument("--dataset", default="alecccdd/celeb-fbi", help="Hugging Face dataset id.")
    parser.add_argument("--output-dir", default="datasets/celeb_fbi_640", help="Output folder.")
    parser.add_argument("--size", type=int, default=640, help="Output square side in pixels.")
    parser.add_argument("--fill", default="127,127,127", help="Padding color as R,G,B.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing images.")
    parser.add_argument("--no-metadata", action="store_true", help="Save only images, without metadata.csv.")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets.\n"
            "Install it in the IAD environment with:\n"
            "  conda run -n IAD python -m pip install -r requirements-celeb-fbi.txt"
        ) from exc

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    fill_parts = tuple(int(part) for part in args.fill.split(","))
    if len(fill_parts) != 3 or any(part < 0 or part > 255 for part in fill_parts):
        raise ValueError("--fill must be R,G,B values in 0..255")

    load_kwargs = {}
    if args.hf_token:
        load_kwargs["token"] = args.hf_token
    dataset = load_dataset(args.dataset, **load_kwargs)
    rows_written = 0

    metadata_path = output_dir / "metadata.csv"
    csv_file = None
    writer = None

    try:
        if not args.no_metadata:
            csv_file = metadata_path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "split",
                    "row_index",
                    "id",
                    "file",
                    "original_width",
                    "original_height",
                    "height",
                    "weight",
                    "gender",
                    "age",
                ],
            )
            writer.writeheader()

        for split_name, split in dataset.items():
            for row_index, row in enumerate(split):
                if args.limit is not None and rows_written >= args.limit:
                    return

                image_id = str(row["id"])
                file_name = f"{split_name}_{image_id}.jpg"
                output_path = image_dir / file_name
                image = row["image"]
                original_width, original_height = image.size

                if args.overwrite or not output_path.exists():
                    letterbox(image, args.size, fill_parts).save(
                        output_path,
                        format="JPEG",
                        quality=95,
                        optimize=True,
                    )

                if writer is not None:
                    writer.writerow(
                        {
                            "split": split_name,
                            "row_index": row_index,
                            "id": image_id,
                            "file": str(output_path.relative_to(output_dir)),
                            "original_width": original_width,
                            "original_height": original_height,
                            "height": row.get("height"),
                            "weight": row.get("weight"),
                            "gender": row.get("gender"),
                            "age": row.get("age"),
                        }
                    )
                rows_written += 1
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()

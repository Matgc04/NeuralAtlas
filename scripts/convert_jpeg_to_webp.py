#!/usr/bin/env python3
"""Convert all .jpg/.jpeg images under a directory to .webp using Pillow."""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


def convert_jpegs(root: Path, quality: int, delete_original: bool, skip_existing: bool) -> tuple[int, int, int]:
    converted = 0
    skipped = 0
    deleted = 0

    print(delete_original, skip_existing)

    for jpg_path in root.rglob("*"):
        if not jpg_path.is_file():
            continue
        if jpg_path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        webp_path = jpg_path.with_suffix(".webp")
        
        if skip_existing and webp_path.exists():
            skipped += 1
            #print(f"Skipping existing file: {webp_path}")
        else:
            with Image.open(jpg_path) as img:
                # Ensure consistent mode for WebP
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.save(webp_path, format="WEBP", quality=quality)
            converted += 1

        if delete_original:
            #print(f"Deleting original file: {jpg_path}")
            jpg_path.unlink()
            deleted += 1

    return converted, skipped, deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .jpg/.jpeg images to .webp in a directory tree.")
    parser.add_argument(
        "root",
        nargs="?",
        help="Root directory to traverse",
    )
    parser.add_argument("--quality", type=int, default=80, help="WebP quality (default: 80)")
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete .jpg/.jpeg files after successful conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .webp files (default: skip existing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    converted, skipped, deleted = convert_jpegs(
        root=root,
        quality=args.quality,
        delete_original=args.delete_original,
        skip_existing=not args.overwrite,
    )
    print(f"Converted: {converted}")
    print(f"Skipped (existing .webp): {skipped}")
    print(f"Deleted original .jpg/.jpeg: {deleted}")


if __name__ == "__main__":
    main()

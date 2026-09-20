#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["kagglehub", "Pillow"]
# ///
"""Build a nano ImageNet subset (N classes x M images) in the NeuralAtlas layout.

Pipeline:
  1. Get the ImageNet-mini source (Kaggle `ifigotin/imagenetmini-1000`, or `--src`).
  2. Sample it with BHI-Research/nano-datasets `gen_nano_imagenet.py`.
  3. Adopt the result into `<dest>/val/<class_id>/*.webp` + `<name>_structure.json`,
     where `class_id` is the canonical ImageNet-1k index (sorted-WNID order).

Defaults rebuild `imagenet-pico`: all 1000 classes of the held-out val split, capped at
3 images per class to keep the attribution run small. Pass --images 0 for the full split
(3923 images, which is what the Hugging Face mirror holds). Sampling is seeded, so
re-running yields the exact same files.

Use `--fill-from-train` with `--split val` to fill classes that have fewer than
`--images` validation images from the corresponding training class.

Examples:
  uv run scripts/download_nano_imagenet.py
  uv run scripts/download_nano_imagenet.py --src /data/imagenet-mini --layout wnid --dest /tmp/nano
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

KAGGLE_DATASET = "ifigotin/imagenetmini-1000"
NANO_DATASETS_REPO = "https://github.com/BHI-Research/nano-datasets.git"
GEN_SCRIPT = "mini-imagenet1k/gen_nano_imagenet.py"


# --------------------------------------------------------------------------- source


def download_source(cache_dir: Path) -> Path:
    """Download ImageNet-mini from Kaggle and return the extracted root."""
    kaggle_dir = Path.home() / ".kaggle"
    has_creds = (
        os.environ.get("KAGGLE_API_TOKEN")
        or os.environ.get("KAGGLE_KEY")
        or (kaggle_dir / "kaggle.json").exists()      # classic API token
        or (kaggle_dir / "access_token").exists()     # `kaggle` CLI sign-in
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")    # authenticated Kaggle notebook
    )
    if not has_creds:
        raise SystemExit(
            "Kaggle credentials not found.\n"
            "  Sign in with the Kaggle CLI, or put an API token at ~/.kaggle/kaggle.json\n"
            "  (Kaggle > Settings > Create New Token), or export KAGGLE_USERNAME / "
            "KAGGLE_API_TOKEN (or KAGGLE_KEY).\n"
            "  Alternatively pass --src with an already-extracted ImageNet-mini directory."
        )

    import kagglehub

    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_dir))
    print(f"Downloading {KAGGLE_DATASET} from Kaggle (~4 GB, cached in {cache_dir})...")
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def resolve_split_dir(root: Path, split: str) -> Path:
    """Find the `train`/`val` directory inside a possibly nested extraction root."""
    candidates = [root / split, *sorted(root.glob(f"*/{split}")), *sorted(root.glob(f"*/*/{split}"))]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"Could not find a '{split}' directory under {root}")


# ------------------------------------------------------------------- nano-datasets


def ensure_nano_datasets(cache_dir: Path) -> Path:
    """Clone BHI-Research/nano-datasets if absent and return the gen script path."""
    repo_dir = cache_dir / "nano-datasets"
    if not repo_dir.exists():
        print(f"Cloning {NANO_DATASETS_REPO} into {repo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", NANO_DATASETS_REPO, str(repo_dir)],
            check=True,
        )
    gen_script = repo_dir / GEN_SCRIPT
    if not gen_script.is_file():
        raise SystemExit(f"Expected {GEN_SCRIPT} in the cloned repo, not found at {gen_script}")
    return gen_script


def run_sampler(
    gen_script: Path, src: Path, staging: Path, classes: int, images: int, seed: int
) -> Path:
    """Run gen_nano_imagenet.py and return the directory holding the sampled WNID dirs."""
    per_class = f"{images} images" if images else "every image"
    print(f"Sampling {classes} classes x {per_class} from {src} (seed={seed})...")
    argv = [
        str(gen_script),
        "--src", str(src),
        "--dest", str(staging),
        "--classes", str(classes),
        "--images", str(images),
    ]
    # gen_nano_imagenet.py picks images with an unseeded random.sample, so run it
    # through runpy after seeding the shared RNG to keep re-runs reproducible.
    bootstrap = (
        "import random, runpy, sys;"
        f"random.seed({seed});"
        f"sys.argv = {argv!r};"
        f"runpy.run_path({str(gen_script)!r}, run_name='__main__')"
    )
    subprocess.run(
        [sys.executable, "-c", bootstrap],
        check=True,
        stdout=subprocess.DEVNULL,  # the script prints one line per copied file
    )
    # gen_nano_imagenet.py nests output under 'val' or 'train' depending on --src.
    subdir = staging / ("val" if "val" in str(src) else "train")
    return subdir if subdir.is_dir() else staging


def fill_from_train(sampled: Path, train_dir: Path, images: int, seed: int) -> int:
    """Fill short sampled classes from the matching WNID in the train split."""
    rng = random.Random(seed)
    copied = 0

    for class_dir in sorted(path for path in sampled.iterdir() if path.is_dir()):
        present = {path.name for path in class_dir.iterdir() if path.is_file()}
        missing = images - len(present)
        if missing <= 0:
            continue

        train_class = train_dir / class_dir.name
        if not train_class.is_dir():
            continue

        candidates = sorted(
            path for path in train_class.iterdir()
            if path.is_file() and path.name not in present
        )
        selected = rng.sample(candidates, min(missing, len(candidates)))
        for source in selected:
            shutil.copy2(source, class_dir / source.name)
        copied += len(selected)

    return copied


# -------------------------------------------------------------------------- adopt


def imagenet_index_map(split_dir: Path) -> dict[str, int]:
    """Map WNID -> canonical ImageNet-1k class index (sorted-WNID order)."""
    wnids = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    if len(wnids) != 1000:
        raise SystemExit(
            f"Expected 1000 WNID directories in {split_dir} to derive canonical ImageNet "
            f"indices, found {len(wnids)}. Use --layout wnid to skip the index remap."
        )
    return {wnid: index for index, wnid in enumerate(wnids)}


def write_image(source: Path, target: Path, image_format: str, quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpeg":
        shutil.copy2(source, target)
        return
    with Image.open(source) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.save(target, format="WEBP", quality=quality)


def adopt(
    sampled: Path,
    dest: Path,
    name: str,
    layout: str,
    image_format: str,
    quality: int,
    index_map: dict[str, int] | None,
) -> dict[str, list[str]]:
    """Copy/convert the sampled tree into `<dest>/val/<class_id>/` and build the structure map."""
    val_dir = dest / "val"
    if val_dir.exists():
        shutil.rmtree(val_dir)

    suffix = ".jpeg" if image_format == "jpeg" else ".webp"
    structure: dict[str, list[str]] = {}

    for class_dir in sorted(p for p in sampled.iterdir() if p.is_dir()):
        class_id = class_dir.name if layout == "wnid" else str(index_map[class_dir.name])
        filenames = []
        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            target = val_dir / class_id / (image_path.stem + suffix)
            write_image(image_path, target, image_format, quality)
            filenames.append(target.name)
        structure[class_id] = filenames

    def sort_key(key: str) -> tuple[int, object]:
        return (0, int(key)) if key.isdigit() else (1, key)

    structure = {key: structure[key] for key in sorted(structure, key=sort_key)}
    (dest / f"{name}_structure.json").write_text(json.dumps(structure, indent=4) + "\n")
    return structure


def adopt_from_structure(
    split_dirs: list[Path],
    dest: Path,
    structure: dict[str, list[str]],
    layout: str,
    quality: int,
    index_map: dict[str, int] | None,
) -> None:
    """Materialize exactly the class/filename pairs declared by a structure file."""
    source_classes = [
        {path.name: path for path in split_dir.iterdir() if path.is_dir()}
        for split_dir in split_dirs
    ]
    if layout == "neuralatlas":
        if index_map is None:
            raise ValueError("index_map is required for neuralatlas layout")
        source_by_class = {
            str(index): [classes[wnid] for classes in source_classes if wnid in classes]
            for wnid, index in index_map.items()
        }
    else:
        source_by_class = {
            class_id: [classes[class_id] for classes in source_classes if class_id in classes]
            for class_id in set().union(*(classes.keys() for classes in source_classes))
        }

    resolved: list[tuple[Path, str, str]] = []
    missing = []
    for class_id, filenames in structure.items():
        candidates = {}
        for source_dir in source_by_class.get(class_id, []):
            for path in source_dir.iterdir():
                if path.is_file():
                    candidates.setdefault(path.stem, path)
        for filename in filenames:
            source = candidates.get(Path(filename).stem)
            if source is None:
                missing.append(f"{class_id}/{filename}")
            else:
                resolved.append((source, class_id, filename))
    if missing:
        preview = ", ".join(missing[:5])
        searched = ", ".join(map(str, split_dirs))
        raise SystemExit(
            f"Structure references {len(missing)} source images not found in {searched}: "
            f"{preview}"
        )

    val_dir = dest / "val"
    staging = dest / ".val-from-structure"
    if staging.exists():
        shutil.rmtree(staging)
    for source, class_id, filename in resolved:
        target = staging / class_id / filename
        suffix = target.suffix.lower()
        image_format = "jpeg" if suffix in {".jpg", ".jpeg"} else "webp"
        write_image(source, target, image_format, quality)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    staging.replace(val_dir)


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=int, default=1000, help="Number of classes (default: 1000)")
    parser.add_argument("--images", type=int, default=3,
                        help="Max images per class, 0 for every image in the split (default: 3)")
    parser.add_argument("--split", choices=["train", "val"], default="val",
                        help="ImageNet-mini split to sample from. 'val' is held out from the "
                             "torchvision pretrained models but has only ~4 images/class; "
                             "'train' has ~35 but the models were trained on it (default: val)")
    parser.add_argument("--fill-from-train", action="store_true",
                        help="With --split val, fill classes below --images using matching "
                             "train images")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--name", default="imagenet-pico", help="Dataset name (default: imagenet-pico)")
    parser.add_argument("--dest", type=Path,
                        help="Output directory (default: interpretability-viewer/public/<name>)")
    parser.add_argument("--src", type=Path,
                        help="Existing extracted ImageNet-mini root; skips the Kaggle download")
    parser.add_argument("--from-structure", type=Path,
                        help="Materialize the exact filenames declared by this structure JSON, "
                             "searching both validation and training splits")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/nano-datasets",
                        help="Where to keep the Kaggle download and the nano-datasets clone")
    parser.add_argument("--layout", choices=["neuralatlas", "wnid"], default="neuralatlas",
                        help="'neuralatlas' renames class dirs to ImageNet-1k indices (default); "
                             "'wnid' keeps the original nXXXXXXXX names")
    parser.add_argument("--format", dest="image_format", choices=["webp", "jpeg"], default="webp",
                        help="Output image format (default: webp)")
    parser.add_argument("--quality", type=int, default=80, help="WebP quality (default: 80)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_structure and args.fill_from_train:
        raise SystemExit("--from-structure cannot be combined with --fill-from-train")
    if args.fill_from_train and (args.split != "val" or args.images == 0):
        raise SystemExit("--fill-from-train requires --split val and --images greater than 0")

    repo_root = Path(__file__).resolve().parent.parent
    dest = args.dest or repo_root / "interpretability-viewer/public" / args.name
    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_root = args.src.expanduser().resolve() if args.src else download_source(cache_dir)
    split_dir = resolve_split_dir(source_root, args.split)
    index_map = imagenet_index_map(split_dir) if args.layout == "neuralatlas" else None

    if args.from_structure:
        other_split = "train" if args.split == "val" else "val"
        source_dirs = [split_dir, resolve_split_dir(source_root, other_split)]
        structure_path = args.from_structure.expanduser().resolve()
        try:
            structure = json.loads(structure_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot read structure {structure_path}: {error}") from error
        if not isinstance(structure, dict) or not all(
            isinstance(class_id, str)
            and bool(class_id)
            and Path(class_id).name == class_id
            and isinstance(filenames, list)
            and all(
                isinstance(filename, str)
                and bool(filename)
                and Path(filename).name == filename
                for filename in filenames
            )
            and filenames == sorted(set(filenames))
            for class_id, filenames in structure.items()
        ):
            raise SystemExit(f"Invalid structure: {structure_path}")
        image_count = sum(map(len, structure.values()))
        print(f"Materializing {image_count} declared images into {dest}...")
        adopt_from_structure(source_dirs, dest, structure, args.layout, args.quality, index_map)
        target_structure = dest / f"{args.name}_structure.json"
        if target_structure.resolve() != structure_path:
            target_structure.write_text(json.dumps(structure, indent=4) + "\n")
        print(f"Done: {len(structure)} classes, {image_count} images -> {dest}")
        return

    gen_script = ensure_nano_datasets(cache_dir)
    with tempfile.TemporaryDirectory(prefix="nano-imagenet-") as tmp:
        sampled = run_sampler(gen_script, split_dir, Path(tmp), args.classes, args.images, args.seed)
        if args.fill_from_train:
            train_dir = resolve_split_dir(source_root, "train")
            copied = fill_from_train(sampled, train_dir, args.images, args.seed)
            print(f"Filled {copied} missing validation images from train.")
        print(f"Adopting into {dest} (layout={args.layout}, format={args.image_format})...")
        structure = adopt(sampled, dest, args.name, args.layout, args.image_format, args.quality, index_map)

    sizes = [len(v) for v in structure.values()]
    print(f"Done: {len(structure)} classes, {sum(sizes)} images "
          f"({min(sizes)}-{max(sizes)} per class) -> {dest}")
    short = {k: len(v) for k, v in structure.items() if len(v) < args.images}
    if short:
        print(f"Warning: {len(short)} classes had fewer than {args.images} images: {list(short)[:10]}")


if __name__ == "__main__":
    main()

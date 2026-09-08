#!/usr/bin/env python
"""Deterministic full sweep: ensure the dataset, run the pipeline in growing steps, push
to Hugging Face, then drop the local attribution files.

Built for a remote GPU box with little disk: only one step of attribution images ever
exists locally. Everything is resumable — progress is derived from the persisted run
JSON, so re-running the script picks up exactly where it left off.

Per (model, step) with --chunk N each step covers the half-open window it adds --
[0,N), [N,2N), [2N,3N) ... up to the total -- so no sample is ever computed twice:
  1. `main.py --model M --dataset D --start-index <start> --num-samples <target>`
  2. upload `outputs/images/M__D__*.avif` + the run JSON to the HF attributions repo
  3. delete the just-uploaded images from disk

Configuration comes from `.env` (see `.env.example`):
  HF_TOKEN                 write token for the attributions repo (required)
  HF_ATTRIBUTIONS_REPO     base repo, default Matgc04/neuralatlas-attributions

Examples:
  uv run python scripts/run_sweep.py --dry-run
  uv run python scripts/run_sweep.py --chunk 100
  uv run python scripts/run_sweep.py --models resnet101 --total 200
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import config  # noqa: E402
from backend.ai_dataset.core import load_env  # noqa: E402
from backend.methods import method_catalog  # noqa: E402
from backend.persistence import OutputRepository  # noqa: E402

DEFAULT_MODELS = [
    "resnet101",
    "efficientnet_b4",
    "inception_v3",
    "mobilenet_v2",
    "convnext_tiny",
]
DEFAULT_ATTRIBUTIONS_REPO = "Matgc04/neuralatlas-attributions"
T = TypeVar("T")


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def with_retries(label: str, action: Callable[[], T], attempts: int = 4) -> T:
    """Retry a network action with exponential backoff; a multi-day run will hit blips."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = 15 * 2 ** (attempt - 1)
            log(f"warn: {label} failed ({error!r}); retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def model_repo_id(base_repo: str, model: str) -> str:
    return f"{base_repo}-{model}"


def classification_model_names() -> list[str]:
    """Return torchvision models compatible with this ImageNet classification pipeline."""
    from torchvision import models

    names = []
    for name in models.list_models():
        weights = models.get_model_weights(name).DEFAULT
        if weights is not None and len(weights.meta.get("categories", ())) == 1000:
            names.append(name)
    return names


def validate_model_names(model_names: list[str]) -> None:
    available = classification_model_names()
    invalid = [name for name in dict.fromkeys(model_names) if name not in available]
    if not invalid:
        return

    details = []
    for name in invalid:
        suggestions = difflib.get_close_matches(name, available, n=3)
        hint = f" (did you mean: {', '.join(suggestions)})" if suggestions else ""
        details.append(f"{name}{hint}")
    raise SystemExit(
        "Unsupported torchvision ImageNet classification model(s): " + "; ".join(details)
    )


def validate_method_names(method_names: list[str] | None) -> None:
    if not method_names:
        return
    available = {entry.id for entry in method_catalog()}
    invalid = sorted(set(method_names) - available)
    if invalid:
        raise SystemExit("Unsupported attribution method(s): " + ", ".join(invalid))


def ensure_model_repos(api, repo_ids: list[str]) -> None:
    for repo_id in repo_ids:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def parse_output_filename(filename: str) -> tuple[str, str, str, str, str]:
    path = Path(filename)
    parts = path.stem.split("__")
    if len(parts) != 5 or not path.suffix:
        raise ValueError(f"Invalid attribution filename: {filename}")
    model, dataset, class_id, image_id, method = parts
    return model, dataset, class_id, image_id, method


def remote_image_path(filename: str) -> str:
    _, dataset, class_id, _, _ = parse_output_filename(filename)
    return f"images/{dataset}/{class_id}/{filename}"


# ------------------------------------------------------------------------- dataset


def dataset_dir(dataset: str) -> Path:
    return REPO_ROOT / config.BASE_PUBLIC_DIR / dataset


def count_dataset_images(dataset: str) -> int:
    val_dir = dataset_dir(dataset) / "val"
    if not val_dir.is_dir():
        return 0
    return sum(1 for path in val_dir.glob("*/*") if path.is_file())


def expected_dataset_images(dataset: str) -> int | None:
    """Image count declared by `<dataset>_structure.json`, if the manifest is present."""
    structure_path = dataset_dir(dataset) / f"{dataset}_structure.json"
    if not structure_path.is_file():
        return None
    structure = json.loads(structure_path.read_text())
    return sum(len(files) for files in structure.values())


def ensure_dataset(dataset: str, dry_run: bool = False) -> int:
    """Return the number of dataset images, building them first if they are missing."""
    present = count_dataset_images(dataset)
    expected = expected_dataset_images(dataset)
    if present > 0 and (expected is None or present == expected):
        log(f"Dataset {dataset} already present: {present} images")
        return present

    if dry_run:
        log(f"Dataset {dataset} incomplete ({present} images, expected {expected}); would download")
        return expected or present

    log(f"Dataset {dataset} incomplete ({present} images, expected {expected}); downloading")
    # Run through `uv run`, not sys.executable: the download script declares its own
    # deps (kagglehub, Pillow) in a PEP-723 header and they are not project deps.
    subprocess.run(
        [
            "uv",
            "run",
            str(REPO_ROOT / "scripts/download_nano_imagenet.py"),
            "--name",
            dataset,
            "--fill-from-train",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    present = count_dataset_images(dataset)
    if present == 0:
        raise SystemExit(f"Dataset {dataset} is still empty after downloading.")
    log(f"Dataset {dataset} ready: {present} images")
    return present


# ------------------------------------------------------------------------ progress


def completed_samples(
    repository: OutputRepository,
    model: str,
    dataset: str,
    image_ext: str,
    metrics: list[str],
    methods: set[str] | None = None,
) -> int:
    """Return the length of the contiguous, fully persisted dataset prefix."""
    from backend.pipeline.atlas import dataset_keys

    return repository.first_incomplete_sample(
        model,
        dataset,
        dataset_keys(dataset_dir(dataset) / "val"),
        image_ext,
        set(metrics),
        methods,
    )


def sync_run_metadata(api, repo_id: str, model: str, dataset: str) -> int:
    """Restore this worker's run checkpoint from HF without touching global metadata."""
    paths = [
        f"runs/{model}/{dataset}/images.json",
        f"runs/{model}/{dataset}/summary.json",
    ]
    checkpoint_exists = [
        with_retries(
            f"check remote {path}",
            lambda path=path: api.file_exists(
                repo_id=repo_id,
                repo_type="dataset",
                filename=path,
            ),
        )
        for path in paths
    ]
    if not all(checkpoint_exists):
        for path in paths:
            (REPO_ROOT / config.OUTPUT_ROOT / path).unlink(missing_ok=True)
        return 0

    for path in paths:
        with_retries(
            f"download remote {path}",
            lambda path=path: api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=path,
                local_dir=str(REPO_ROOT / config.OUTPUT_ROOT),
                force_download=True,
            ),
        )
    return len(paths)


# --------------------------------------------------------------------------- steps


def run_step(model: str, dataset: str, start: int, target: int, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "main.py",
        "--model", model,
        "--dataset", dataset,
        "--start-index", str(start),
        "--num-samples", str(target),
        "--image-ext", args.image_ext,
        "--export-batch-images", str(args.export_batch_images),
        "--prune-stale-images",
        "--metrics", *args.metrics,
    ]
    if args.methods:
        command.extend(["--methods", *args.methods])
    if args.recompute:
        command.append("--recompute")
    if args.metadata_only:
        command.append("--metadata-only")
    log(f"$ {' '.join(command[1:])}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def step_image_files(model: str, dataset: str, image_ext: str) -> list[Path]:
    images_dir = REPO_ROOT / config.OUTPUT_IMAGES_DIR
    if not images_dir.is_dir():
        return []
    prefix = f"{model}__{dataset}__"
    suffix = f".{image_ext}"
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() == suffix
    )


def run_metadata_files(model: str, dataset: str) -> list[Path]:
    run_dir = REPO_ROOT / config.OUTPUT_ROOT / "runs" / model / dataset
    return [run_dir / "images.json", run_dir / "summary.json"]


def upload_step(
    api,
    repo_id: str,
    model: str,
    label: str,
    dataset: str,
    image_ext: str,
    *,
    include_images: bool = True,
) -> int:
    """Upload this worker's images and run metadata, never shared global metadata."""
    from huggingface_hub import CommitOperationAdd

    files = step_image_files(model, dataset, image_ext) if include_images else []
    operations = []
    for path in files:
        parsed_model, parsed_dataset, _, _, _ = parse_output_filename(path.name)
        if (parsed_model, parsed_dataset) != (model, dataset):
            raise ValueError(f"Attribution file does not belong to {model}/{dataset}: {path.name}")
        operations.append(
            CommitOperationAdd(path_in_repo=remote_image_path(path.name), path_or_fileobj=path)
        )

    output_root = REPO_ROOT / config.OUTPUT_ROOT
    for path in run_metadata_files(model, dataset):
        path_in_repo = path.relative_to(output_root)
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo.as_posix(),
                path_or_fileobj=path,
            )
        )

    with_retries(
        f"upload checkpoint for {label}",
        lambda: api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"checkpoint: {label}",
        ),
    )
    return len(files)


def cleanup_step(model: str, dataset: str, image_ext: str) -> int:
    files = step_image_files(model, dataset, image_ext)
    for path in files:
        path.unlink()
    return len(files)


# ---------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Torchvision models to sweep (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--dataset", default="imagenet-pico",
                        help="Dataset directory under interpretability-viewer/public (default: imagenet-pico)")
    parser.add_argument("--total", type=int,
                        help="Upper bound on samples per model (default: the whole dataset). "
                             "Samples run in class order, so a partial total only covers the "
                             "lowest class ids.")
    parser.add_argument("--chunk", type=int, default=5,
                        help="How much the --num-samples target grows per run/upload/cleanup "
                             "cycle (default: 5)")
    parser.add_argument("--image-ext", default=config.DEFAULT_IMAGE_EXT,
                        help=f"Attribution image format (default: {config.DEFAULT_IMAGE_EXT})")
    parser.add_argument("--metrics", nargs="*", default=list(config.FAITHFULNESS_METRICS),
                        choices=list(config.FAITHFULNESS_METRICS),
                        metavar="{" + ",".join(config.FAITHFULNESS_METRICS) + "}",
                        help="Faithfulness metrics to compute (default: all)")
    parser.add_argument("--methods", nargs="+",
                        help="Only run these attribution method ids (default: all)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute the selected methods from the first sample, or from "
                             "--resume-from when provided")
    parser.add_argument("--resume-from", type=int,
                        help="Explicit sample checkpoint for a recomputation; samples before "
                             "this index are skipped")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Do not render or upload images; commit run JSON after each chunk")
    parser.add_argument("--export-batch-images", type=int, default=10,
                        help="Images buffered before the run JSON is rewritten (default: 10)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip the Hugging Face upload (implies --keep-local)")
    parser.add_argument("--keep-local", action="store_true",
                        help="Do not delete the attribution images after uploading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without running anything")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk <= 0 or (args.total is not None and args.total <= 0):
        raise SystemExit("--chunk and --total must be positive.")
    if args.resume_from is not None and args.resume_from < 0:
        raise SystemExit("--resume-from must not be negative.")
    if args.resume_from is not None and not args.recompute:
        raise SystemExit("--resume-from requires --recompute.")
    args.image_ext = args.image_ext.lstrip(".").lower()
    validate_model_names(args.models)
    validate_method_names(args.methods)
    force_full_window = args.recompute or (args.metadata_only and not args.metrics)

    load_env(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    attributions_repo_base = os.getenv("HF_ATTRIBUTIONS_REPO", DEFAULT_ATTRIBUTIONS_REPO)
    model_repos = {
        model: model_repo_id(attributions_repo_base, model) for model in args.models
    }
    upload = not args.no_upload
    cleanup = upload and not args.keep_local

    api = None
    if upload and not args.dry_run:
        from huggingface_hub import HfApi

        if not token:
            raise SystemExit("HF_TOKEN is not set (put it in .env), or pass --no-upload.")
        api = HfApi(token=token)
        owner = api.whoami().get("name", "?")
        # Idempotently create missing per-model repos before any expensive compute.
        # Model names have already been checked against torchvision above.
        ensure_model_repos(api, list(model_repos.values()))
        log(f"Ensured {len(model_repos)} model repos as {owner}")
    elif upload:
        log(f"Dry run: would ensure {len(model_repos)} model repos")

    available = ensure_dataset(args.dataset, dry_run=args.dry_run)
    total = min(args.total, available) if args.total else available
    if args.resume_from is not None and args.resume_from > total:
        raise SystemExit(
            f"--resume-from ({args.resume_from}) exceeds the sweep total ({total})."
        )
    targets = [
        min(target, total)
        for target in range(args.chunk, total + args.chunk, args.chunk)
    ]
    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)

    log(
        f"Plan: {len(args.models)} models x {total} samples in {len(targets)} steps of "
        f"{args.chunk} | "
        f"metrics={args.metrics or ['none']} | upload={upload} cleanup={cleanup}"
    )
    for model in args.models:
        done = completed_samples(
            repository,
            model,
            args.dataset,
            args.image_ext,
            args.metrics,
            set(args.methods) if args.methods else None,
        )
        if args.resume_from is not None:
            log(f"  {model}: {args.resume_from}/{total} explicit recompute checkpoint")
        elif force_full_window:
            log(f"  {model}: 0/{total} samples scheduled for recompute")
        else:
            log(f"  {model}: {done}/{total} samples already complete")
    if args.dry_run:
        return

    failures: list[str] = []
    for model in args.models:
        model_repo = model_repos[model]
        # An interrupted or upload-failed run leaves images on disk that the run JSON
        # already counts as done. That local checkpoint is newer than HF, so flush it
        # before considering a remote restore. With no pending files, HF is the source
        # of truth and lets a fresh GPU box resume an existing run.
        if upload:
            pending = step_image_files(model, args.dataset, args.image_ext)
            metadata_complete = all(
                path.is_file() for path in run_metadata_files(model, args.dataset)
            )
            if args.metadata_only:
                downloaded = sync_run_metadata(api, model_repo, model, args.dataset)
                if downloaded:
                    log(f"Restored {downloaded} checkpoint files from HF for {model}")
            elif pending and metadata_complete:
                log(f"Reconciling {len(pending)} leftover files for {model}")
                upload_step(api, model_repo, model, f"{model} resume", args.dataset, args.image_ext)
                if cleanup:
                    cleanup_step(model, args.dataset, args.image_ext)
            else:
                if pending:
                    removed = cleanup_step(model, args.dataset, args.image_ext)
                    log(
                        f"Discarded {removed} orphan files for {model}: "
                        "local run metadata is incomplete"
                    )
                downloaded = sync_run_metadata(api, model_repo, model, args.dataset)
                if downloaded:
                    log(f"Restored {downloaded} checkpoint files from HF for {model}")

        done = completed_samples(
            repository,
            model,
            args.dataset,
            args.image_ext,
            args.metrics,
            set(args.methods) if args.methods else None,
        )
        if args.resume_from is not None:
            done = args.resume_from
            checkpoint_label = "explicit recompute checkpoint"
        elif force_full_window:
            done = 0
            checkpoint_label = "forced recompute start"
        else:
            checkpoint_label = "remote-confirmed checkpoint"
        log(f"Starting {model} from {checkpoint_label} {done}/{total}")

        for target in targets:
            if done >= target:
                log(f"Skipping {model} @{target}: already complete")
                continue
            # Each step covers only the samples it adds. Passing a plain prefix instead
            # would recompute every earlier sample, making the sweep quadratic.
            start = done
            label = f"{model} @{start}-{target}"
            try:
                with_retries(
                    f"step {label}",
                    lambda: run_step(model, args.dataset, start, target, args),
                    attempts=2,
                )
                if upload:
                    uploaded = upload_step(
                        api,
                        model_repo,
                        model,
                        label,
                        args.dataset,
                        args.image_ext,
                        include_images=not args.metadata_only,
                    )
                    if args.metadata_only:
                        log(f"Uploaded metadata-only checkpoint for {label}")
                    else:
                        log(f"Uploaded {uploaded} attribution files for {label}")
            except Exception as error:
                # Never clean up after a failed upload — the only copy is still local.
                log(f"error: {label} failed ({error!r}); abandoning {model}")
                failures.append(label)
                break

            done = target
            if cleanup and not args.metadata_only:
                log(f"Freed {cleanup_step(model, args.dataset, args.image_ext)} local files")
            free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
            log(f"Done {label} | {free_gb:.1f} GB free")

    if failures:
        raise SystemExit(f"Sweep finished with failures: {', '.join(failures)}")
    log("Sweep complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plan, prepare, validate and optionally upload a filename-alignment repair."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import config  # noqa: E402
from backend.hf import attributions_base_repo, model_repo_id  # noqa: E402
from backend.methods import extra_metric_keys, method_catalog  # noqa: E402
from backend.persistence import OutputRepository  # noqa: E402
from backend.pipeline.atlas import build_output_filename  # noqa: E402
from backend.records import ImageRecord  # noqa: E402

DEFAULT_MODELS = [
    "resnet101",
    "efficientnet_b4",
    "inception_v3",
    "mobilenet_v2",
    "convnext_tiny",
]


def mixed_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def records_by_sample(records: Iterable[ImageRecord]) -> dict[tuple[str, str], ImageRecord]:
    indexed: dict[tuple[str, str], ImageRecord] = {}
    for record in records:
        if record.sample_key in indexed:
            raise ValueError(f"Duplicate source sample: {record.sample_key}")
        indexed[record.sample_key] = record
    return indexed


def canonical_records(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    ordered = sorted(
        records,
        key=lambda record: (mixed_id(record.class_id), mixed_id(record.image_id)),
    )
    slots: set[tuple[str, str]] = set()
    for record in ordered:
        slot = (record.class_id, record.image_id)
        if slot in slots:
            raise ValueError(f"Duplicate canonical slot: {slot}")
        slots.add(slot)
    return ordered


def same_alignment(left: Iterable[ImageRecord], right: Iterable[ImageRecord]) -> bool:
    def signature(records: Iterable[ImageRecord]) -> list[tuple[str, str, str]]:
        return [
            (record.class_id, record.image_id, record.source_filename)
            for record in canonical_records(records)
        ]

    return signature(left) == signature(right)


def build_plan(
    repository: OutputRepository,
    dataset: str,
    models: list[str],
    canonical_model: str,
) -> dict[str, Any]:
    if canonical_model not in models:
        raise ValueError("The canonical model must be included in --models")

    runs = {model: repository.load_images(model, dataset) for model in models}
    empty = [model for model, records in runs.items() if not records]
    if empty:
        raise ValueError(f"No run records for: {', '.join(empty)}")

    canonical = canonical_records(runs[canonical_model])
    structure: dict[str, list[str]] = {}
    for record in canonical:
        structure.setdefault(record.class_id, []).append(record.source_filename)

    model_plans: dict[str, Any] = {}
    for model in models:
        indexed = records_by_sample(runs[model])
        remap = []
        recompute = []
        for target in canonical:
            existing = indexed.get(target.sample_key)
            entry = {
                "class_id": target.class_id,
                "image_id": target.image_id,
                "source_filename": target.source_filename,
            }
            if existing is None:
                recompute.append(entry)
            elif existing.image_id != target.image_id:
                remap.append({
                    **entry,
                    "from_image_id": existing.image_id,
                    "source_outputs": dict(existing.outputs),
                })
        model_plans[model] = {
            "aligned_with_canonical": same_alignment(runs[model], canonical),
            "remap": remap,
            "recompute": recompute,
        }

    return {
        "schema_version": 1,
        "dataset": dataset,
        "canonical_model": canonical_model,
        "canonical_structure": structure,
        "models": model_plans,
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read repair plan {path}: {error}") from error
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise SystemExit(f"Unsupported repair plan: {path}")
    if not isinstance(plan.get("canonical_structure"), dict) or not isinstance(
        plan.get("models"), dict
    ):
        raise SystemExit(f"Incomplete repair plan: {path}")
    return plan


def manifest_base_urls(dataset: str, models: Iterable[str]) -> dict[tuple[str, str], str]:
    manifest_path = REPO_ROOT / config.OUTPUT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    urls = {}
    for model in models:
        run = manifest.get("runs", {}).get(model, {}).get(dataset, {})
        base_url = run.get("base_url") if isinstance(run, dict) else None
        if not base_url:
            raise SystemExit(f"Manifest has no remote base_url for {model}/{dataset}")
        urls[(model, dataset)] = str(base_url)
    return urls


def download_file(url: str, target: Path, attempts: int = 4) -> None:
    for attempt in range(attempts):
        try:
            download_file_once(url, target)
            return
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def download_file_once(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    handle = os.fdopen(descriptor, "wb")
    try:
        with urllib.request.urlopen(url) as response, handle:
            shutil.copyfileobj(response, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        if not handle.closed:
            handle.close()
        Path(temporary_name).unlink(missing_ok=True)
        raise


def download_item(item: tuple[Path, str]) -> None:
    target, url = item
    download_file(url, target)


def output_filename(
    model: str,
    dataset: str,
    class_id: str,
    image_id: str,
    method: str,
    old_url: str,
) -> str:
    extension = Path(urllib.parse.urlparse(old_url).path).suffix.lstrip(".")
    if not extension:
        raise ValueError(f"Output URL has no extension: {old_url}")
    return build_output_filename(
        model, dataset, class_id, image_id, method, extension
    )


def prepare_records(
    repository: OutputRepository,
    plan: dict[str, Any],
    base_urls: dict[tuple[str, str], str],
) -> dict[str, list[ImageRecord]]:
    dataset = str(plan["dataset"])
    prepared: dict[str, list[ImageRecord]] = {}
    downloads: dict[Path, str] = {}
    for model, model_plan in plan["models"].items():
        records = records_by_sample(repository.load_images(model, dataset))
        remaps = {
            (str(item["class_id"]), str(item["source_filename"])): item
            for item in model_plan["remap"]
        }
        target_records = []
        for class_id, filenames in plan["canonical_structure"].items():
            for image_number, source_filename in enumerate(filenames):
                key = (str(class_id), str(source_filename))
                existing = records.get(key)
                if existing is None:
                    continue
                record = copy.deepcopy(existing)
                record.image_id = str(image_number)
                record.original_url = (
                    f"/{dataset}/val/{class_id}/{source_filename}"
                )
                remap = remaps.get(key)
                if remap is not None:
                    normalized_outputs = {}
                    for method, old_url in remap["source_outputs"].items():
                        old_filename = Path(
                            urllib.parse.urlparse(old_url).path
                        ).name
                        new_filename = output_filename(
                            model,
                            dataset,
                            str(class_id),
                            str(image_number),
                            method,
                            old_url,
                        )
                        remote_url = "/".join([
                            base_urls[(model, dataset)].rstrip("/"),
                            urllib.parse.quote(str(class_id), safe=""),
                            urllib.parse.quote(old_filename, safe=""),
                        ])
                        target_path = REPO_ROOT / config.OUTPUT_IMAGES_DIR / new_filename
                        previous_url = downloads.setdefault(target_path, remote_url)
                        if previous_url != remote_url:
                            raise ValueError(f"Conflicting sources for {target_path}")
                        normalized_outputs[method] = (
                            f"{config.OUTPUT_IMAGES_BASE_URL}/{new_filename}"
                        )
                    record.outputs = normalized_outputs
                target_records.append(record)
        prepared[model] = target_records
    print(f"Downloading {len(downloads)} remapped attribution artifacts...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_item, downloads.items()), total=len(downloads)))
    return prepared


def write_selections(
    plan: dict[str, Any],
    selection_dir: Path,
) -> None:
    for model, model_plan in plan["models"].items():
        atomic_write_json(
            selection_dir / f"{model}.json",
            {
                "schema_version": 1,
                "dataset": plan["dataset"],
                "samples": model_plan["recompute"],
            },
        )


def affected_models(plan: dict[str, Any]) -> list[str]:
    return [
        model
        for model, model_plan in plan["models"].items()
        if model_plan["remap"] or model_plan["recompute"]
    ]


def command_plan(args: argparse.Namespace) -> None:
    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)
    try:
        plan = build_plan(repository, args.dataset, args.models, args.canonical_model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    not_aligned = [
        model
        for model in args.confirm_aligned
        if model not in plan["models"]
        or not plan["models"][model]["aligned_with_canonical"]
    ]
    if not_aligned:
        raise SystemExit(
            "Canonical alignment was not confirmed by: " + ", ".join(not_aligned)
        )
    atomic_write_json(args.output, plan)
    write_selections(plan, args.selection_dir)

    print(f"Wrote {args.output} using {args.canonical_model} as canonical")
    for model, model_plan in plan["models"].items():
        selection_path = args.selection_dir / f"{model}.json"
        print(
            f"  {model}: {len(model_plan['remap'])} remap, "
            f"{len(model_plan['recompute'])} recompute -> {selection_path}"
        )


def command_prepare(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    dataset = str(plan["dataset"])
    affected = affected_models(plan)
    print(
        f"Prepare {dataset}: replace structure, rebuild source images, normalize "
        f"metadata for {len(affected)} affected models"
    )
    if not args.apply:
        print("Dry run only; pass --apply to perform the preparation.")
        return

    for model in affected:
        vlm_dir = REPO_ROOT / config.OUTPUT_ROOT / "runs" / model / dataset / "vlm"
        if vlm_dir.exists():
            raise SystemExit(f"VLM metadata needs an explicit migration first: {vlm_dir}")

    structure_path = (
        REPO_ROOT / config.BASE_PUBLIC_DIR / dataset / f"{dataset}_structure.json"
    )
    if args.backup_dir.exists():
        raise SystemExit(f"Backup directory already exists: {args.backup_dir}")
    backup_root = args.backup_dir.resolve()
    backup_root.mkdir(parents=True)
    shutil.copy2(structure_path, backup_root / structure_path.name)
    manifest_path = REPO_ROOT / config.OUTPUT_ROOT / "manifest.json"
    shutil.copy2(manifest_path, backup_root / "manifest.json")
    for model in plan["models"]:
        run_dir = REPO_ROOT / config.OUTPUT_ROOT / "runs" / model / dataset
        model_backup = backup_root / "runs" / model / dataset
        model_backup.mkdir(parents=True)
        for filename in ("images.json", "summary.json"):
            shutil.copy2(run_dir / filename, model_backup / filename)
    print(f"Backed up structure and run metadata to {backup_root}")

    atomic_write_json(structure_path, plan["canonical_structure"])
    command = [
        "uv",
        "run",
        str(REPO_ROOT / "scripts/download_nano_imagenet.py"),
        "--name",
        dataset,
        "--from-structure",
        str(structure_path),
    ]
    if args.src:
        command.extend(["--src", str(args.src)])
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)
    base_urls = manifest_base_urls(dataset, plan["models"])
    prepared = prepare_records(repository, plan, base_urls)
    for model, records in prepared.items():
        repository.replace_image_records(model, dataset, records)
    repository.refresh_manifest(base_urls)
    print("Preparation complete. No remote state was changed.")


def command_commands(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    write_selections(plan, args.selection_dir)
    image_ext = args.image_ext.lstrip(".").lower()
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for model, model_plan in plan["models"].items():
        if not model_plan["recompute"]:
            continue
        command = [
            args.python,
            "main.py",
            "--model",
            model,
            "--dataset",
            str(plan["dataset"]),
            "--sample-selection",
            str(args.selection_dir / f"{model}.json"),
            "--recompute",
            "--image-ext",
            image_ext,
        ]
        lines.append(" ".join(shlex.quote(part) for part in command))
    args.output.write_text("\n".join(lines) + "\n")
    args.output.chmod(args.output.stat().st_mode | 0o111)
    print(f"Wrote {args.output}; it contains {len(lines) - 3} model runs.")


def validation_errors(plan: dict[str, Any], image_ext: str) -> list[str]:
    from scripts.run_sweep import dataset_file_mismatches

    dataset = str(plan["dataset"])
    errors = []
    structure_path = (
        REPO_ROOT / config.BASE_PUBLIC_DIR / dataset / f"{dataset}_structure.json"
    )
    try:
        persisted_structure = json.loads(structure_path.read_text())
    except (OSError, json.JSONDecodeError):
        persisted_structure = None
    if persisted_structure != plan["canonical_structure"]:
        errors.append("persisted structure does not match the repair plan")
    missing_files, unexpected_files = dataset_file_mismatches(dataset)
    if missing_files or unexpected_files:
        errors.append(
            f"dataset files: {len(missing_files)} missing, "
            f"{len(unexpected_files)} unexpected"
        )
    required_methods = {entry.id for entry in method_catalog()}
    metrics = set(config.FAITHFULNESS_METRICS)
    extra_metrics = extra_metric_keys(metrics)
    canonical = {
        (str(class_id), str(index)): str(filename)
        for class_id, filenames in plan["canonical_structure"].items()
        for index, filename in enumerate(filenames)
    }
    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)
    for model, model_plan in plan["models"].items():
        records = repository.load_images(model, dataset)
        summary_path = (
            REPO_ROOT / config.OUTPUT_ROOT / "runs" / model / dataset / "summary.json"
        )
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
        if (
            summary.get("model") != model
            or summary.get("dataset") != dataset
            or summary.get("imageCount") != len(records)
        ):
            errors.append(f"{model}: stale or invalid summary")
        slots = {(record.class_id, record.image_id): record for record in records}
        if len(records) != len(canonical) or set(slots) != set(canonical):
            errors.append(
                f"{model}: expected {len(canonical)} canonical records, "
                f"got {len(records)}"
            )
            continue
        for slot, filename in canonical.items():
            record = slots[slot]
            if record.source_filename != filename:
                errors.append(f"{model}: wrong filename at {slot}")
                break
            if record.original_url is None or Path(record.original_url).name != filename:
                errors.append(f"{model}: wrong original_url at {slot}")
                break
            for method, url in record.outputs.items():
                expected = output_filename(
                    model,
                    dataset,
                    slot[0],
                    slot[1],
                    method,
                    url,
                )
                if Path(urllib.parse.urlparse(url).path).name != expected:
                    errors.append(f"{model}: non-canonical output name at {slot}/{method}")
                    break
        affected = [*model_plan["remap"], *model_plan["recompute"]]
        for entry in affected:
            slot = (str(entry["class_id"]), str(entry["image_id"]))
            record = slots[slot]
            completed = record.completed_methods(
                image_ext,
                metrics,
                extra_metrics,
            )
            if not required_methods <= completed:
                errors.append(
                    f"{model}/{slot[0]}/{slot[1]}: incomplete methods "
                    f"({len(completed)}/{len(required_methods)})"
                )
                continue
            for url in record.outputs.values():
                path = REPO_ROOT / config.OUTPUT_IMAGES_DIR / Path(url).name
                if path.suffix.lower() != f".{image_ext}" or not path.is_file():
                    errors.append(f"{model}: missing local artifact {path.name}")
    return errors


def repair_artifact_paths(
    repository: OutputRepository,
    plan: dict[str, Any],
    model: str,
) -> list[Path]:
    dataset = str(plan["dataset"])
    records = {
        (record.class_id, record.image_id): record
        for record in repository.load_images(model, dataset)
    }
    paths = set()
    model_plan = plan["models"][model]
    for entry in [*model_plan["remap"], *model_plan["recompute"]]:
        slot = (str(entry["class_id"]), str(entry["image_id"]))
        record = records[slot]
        paths.update(
            REPO_ROOT / config.OUTPUT_IMAGES_DIR / Path(url).name
            for url in record.outputs.values()
        )
    return sorted(paths)


def command_validate(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    errors = validation_errors(plan, args.image_ext.lstrip(".").lower())
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        extra = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise SystemExit(f"Repair validation failed:\n{preview}{extra}")
    print("Repair validation passed: all models and local repair artifacts are complete.")


def command_upload(args: argparse.Namespace) -> None:
    from backend.ai_dataset.core import load_env
    from backend.hf import with_retries
    from scripts.run_sweep import remote_image_path

    plan = load_plan(args.plan)
    image_ext = args.image_ext.lstrip(".").lower()
    errors = validation_errors(plan, image_ext)
    if errors:
        raise SystemExit("Refusing upload because local validation failed; run validate first.")
    affected = affected_models(plan)
    dataset = str(plan["dataset"])
    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)
    for model in affected:
        print(f"  {model}: {len(repair_artifact_paths(repository, plan, model))} artifacts")
    if not args.apply:
        print("Dry run only; pass --apply to commit these files to Hugging Face.")
        return

    load_env(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    base_repo = attributions_base_repo()
    for model in affected:
        repo_id = model_repo_id(base_repo, model)
        artifacts = repair_artifact_paths(repository, plan, model)
        operations = [
            CommitOperationAdd(
                path_in_repo=remote_image_path(path.name),
                path_or_fileobj=path,
            )
            for path in artifacts
        ]
        for filename in ("images.json", "summary.json"):
            local_path = (
                REPO_ROOT / config.OUTPUT_ROOT / "runs" / model / dataset / filename
            )
            operations.append(CommitOperationAdd(
                path_in_repo=f"runs/{model}/{dataset}/{filename}",
                path_or_fileobj=local_path,
            ))
        with_retries(
            f"upload repair for {model}",
            lambda repo_id=repo_id, operations=operations: api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"repair: align {dataset} filenames",
            ),
        )
        print(f"Uploaded {len(artifacts)} artifacts and metadata to {repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Typical workflow on the compute host:
  repair_run_alignment.py plan
  repair_run_alignment.py prepare --apply [--src /data/imagenet-mini]
  repair_run_alignment.py commands
  ./run-repair-compute.sh
  repair_run_alignment.py validate
  repair_run_alignment.py upload          # dry run
  repair_run_alignment.py upload --apply  # explicit HF mutation
""",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan", help="Audit runs and write the repair plan")
    plan_parser.add_argument("--dataset", default="imagenet-pico")
    plan_parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    plan_parser.add_argument("--canonical-model", default="efficientnet_b4")
    plan_parser.add_argument(
        "--confirm-aligned",
        nargs="*",
        default=["inception_v3", "convnext_tiny"],
        help="Models that must exactly match the canonical model before writing the plan",
    )
    plan_parser.add_argument("--output", type=Path, default=Path("repair-plan.json"))
    plan_parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("repair-selections"),
        help="Directory for generic sparse-selection JSON files consumed by main.py",
    )

    prepare_parser = commands.add_parser("prepare", help="Materialize and remap locally")
    prepare_parser.add_argument("--plan", type=Path, default=Path("repair-plan.json"))
    prepare_parser.add_argument("--src", type=Path, help="Extracted ImageNet-mini root")
    prepare_parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("repair-backup"),
    )
    prepare_parser.add_argument("--apply", action="store_true")

    commands_parser = commands.add_parser("commands", help="Write the sparse compute script")
    commands_parser.add_argument("--plan", type=Path, default=Path("repair-plan.json"))
    commands_parser.add_argument("--selection-dir", type=Path, default=Path("repair-selections"))
    commands_parser.add_argument("--output", type=Path, default=Path("run-repair-compute.sh"))
    commands_parser.add_argument("--python", default=".venv/bin/python")
    commands_parser.add_argument("--image-ext", default=config.DEFAULT_IMAGE_EXT)

    validate_parser = commands.add_parser("validate", help="Validate the completed local repair")
    validate_parser.add_argument("--plan", type=Path, default=Path("repair-plan.json"))
    validate_parser.add_argument("--image-ext", default=config.DEFAULT_IMAGE_EXT)

    upload_parser = commands.add_parser("upload", help="Upload an already validated repair")
    upload_parser.add_argument("--plan", type=Path, default=Path("repair-plan.json"))
    upload_parser.add_argument("--image-ext", default=config.DEFAULT_IMAGE_EXT)
    upload_parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handlers = {
        "plan": command_plan,
        "prepare": command_prepare,
        "commands": command_commands,
        "validate": command_validate,
        "upload": command_upload,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()

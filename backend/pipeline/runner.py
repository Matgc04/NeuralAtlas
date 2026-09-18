from __future__ import annotations

from argparse import Namespace

import torch

from backend import config
from backend.methods import build_interp_methods, method_catalog
from backend.models import build_model_runtime
from backend.persistence import ModelCatalogEntry, OutputRepository
from backend.pipeline.atlas import AtlasRunner, dataset_keys


def run_generation(args: Namespace) -> None:
    args.image_ext = args.image_ext.lstrip(".").lower()
    dataset_name = args.dataset.strip()
    if not dataset_name:
        raise SystemExit("--dataset must not be empty.")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be a positive integer.")
    start_index = args.start_index
    if start_index < 0:
        raise SystemExit("--start-index must not be negative.")
    if start_index >= args.num_samples:
        raise SystemExit(
            f"--start-index ({start_index}) must be below --num-samples "
            f"({args.num_samples}); the window [start, num-samples) would be empty."
        )
    if args.export_batch_images <= 0:
        raise SystemExit("--export-batch-images must be a positive integer.")
    selected_methods = args.methods
    known_methods = {entry.id for entry in method_catalog()}
    if selected_methods:
        unknown_methods = sorted(set(selected_methods) - known_methods)
        if unknown_methods:
            raise SystemExit(
                f"Unknown attribution method(s): {', '.join(unknown_methods)}"
            )

    dataset_dir = config.BASE_PUBLIC_DIR / dataset_name / "val"
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    torch.manual_seed(0)
    runtime = build_model_runtime(args.model)
    print(f"Using device: {runtime.device}, dtype: {runtime.dtype}")
    print(f"Model {args.model} total parameters: {runtime.parameter_count}")

    repository = OutputRepository()
    repository.write_catalogs(
        model_entries=[
            ModelCatalogEntry(
                id=args.model,
                label=args.model,
                family=args.model,
                parameter_count=runtime.parameter_count,
            )
        ],
        method_entries=method_catalog(),
    )

    if args.prune_stale_images:
        removed_json_entries, removed_files = repository.prune_stale_artifacts(
            args.model,
            dataset_name,
            args.image_ext,
        )
        print(
            "Pruned stale outputs: "
            f"{removed_files} files, {removed_json_entries} JSON entries."
            f" (model={args.model}, dataset={dataset_name}, ext={args.image_ext})"
        )

    if args.metadata_only and not args.metrics:
        interp_methods = []
    else:
        interp_methods = build_interp_methods(
            runtime.last_conv_layer,
            runtime.device,
        )
        if selected_methods:
            selected = set(selected_methods)
            interp_methods = [
                method for method in interp_methods if str(method) in selected
            ]
    if interp_methods and not args.recompute:
        complete = repository.methods_complete_for_all(
            args.model,
            dataset_name,
            dataset_keys(dataset_dir, start_index, args.num_samples),
            args.image_ext,
            set(args.metrics),
        )
        skipped = sorted(str(method) for method in interp_methods if str(method) in complete)
        interp_methods = [
            method for method in interp_methods if str(method) not in complete
        ]
        if skipped:
            print(f"Skipping {len(skipped)} methods complete for this window: {', '.join(skipped)}")

    if not interp_methods:
        print("No new methods to run; exporting model predictions only.")

    atlas = AtlasRunner(
        runtime.model,
        str(dataset_dir),
        interp_methods,
        transform=runtime.transform,
    )

    buffer = []
    for record in atlas.stream(
        num_samples=args.num_samples,
        start_index=start_index,
        output_dir=config.OUTPUT_IMAGES_DIR,
        model_name=args.model,
        dataset_name=dataset_name,
        image_ext=args.image_ext,
        metrics=set(args.metrics),
        render_images=not args.metadata_only,
    ):
        buffer.append(record)
        if len(buffer) >= args.export_batch_images:
            repository.upsert_image_records(args.model, dataset_name, buffer)
            buffer.clear()

    if buffer:
        repository.upsert_image_records(args.model, dataset_name, buffer)

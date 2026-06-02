from __future__ import annotations

from argparse import Namespace

import torch

from backend import config
from backend.methods import build_interp_methods, method_catalog, to_rgb_heatmap
from backend.models import build_model_runtime
from backend.persistence import ModelCatalogEntry, OutputRepository
from backend.pipeline.atlas import AtlasRunner


def run_generation(args: Namespace) -> None:
    args.image_ext = args.image_ext.lstrip(".").lower()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be a positive integer.")
    if args.export_batch_images <= 0:
        raise SystemExit("--export-batch-images must be a positive integer.")

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
            config.DATASET_NAME,
            args.image_ext,
        )
        print(
            "Pruned stale outputs: "
            f"{removed_files} files, {removed_json_entries} JSON entries."
            f" (model={args.model}, dataset={config.DATASET_NAME}, ext={args.image_ext})"
        )

    interp_methods = build_interp_methods(
        runtime.last_conv_layer,
        runtime.device,
        to_rgb_heatmap,
    )
    if not args.recompute:
        filtered_methods = []
        for method in interp_methods:
            method_name = str(method)
            existing_count = repository.count_method_outputs(
                args.model,
                config.DATASET_NAME,
                method_name,
                args.image_ext,
            )
            if existing_count >= args.num_samples:
                print(
                    f"Skipping {method_name}: {existing_count} outputs already exist."
                )
            else:
                filtered_methods.append(method)
        interp_methods = filtered_methods

    if not interp_methods:
        print("No new methods to run; exporting model predictions only.")

    atlas = AtlasRunner(
        runtime.model,
        str(config.BASE_PUBLIC_DIR / config.DATASET_NAME / "val"),
        interp_methods,
        transform=runtime.transform,
    )

    buffer = []
    for record in atlas.stream(
        num_samples=args.num_samples,
        output_dir=config.OUTPUT_IMAGES_DIR,
        model_name=args.model,
        dataset_name=config.DATASET_NAME,
        image_ext=args.image_ext,
        method="heat_map",
        sign="absolute_value",
        cmap="jet",
        show_colorbar=False,
    ):
        buffer.append(record)
        if len(buffer) >= args.export_batch_images:
            repository.upsert_image_records(args.model, config.DATASET_NAME, buffer)
            buffer.clear()

    if buffer:
        repository.upsert_image_records(args.model, config.DATASET_NAME, buffer)

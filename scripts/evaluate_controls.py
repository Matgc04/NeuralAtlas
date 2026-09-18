#!/usr/bin/env python
"""Evaluate model-independent controls using the thesis metrics.

Run from the repository: uv run python scripts/evaluate_controls.py
Smoke run: add --total 1 --noise-seeds 0 --output /tmp/controls-smoke.jsonl

The first JSONL line records the protocol; subsequent lines hold per-image scores.
Existing files are never overwritten. --start-index/--total select a half-open
dataset window for separate chunks. No attribution maps or catalogs are written.

Uniform is a diagnostic of the existing evaluator: min-max normalization makes
it zero, and ranking ties retain the evaluator's block-shuffling behavior.
FID results are grouped by the catalog's family, calibration and segmentation;
the header lists the methods compatible with each group. Raw pixel controls
sum to the spatial map over RGB. Segmented controls use the spatial mean within
each feature as its coefficient, repeated over RGB like Captum. These arbitrary
units are fixed, not fitted: raw FID remains scale dependent. Noise repeats are
separate observations: average them within each image before comparing methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from backend import config
from backend.methods import (
    GLOBAL_FAMILY, InsufficientFeaturesError, build_interp_methods, method_catalog,
)
from backend.metrics import KmeansConfig
from backend.models import build_model_runtime
from backend.pipeline.atlas import (
    PERTURBATION_FOR_FAMILY,
    evaluate_faithfulness,
    sample_keys,
)


def control_maps(height: int, width: int, sigma: float, seeds: list[int], index: int):
    """One spatial coefficient per pixel, independent of image content/model."""
    yield "uniform", None, torch.ones(1, 1, height, width)
    y = (torch.arange(height) - (height - 1) / 2) / height
    x = (torch.arange(width) - (width - 1) / 2) / width
    gaussian = torch.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    yield "center_gaussian", None, gaussian[None, None]
    for seed in seeds:
        # Stable across models and chunk boundaries; different for each image.
        generator = torch.Generator().manual_seed(seed + index * 1_000_003)
        yield "noise", seed, torch.rand(1, 1, height, width, generator=generator)


def fidelity_protocols() -> dict:
    protocols = {}
    for entry in method_catalog():
        key = f"{entry.family}_{'calibrated' if entry.calibrate_fidelity else 'raw'}_{entry.segmentation or 'pixel'}"
        group = protocols.setdefault(key, {
            "family": entry.family, "calibrate": entry.calibrate_fidelity,
            "segmentation": entry.segmentation, "methods": [],
        })
        group["methods"].append(entry.id)
    return protocols


def feature_coefficients(spatial: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean control value per feature, repeated over pixels as Captum does."""
    ids = mask.flatten()
    counts = torch.bincount(ids)
    sums = spatial.new_zeros(counts.numel()).scatter_add_(0, ids, spatial.flatten())
    return (sums / counts.clamp_min(1))[ids].reshape_as(spatial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[
        "resnet101", "efficientnet_b4", "inception_v3", "mobilenet_v2", "convnext_tiny",
    ])
    parser.add_argument("--dataset", default="imagenet-pico")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--total", type=int, help="Exclusive stop index; default: all images")
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--sigma", type=float, default=0.25,
                        help="Gaussian standard deviation as fraction of image size")
    parser.add_argument("--output", type=Path, default=Path("controls.jsonl"))
    args = parser.parse_args()
    if args.start_index < 0 or (args.total is not None and args.total <= args.start_index):
        parser.error("Require 0 <= start-index < total")
    if not 0 < args.sigma < float("inf"):
        parser.error("sigma must be positive and finite")
    if len(set(args.noise_seeds)) != len(args.noise_seeds) or min(args.noise_seeds) < 0:
        parser.error("noise seeds must be distinct nonnegative integers")

    dataset_path = ROOT / config.BASE_PUBLIC_DIR / args.dataset / "val"
    data = ImageFolder(dataset_path)
    keys = sample_keys(data)
    stop = min(args.total, len(data)) if args.total is not None else len(data)
    if args.start_index >= stop:
        parser.error("Selected dataset window is empty")
    fid_protocols = fidelity_protocols()
    protocol = {
        "type": "protocol", "dataset": args.dataset, "models": args.models,
        "start_index": args.start_index, "stop_index": stop,
        "noise_seeds": args.noise_seeds, "sigma_fraction": args.sigma,
        "target": "ground_truth", "fidelity_protocols": fid_protocols,
        "control_scale": "pixel: spatial value split over RGB; feature: spatial mean repeated over RGB",
        "calibrated_uniform_scale": "exact ones per RGB channel to avoid roundoff in constant patch sums",
        "uniform_policy": "unchanged evaluator: zero normalized map, native tie handling",
        "torch_seed": "dataset index", "torch_version": torch.__version__,
        "metric_config": {name: getattr(config, name) for name in vars(config)
                          if name.startswith(("FAITHFULNESS_", "FIDELITY_", "MORPH_", "METRIC_"))},
    }
    with args.output.open("x") as output:
        output.write(json.dumps(protocol) + "\n")
        output.flush()
        for model_name in args.models:
            runtime = build_model_runtime(model_name)
            runtime.model.eval()
            methods = {str(method): method for method in
                       build_interp_methods(runtime.last_conv_layer, runtime.device)}
            data.transform = runtime.transform
            print(f"{model_name}: {runtime.device}, {stop - args.start_index} images", flush=True)
            with torch.no_grad():
                for index in tqdm(range(args.start_index, stop), desc=model_name):
                    inputs = data[index][0].unsqueeze(0)
                    class_id, image_id = keys[index]
                    target = torch.tensor([int(class_id)], device=runtime.device)
                    segments = KmeansConfig().segment(inputs)
                    feature_masks, failures = {}, {}
                    for key, group in fid_protocols.items():
                        if group["segmentation"]:
                            provider = methods[group["methods"][0]].runtime_kwargs_fn
                            try:
                                feature_masks[key] = provider(inputs, target)["feature_mask"]
                            except InsufficientFeaturesError as error:
                                failures[key] = {"code": "insufficient_features",
                                                 "feature_count": error.feature_count}
                    for name, seed, spatial in control_maps(
                        *inputs.shape[-2:], args.sigma, args.noise_seeds, index,
                    ):
                        torch.manual_seed(index)
                        spatial = spatial.to(inputs)
                        attribution = spatial.expand_as(inputs) / inputs.shape[1]
                        scores = evaluate_faithfulness(
                            runtime.model, inputs, attribution, target,
                            set(config.FAITHFULNESS_METRICS) - {"fidelity"},
                            PERTURBATION_FOR_FAMILY[GLOBAL_FAMILY],
                            segments=segments,
                        )
                        fidelity = {}
                        for key, group in fid_protocols.items():
                            if key in failures:
                                continue
                            mask = feature_masks.get(key)
                            fid_attr = attribution
                            if mask is not None:
                                fid_attr = feature_coefficients(spatial, mask).expand_as(inputs)
                            elif name == "uniform" and group["calibrate"]:
                                # Calibration cancels scale; exact ones prevent spurious variance.
                                fid_attr = torch.ones_like(inputs)
                            fidelity[key] = evaluate_faithfulness(
                                runtime.model, inputs, fid_attr, target, {"fidelity"},
                                PERTURBATION_FOR_FAMILY[group["family"]],
                                feature_mask=mask, calibrate_fidelity=group["calibrate"],
                            )
                        output.write(json.dumps({
                            "model": model_name, "dataset": args.dataset,
                            "class_id": class_id, "image_id": image_id,
                            "filename": Path(data.samples[index][0]).name,
                            "control": name, "seed": seed, "metrics": scores,
                            "fidelity": fidelity, "fidelity_failures": failures,
                        }, allow_nan=False) + "\n")
                        output.flush()
            del methods, runtime
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import warnings

from backend import config

warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations")
warnings.filterwarnings(
    "ignore",
    message=(
        r"Setting forward, backward hooks and attributes on non-linear\s+"
        r"activations\.\s+The hooks and attributes will be removed\s+"
        r"after the attribution is finished"
    ),
    category=UserWarning,
    module=r"captum\.log\.dummy_log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attribution outputs.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all methods regardless of existing outputs.",
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL_NAME,
        help=f"Torchvision model name (default: {config.DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=config.DEFAULT_NUM_SAMPLES,
        help=(
            "Number of dataset samples to process in order "
            f"(default: {config.DEFAULT_NUM_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--export-batch-images",
        type=int,
        default=config.DEFAULT_EXPORT_BATCH_IMAGES,
        help=(
            "How many processed images to buffer before updating JSON metadata "
            f"(default: {config.DEFAULT_EXPORT_BATCH_IMAGES})."
        ),
    )
    parser.add_argument(
        "--image-ext",
        default=config.DEFAULT_IMAGE_EXT,
        help=f"Image file extension (default: {config.DEFAULT_IMAGE_EXT}).",
    )
    parser.add_argument(
        "--prune-stale-images",
        default=False,
        action="store_true",
        help=(
            "Delete generated image files and JSON output references for the active "
            "model/dataset that do not match --image-ext."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    #Import after parsing to avoid heavy imports prematurely
    from backend.pipeline.runner import run_generation

    run_generation(args)

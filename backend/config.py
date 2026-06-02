from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_NAME = "alexnet"
DATASET_NAME = "imagenet-pico"
BASE_PUBLIC_DIR = Path("interpretability-viewer/public")
OUTPUT_ROOT = BASE_PUBLIC_DIR / "outputs"
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_CATALOGS_DIR = OUTPUT_ROOT / "catalogs"
OUTPUT_RUNS_DIR = OUTPUT_ROOT / "runs"
LEGACY_OUTPUT_STRUCTURE_PATH = OUTPUT_ROOT / "outputs_structure.json"
DEFAULT_IMAGE_EXT = "avif"
DEFAULT_NUM_SAMPLES = 20
DEFAULT_EXPORT_BATCH_IMAGES = 5
OUTPUT_IMAGES_BASE_URL = "/outputs/images"
ATTRIBUTION_ENCODING = {
    "format": "normalized_grayscale",
    "encoded_range": "uint8_0_255",
    "sign": "absolute_value",
    "channel_reduction": "sum",
    "normalization": "cumulative_sum_threshold",
    "outlier_perc": 2.0,
    "colormap": "jet",
    "colormap_applied_by": "frontend",
}


def manifest_path() -> Path:
    return OUTPUT_ROOT / "manifest.json"


def models_catalog_path() -> Path:
    return OUTPUT_CATALOGS_DIR / "models.json"


def methods_catalog_path() -> Path:
    return OUTPUT_CATALOGS_DIR / "methods.json"


def run_dir(model: str, dataset: str) -> Path:
    return OUTPUT_RUNS_DIR / model / dataset


def run_images_path(model: str, dataset: str) -> Path:
    return run_dir(model, dataset) / "images.json"


def run_summary_path(model: str, dataset: str) -> Path:
    return run_dir(model, dataset) / "summary.json"

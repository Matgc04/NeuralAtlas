from __future__ import annotations
from pathlib import Path

DEFAULT_MODEL_NAME = "alexnet"
DATASET_NAME = "imagenet-pico-ai"
BASE_PUBLIC_DIR = Path("interpretability-viewer/public")
OUTPUT_ROOT = BASE_PUBLIC_DIR / "outputs"
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
DEFAULT_IMAGE_EXT = "avif"
DEFAULT_NUM_SAMPLES = 20
DEFAULT_EXPORT_BATCH_IMAGES = 5
FAITHFULNESS_METRICS = ("lif", "morph", "segment", "fidelity")
FAITHFULNESS_N_STEPS = 100
FAITHFULNESS_BLUR_SIGMA = None
# Gaussian blur sigma of the PeS/PdS source paper (tau=0.5, phi=1%, 100 steps).
MORPH_BLUR_SIGMA = 10.0
# Draws per image. Methods marked `calibrate_fidelity` in the catalog draw this
# many twice: one set fits the scale, the other is scored against it.
FIDELITY_N_PERTURB_SAMPLES = 25
# Local explanations use the noisy baseline of Yeh et al. (2019), global ones
# square removal. Both land under the same "fidelity" key, so the scores of the
# two families are not comparable and must not be ranked against each other.
FIDELITY_NOISE_STD = 0.2
FIDELITY_SQUARE_SIZE = 56
# Same reference point as the IntegratedGradients/DeepLift/GradientShap
# baselines in the catalog, so a removed patch means the same thing everywhere.
FIDELITY_SQUARE_BASELINE = 0.0
FIDELITY_MAX_EXAMPLES_PER_BATCH = 5
FIDELITY_RANDOM_SEED = 0
METRIC_BATCH_SIZE = 32
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

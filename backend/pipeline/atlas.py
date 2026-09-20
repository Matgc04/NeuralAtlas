from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterator

import numpy as np
import torch
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm.auto import tqdm

from attr_config import AttributionConfig
from backend import config
from backend.methods import (
    GLOBAL_FAMILY,
    LOCAL_FAMILY,
    InsufficientFeaturesError,
    method_catalog,
)
from backend.metrics.fidelity_score import (
    GaussianNoise, Perturbation, SquareRemoval, SuperpixelRemoval,
)
from backend.records import AttributionFailure, ImageRecord, MetricValue, PredictionRecord

# Yeh et al. (2019) pair each explanation family with its own perturbation
# (section 2.5); this is the only place that mapping is decided. Both are
# frozen and stateless, so one instance each is shared by every method.
PERTURBATION_FOR_FAMILY: dict[str, Perturbation] = {
    LOCAL_FAMILY: GaussianNoise(std=config.FIDELITY_NOISE_STD),
    GLOBAL_FAMILY: SquareRemoval(
        size=config.FIDELITY_SQUARE_SIZE,
        baseline=config.FIDELITY_SQUARE_BASELINE,
    ),
}


def evaluate_faithfulness(
    model: Module,
    inputs: torch.Tensor,
    attribution: torch.Tensor,
    target: torch.Tensor,
    metrics: set[str],
    perturbation: Perturbation,
    *,
    segments: torch.Tensor | None = None,
    feature_mask: torch.Tensor | None = None,
    calibrate_fidelity: bool = False,
) -> dict[str, MetricValue]:
    """Faithfulness scores for one attribution map, keyed by metric name.

    The heatmap is reduced from the attribution the same way the renderer does
    (channel sum, absolute value) and min-max normalized to [0, 1] so the
    morphology threshold is meaningful across methods with different scales.
    Fidelity instead receives the original, unreduced attribution tensor
    because its magnitude and sign are part of the underlying infidelity
    calculation, and `perturbation` is the one Yeh et al. (2019) pair with the
    method's explanation family. `feature_mask` is set only for the superpixel
    methods, whose attribution holds one coefficient per segment rather than per
    pixel; fidelity needs it to count each coefficient once.
    """
    from backend.metrics import (
        FidelityScore,
        ImportanceScore,
        KmeansConfig,
        MorphScore,
        SegmentScore,
    )

    heatmap = attribution.detach().sum(dim=1, keepdim=True).abs()  # (B, 1, H, W)
    flat = heatmap.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    heatmap = (heatmap - lo) / (hi - lo).clamp_min(1e-8)
    inputs = inputs.detach()

    n_steps = config.FAITHFULNESS_N_STEPS
    blur_sigma = config.FAITHFULNESS_BLUR_SIGMA

    scores: dict[str, MetricValue] = {}
    for mode in ("mif", "lif"):
        if mode in metrics:
            metric = ImportanceScore(model, inputs, heatmap, target, blur_sigma=blur_sigma)
            metric.update(mode=mode, n_steps=n_steps)
            scores[mode] = float(metric.compute()[0].item())
    if "morph" in metrics:
        metric = MorphScore(model, inputs, heatmap, target, blur_sigma=config.MORPH_BLUR_SIGMA)
        metric.update(mode="erode", n_steps=n_steps)
        scores["morph"] = float(metric.compute()[0].item())
    if "segment" in metrics:
        metric = SegmentScore(
            model,
            inputs,
            heatmap,
            target,
            KmeansConfig(),
            segmentation_inputs=inputs,
            segments=segments,
            mode="deletion",
            blur_sigma=blur_sigma,
        )
        metric.update()
        scores["segment"] = float(metric.compute()[0].item())
    if "fidelity" in metrics:
        metric = FidelityScore(model, inputs, attribution, target, feature_mask)
        variants = {"fidelity": perturbation}
        if feature_mask is not None:
            variants["fidelity_superpixel"] = SuperpixelRemoval(
                feature_mask, baseline=config.FIDELITY_SQUARE_BASELINE
            )
        for name, removal in variants.items():
            metric.update(
                n_perturb_samples=config.FIDELITY_N_PERTURB_SAMPLES,
                perturbation=removal,
                max_examples_per_batch=config.FIDELITY_MAX_EXAMPLES_PER_BATCH,
                random_seed=config.FIDELITY_RANDOM_SEED,
                calibrate=calibrate_fidelity,
            )
            fidelity = float(metric.compute()[0].item())
            scores[name] = fidelity if math.isfinite(fidelity) else None
    return scores


def sample_keys(data: datasets.ImageFolder) -> list[tuple[str, str]]:
    """(class_id, image_id) of every sample, in dataset order.

    `image_id` is a per-class sequence number, so the ids of a window can only be
    derived by walking the dataset from the start. ImageFolder sorts class
    directories as strings, so the target index only matches the directory name for
    small datasets ("0".."9"); read the name back instead.
    """
    counters: defaultdict[str, int] = defaultdict(int)
    keys = []
    for _, target in data.samples:
        class_id = data.classes[target]
        keys.append((class_id, str(counters[class_id])))
        counters[class_id] += 1
    return keys


def dataset_keys(
    dataset_dir: str | Path,
    start_index: int = 0,
    stop_index: int | None = None,
) -> list[tuple[str, str]]:
    """Stable (class_id, source_filename) keys, sliced to a window."""
    data = datasets.ImageFolder(str(dataset_dir))
    return [
        (data.classes[target], Path(path).name)
        for path, target in data.samples[start_index:stop_index]
    ]


def build_output_filename(
    model_name: str,
    dataset_name: str,
    class_id: str,
    image_id: str,
    method_name: str,
    image_ext: str,
) -> str:
    return f"{model_name}__{dataset_name}__{class_id}__{image_id}__{method_name}.{image_ext}"


@dataclass(slots=True)
class AttributionRenderer:
    out_px: int = 224
    outlier_perc: float = 2.0

    @staticmethod
    def _cumulative_sum_threshold(values: np.ndarray, percentile: float) -> float:
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile for thresholding must be between 0 and 100.")

        flat_values = np.sort(values.reshape(-1))
        if flat_values.size == 0:
            return 0.0

        cum_sums = np.cumsum(flat_values)
        total = float(cum_sums[-1])
        if total <= 0:
            return 0.0

        threshold_idx = int(np.searchsorted(cum_sums, total * 0.01 * percentile))
        threshold_idx = min(threshold_idx, flat_values.size - 1)
        return float(flat_values[threshold_idx])

    def _normalize_absolute_heatmap(self, attr: torch.Tensor, outlier_perc: float) -> np.ndarray:
        attr_np = attr.permute(1, 2, 0).detach().cpu().numpy()
        attr_combined = np.abs(np.sum(attr_np, axis=-1))
        attr_combined -= attr_combined.min()
        threshold = self._cumulative_sum_threshold(
            attr_combined,
            100.0 - outlier_perc,
        )
        if threshold <= 0:
            return np.zeros_like(attr_combined, dtype=np.float32)
        return np.clip(attr_combined / threshold, 0, 1)

    def render(
        self,
        attr: TensorOrTupleOfTensorsGeneric,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        class_id: str,
        image_id: str,
        method_name: str,
        image_ext: str,
        **kwargs: Any,
    ) -> str:
        if not isinstance(attr, torch.Tensor):
            raise TypeError(
                f"Expected attribution tensor for visualization, got {type(attr)}"
            )
        outlier_perc = float(kwargs.get("outlier_perc", self.outlier_perc))
        attr_np = self._normalize_absolute_heatmap(attr, outlier_perc)
        gray = (attr_np * 255).round().astype(np.uint8)
        img = Image.fromarray(gray, mode='L').resize((self.out_px, self.out_px), Image.LANCZOS)

        filename = build_output_filename(
            model_name=model_name,
            dataset_name=dataset_name,
            class_id=class_id,
            image_id=image_id,
            method_name=method_name,
            image_ext=image_ext,
        )
        save_kwargs: dict[str, Any] = {"format": image_ext.upper()}
        if image_ext == "webp":
            save_kwargs.update({"quality": 85, "method": 6, "lossless": False})
        elif image_ext in {"jpg", "jpeg"}:
            save_kwargs.update({"quality": 85, "optimize": True, "progressive": True})
        elif image_ext == "avif":
            save_kwargs.update({"quality": 75, "speed": 6})
        img.save(output_dir / filename, **save_kwargs)
        return f"{config.OUTPUT_IMAGES_BASE_URL}/{filename}"


class AtlasRunner:
    def __init__(
        self,
        model: Module,
        data: str | datasets.VisionDataset,
        interp_methods: list[AttributionConfig],
        renderer: AttributionRenderer | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        if isinstance(data, str):
            self.data = datasets.ImageFolder(data, transform=kwargs.get("transform", None))
        else:
            self.data = data
        self.interp_methods = interp_methods
        self.renderer = renderer or AttributionRenderer()

    @staticmethod
    def _serialize_prediction(model_output: object) -> PredictionRecord:
        if isinstance(model_output, torch.Tensor):
            logits = model_output
        elif (
            isinstance(model_output, (list, tuple))
            and len(model_output) > 0
            and isinstance(model_output[0], torch.Tensor)
        ):
            logits = model_output[0]
        else:
            raise TypeError(
                "Expected model output tensor (or tuple/list with tensor first item), "
                f"got {type(model_output)}."
            )
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        sample_logits = logits[0].detach().cpu()
        predicted_class_id = int(torch.argmax(sample_logits).item())
        confidence = float(torch.softmax(sample_logits, dim=0)[predicted_class_id].item())
        return PredictionRecord(
            predicted_class_id=str(predicted_class_id),
            confidence=confidence,
        )

    def _resolve_original_url(
        self,
        dataset_name: str,
        class_id: str,
        sample_index: int,
    ) -> str | None:
        sample_path: str | None = None
        if hasattr(self.data, "samples"):
            samples = getattr(self.data, "samples")
            if isinstance(samples, list) and sample_index < len(samples):
                sample = samples[sample_index]
                if isinstance(sample, tuple) and sample:
                    sample_path = str(sample[0])
        if sample_path is None:
            return None
        filename = Path(sample_path).name
        return f"/{dataset_name}/val/{class_id}/{filename}"

    def stream(
        self,
        num_samples: int,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        image_ext: str = "webp",
        metrics: set[str] | None = None,
        start_index: int = 0,
        sample_indices: Sequence[int] | None = None,
        render_images: bool = True,
        **kwargs: Any,
    ) -> Iterator[ImageRecord]:

        if sample_indices is None:
            if start_index < 0:
                raise ValueError(f"start_index must be non-negative, got {start_index}")
            indices = list(range(start_index, min(num_samples, len(self.data))))
        else:
            indices = list(sample_indices)
            if len(indices) != len(set(indices)):
                raise ValueError("sample_indices must not contain duplicates")
            invalid = [index for index in indices if index < 0 or index >= len(self.data)]
            if invalid:
                raise ValueError(f"sample index out of range: {invalid[0]}")
        if not indices:
            return

        self.model.eval()
        # Subset instead of skipping inside the loop: samples before the window are
        # never decoded, so a late chunk costs the same as an early one.
        dataloader = DataLoader(Subset(self.data, indices), batch_size=1, shuffle=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_ext = image_ext.lstrip(".").lower()

        keys = sample_keys(self.data)
        entries = {entry.id: entry for entry in method_catalog()}
        uncatalogued = sorted(
            name for name in map(str, self.interp_methods) if name not in entries
        )
        if metrics and uncatalogued:
            raise ValueError(
                "No explanation family for: "
                + ", ".join(uncatalogued)
                + ". Add the method to `method_catalog()`; the family selects which "
                "infidelity perturbation the metric samples."
            )

        with tqdm(total=len(indices), desc="Interpreting + Saving") as pbar:
            for offset, (inputs, target) in enumerate(dataloader):
                sample_index = indices[offset]

                class_id, image_id = keys[sample_index]
                attribution_target = torch.tensor(
                    [int(class_id)],
                    device=inputs.device,
                    dtype=target.dtype,
                )
                original_url = self._resolve_original_url(dataset_name, class_id, sample_index)

                with torch.no_grad():
                    prediction = self._serialize_prediction(self.model(inputs))

                segments = None
                if self.interp_methods and metrics and "segment" in metrics:
                    from backend.metrics import KmeansConfig

                    with torch.no_grad():
                        segments = KmeansConfig().segment(inputs)

                record = ImageRecord(
                    model=model_name,
                    dataset=dataset_name,
                    class_id=class_id,
                    image_id=image_id,
                    source_filename=Path(self.data.samples[sample_index][0]).name,
                    original_url=original_url,
                    prediction=prediction,
                    interpretability_metrics={},
                )

                inputs.requires_grad = True
                with tqdm(self.interp_methods, leave=False) as method_pbar:
                    for interp_method in method_pbar:
                        method_name = str(interp_method)
                        method_pbar.set_description(f"Attribution {method_name}")
                        try:
                            attribution = interp_method.attribute(
                                self.model,
                                inputs,
                                attribution_target,
                            )
                        except InsufficientFeaturesError as error:
                            record.attribution_failures[method_name] = AttributionFailure(
                                code="insufficient_features",
                                feature_count=error.feature_count,
                            )
                            method_pbar.write(
                                f"Skipping {method_name} for sample {sample_index}: {error}"
                            )
                            continue
                        if not isinstance(attribution, torch.Tensor):
                            raise TypeError(
                                "Streaming visualization requires tensor attribution output; "
                                f"got {type(attribution)} from {method_name}."
                            )
                        if render_images:
                            output_url = self.renderer.render(
                                attr=attribution[0],
                                output_dir=output_dir,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                class_id=class_id,
                                image_id=image_id,
                                method_name=method_name,
                                image_ext=image_ext,
                                **kwargs,
                            )
                            record.outputs[method_name] = output_url

                        if metrics:
                            record.interpretability_metrics[method_name] = (
                                evaluate_faithfulness(
                                    self.model,
                                    inputs,
                                    attribution,
                                    attribution_target,
                                    metrics,
                                    PERTURBATION_FOR_FAMILY[
                                        entries[method_name].family
                                    ],
                                    segments=segments,
                                    feature_mask=(
                                        interp_method.last_runtime_kwargs.get(
                                            "feature_mask"
                                        )
                                    ),
                                    calibrate_fidelity=entries[
                                        method_name
                                    ].calibrate_fidelity,
                                )
                            )

                pbar.update(1)
                yield record

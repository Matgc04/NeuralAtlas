from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import numpy as np
from attr_config import AttributionConfig
import torch

from backend import config

if TYPE_CHECKING:
    from captum._utils.typing import TensorOrTupleOfTensorsGeneric
    from torch import nn


LOCAL_FAMILY = "local"
GLOBAL_FAMILY = "global"

# Every model in the catalog is Resize(256) + CenterCrop(224) on RGB, so the
# lift to input space and the zero baselines all share one shape.
INPUT_SIZE = 224
INPUT_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class MethodCatalogEntry:
    """One attribution method as exposed to the viewer.

    `family` selects the infidelity perturbation (Yeh et al., 2019, §2.5):
    local methods report sensitivity; global methods estimate output change.
    `category` describes computation, so a gradient method can be global.
    `calibrate_fidelity` marks the mask-based methods whose attribution is on an
    arbitrary scale, so fidelity fits a scale before scoring it.
    """

    id: str
    label: str
    category: str
    family: str
    requires_layer: bool = False
    segmentation: str | None = None
    calibrate_fidelity: bool = False

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "family": self.family,
            "requires_layer": self.requires_layer,
            "segmentation": self.segmentation,
        }


class InsufficientFeaturesError(ValueError):
    def __init__(self, feature_count: int) -> None:
        self.feature_count = feature_count
        super().__init__(
            f"Attribution requires at least two interpretable features, got {feature_count}."
        )


def to_input_space(attr: object) -> torch.Tensor:
    """Lift a layer attribution to input resolution, keeping its sign and total.

    Sum layer channels to preserve signed evidence, then divide across input
    channels to avoid counting each pixel three times in infidelity.
    """
    from captum.attr import LayerAttribution

    if not isinstance(attr, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(attr)}")
    if attr.dim() != 4:
        raise ValueError(f"Unexpected attribution shape: {tuple(attr.shape)}")

    attr = attr.sum(dim=1, keepdim=True)
    layer_cells = attr.shape[-2] * attr.shape[-1]
    attr = LayerAttribution.interpolate(
        attr,
        (INPUT_SIZE, INPUT_SIZE),
        interpolate_mode="bilinear",
    )
    # Bilinear upsampling preserves values, inflating the sum by INPUT_SIZE^2 / layer_cells.
    # Rescale so the lifted map sums to the original layer attribution.
    attr = attr * (layer_cells / (INPUT_SIZE * INPUT_SIZE))
    return attr.expand(-1, INPUT_CHANNELS, -1, -1) / INPUT_CHANNELS


class SignedGuidedGradCam:
    """GuidedGradCam with the Grad-CAM ReLU exposed instead of hardcoded.

    Multiply GuidedBackprop by interpolated LayerGradCam, allowing signed CAMs.
    """

    def __init__(self, model: "nn.Module", layer: "nn.Module") -> None:
        from captum.attr import GuidedBackprop, LayerGradCam

        self.grad_cam = LayerGradCam(model, layer)
        self.guided_backprop = GuidedBackprop(model)

    def attribute(
        self,
        inputs: "TensorOrTupleOfTensorsGeneric",
        target: object,
        relu_attributions: bool,
    ) -> torch.Tensor:
        grad_cam = self.grad_cam.attribute(
            inputs,
            target,
            relu_attributions=relu_attributions,
        )
        guided = self.guided_backprop.attribute(inputs, target)
        return guided * self.grad_cam.interpolate(
            grad_cam,
            tuple(inputs.shape[2:]),
            interpolate_mode="bilinear",
        )


def make_superpixel_mask(
    mask_function: Callable[..., object],
    img: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    img_np = img.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    seg = mask_function(img_np, **kwargs)
    seg_t = torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)
    return seg_t.to(img.device)


def kmeans_superpixels(
    img_np: np.ndarray,
    n_clusters: int = 100,
    add_xy: bool = True,
    xy_weight: float = 0.2,
    random_state: int = 0,
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {img_np.shape}.")

    h, w, _ = img_np.shape
    rgb = img_np.reshape(-1, 3).astype(np.float32)

    if add_xy:
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        xy = np.stack((yy, xx), axis=-1).reshape(-1, 2) * float(xy_weight)
        features = np.concatenate((rgb, xy), axis=1)
    else:
        features = rgb

    cluster_count = int(np.clip(int(n_clusters), 2, h * w))
    labels = KMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    ).fit_predict(features)

    return labels.reshape(h, w).astype(np.int64, copy=False)


class _SuperpixelFeatures:
    """Segment an input once and hand the mask to Captum as `feature_mask`.

    Holds only the segmentation and its last input, rather than capturing the
    enclosing scope: the provider lives as long as the catalog entry does.
    `AttributionConfig` records the kwargs it passed, so fidelity reads the mask
    back from there to undo Captum's per-pixel repetition of each coefficient.
    """

    __slots__ = ("_mask_fn", "_seg_kwargs", "_cached_inputs", "_cached_mask")

    def __init__(self, mask_fn: Callable[..., object], **seg_kwargs: object) -> None:
        self._mask_fn = mask_fn
        self._seg_kwargs = seg_kwargs
        self._cached_inputs: torch.Tensor | None = None
        self._cached_mask: torch.Tensor | None = None

    def __call__(
        self, inputs: "TensorOrTupleOfTensorsGeneric", _target: object
    ) -> dict[str, "torch.Tensor"]:
        inputs_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
        if not isinstance(inputs_tensor, torch.Tensor):
            raise TypeError(
                "Lime runtime kwargs expected tensor inputs or tuple[Tensor, ...], "
                f"got {type(inputs_tensor)}."
            )

        if inputs_tensor is not self._cached_inputs:
            mask = make_superpixel_mask(
                mask_function=self._mask_fn,
                img=inputs_tensor,
                **self._seg_kwargs,
            )
            # Validate on a miss only; the count cannot change while the mask is cached.
            feature_count = int(torch.unique(mask).numel())
            if feature_count < 2:
                raise InsufficientFeaturesError(feature_count)
            self._cached_mask = mask
            self._cached_inputs = inputs_tensor

        if self._cached_mask is None:
            raise RuntimeError("Superpixel mask cache was not initialized.")
        return {"feature_mask": self._cached_mask}


@lru_cache(maxsize=1)
def _catalog() -> tuple[MethodCatalogEntry, ...]:
    return tuple(_build_catalog())


def method_catalog() -> list[MethodCatalogEntry]:
    """The catalog, rebuilt from a cached immutable copy so callers may mutate it."""
    return list(_catalog())


def _build_catalog() -> list[MethodCatalogEntry]:
    base_entries = [
        MethodCatalogEntry("CB-RISE", "CB-RISE", "perturbation", GLOBAL_FAMILY, calibrate_fidelity=True),
        MethodCatalogEntry("RISE", "RISE", "perturbation", GLOBAL_FAMILY, calibrate_fidelity=True),
        MethodCatalogEntry("Occlusion", "Occlusion", "perturbation", GLOBAL_FAMILY, calibrate_fidelity=True),
        MethodCatalogEntry("GuidedGradCam", "GuidedGradCam", "gradient", LOCAL_FAMILY, True),
        MethodCatalogEntry("GradientShap", "GradientShap", "gradient", GLOBAL_FAMILY),
        MethodCatalogEntry("Saliency", "Saliency", "gradient", LOCAL_FAMILY),
        MethodCatalogEntry("IntegratedGradients", "IntegratedGradients", "gradient", GLOBAL_FAMILY),
        MethodCatalogEntry("LayerGradCam", "LayerGradCam", "gradient", LOCAL_FAMILY, True),
        MethodCatalogEntry("DeepLift", "DeepLift", "gradient", GLOBAL_FAMILY),
        MethodCatalogEntry("GuidedBackprop", "GuidedBackprop", "gradient", LOCAL_FAMILY),
        MethodCatalogEntry("InputXGradient", "InputXGradient", "gradient", GLOBAL_FAMILY),
        MethodCatalogEntry("Deconvolution", "Deconvolution", "gradient", LOCAL_FAMILY),
        MethodCatalogEntry(
            "LayerIntegratedGradients",
            "LayerIntegratedGradients",
            "gradient",
            GLOBAL_FAMILY,
            True,
        ),
    ]
    segmented_methods = ["Lime", "KernelShap"]
    segmentations = ["SLIC", "KMeans"]
    for method_name in segmented_methods:
        for segmentation in segmentations:
            base_entries.append(
                MethodCatalogEntry(
                    id=f"{method_name} ({segmentation})",
                    label=f"{method_name} ({segmentation})",
                    category="perturbation",
                    family=GLOBAL_FAMILY,
                    segmentation=segmentation,
                )
            )
    return base_entries


def extra_metric_keys(metrics: set[str]) -> dict[str, set[str]]:
    """Metric keys a method emits beyond `metrics`, keyed by method id.

    Superpixel methods carry a `feature_mask`, so `evaluate_faithfulness` scores a
    second fidelity variant for them whenever fidelity is requested. Deriving this
    from `segmentation` keeps the producer and the completion check reading the
    same definition instead of matching on method names.
    """
    if "fidelity" not in metrics:
        return {}
    return {
        entry.id: {"fidelity_superpixel"}
        for entry in _catalog()
        if entry.segmentation is not None
    }


def build_interp_methods(
    last_conv_layer: "nn.Module",
    device: "torch.device",
) -> list[AttributionConfig]:
    from captum.attr import (
        Deconvolution,
        DeepLift,
        GradientShap,
        GuidedBackprop,
        InputXGradient,
        IntegratedGradients,
        KernelShap,
        LayerGradCam,
        LayerIntegratedGradients,
        Lime,
        Occlusion,
        Saliency,
    )
    from captum.attr._core.deep_lift import SUPPORTED_NON_LINEAR, nonlinear
    from skimage.segmentation import slic

    from backend.cb_rise import CBRISE
    from backend.models import HookableReLU, HookableReLU6
    from backend.rise import RISE

    # DeepLift picks its rule by exact module type. 
    # Since we changed ReLu6 and F.Relu activations we need this so captum can capture them
    SUPPORTED_NON_LINEAR.update(dict.fromkeys((HookableReLU, HookableReLU6), nonlinear))

    slic_medium = _SuperpixelFeatures(
        slic,
        n_segments=32,
        compactness=10.0,
        start_label=0,
    )
    kmeans_medium = _SuperpixelFeatures(
        kmeans_superpixels,
        n_clusters=32,
        add_xy=True,
        xy_weight=0.2,
        random_state=0,
        n_init=10,
    )
    # One reference point for the attribution baselines and the fidelity removal,
    # so a removed patch means the same thing everywhere.
    zero_baseline = torch.full(
        (1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        config.FIDELITY_SQUARE_BASELINE,
        device=device,
    )

    return [
        AttributionConfig(
            CBRISE,
            name="CB-RISE",
            n_masks=4096,
            grid_size=7,
            probability=0.5,
            mask_batch_size=32,
            sigma=10.0,
            patience=64,
            epsilon=1e-3,
            threshold=0.3,
            seed=0,
        ),
        AttributionConfig(
            RISE,
            n_masks=2048,
            grid_size=7,
            probability=0.5,
            mask_batch_size=64,
            seed=0,
        ),
        AttributionConfig(
            Occlusion,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
            perturbations_per_eval=16,
        ),
        AttributionConfig(
            SignedGuidedGradCam,
            layer=last_conv_layer,
            relu_attributions=False,
            name="GuidedGradCam",
        ),
        AttributionConfig(
            GradientShap,
            n_samples=100,
            stdevs=0.05,
            baselines=zero_baseline,
        ),
        AttributionConfig(
            Saliency,
            # Preserve the sign for infidelity.
            abs=False,
        ),
        AttributionConfig(IntegratedGradients, n_steps=50),
        AttributionConfig(
            LayerGradCam,
            layer=last_conv_layer,
            # Preserve the sign for infidelity.
            relu_attributions=False,
            callback=to_input_space,
        ),
        AttributionConfig(
            DeepLift,
            baselines=zero_baseline,
        ),
        AttributionConfig(GuidedBackprop),
        AttributionConfig(InputXGradient),
        AttributionConfig(Deconvolution),
        AttributionConfig(
            Lime,
            runtime_kwargs_fn=slic_medium,
            n_samples=500,
            perturbations_per_eval=32,
            suffix="(SLIC)",
        ),
        AttributionConfig(
            Lime,
            runtime_kwargs_fn=kmeans_medium,
            n_samples=500,
            perturbations_per_eval=32,
            suffix="(KMeans)",
        ),
        AttributionConfig(
            KernelShap,
            runtime_kwargs_fn=slic_medium,
            n_samples=500,
            perturbations_per_eval=32,
            suffix="(SLIC)",
        ),
        AttributionConfig(
            KernelShap,
            runtime_kwargs_fn=kmeans_medium,
            n_samples=500,
            perturbations_per_eval=32,
            suffix="(KMeans)",
        ),
        AttributionConfig(
            LayerIntegratedGradients,
            layer=last_conv_layer,
            baselines=zero_baseline,
            n_steps=50,
            attribute_to_layer_input=False,
            callback=to_input_space,
        ),
    ]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from attr_config import AttributionConfig
import torch

if TYPE_CHECKING:
    from captum._utils.typing import TensorOrTupleOfTensorsGeneric
    from torch import nn


@dataclass(frozen=True, slots=True)
class MethodCatalogEntry:
    id: str
    label: str
    category: str
    requires_layer: bool = False
    segmentation: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "requires_layer": self.requires_layer,
            "segmentation": self.segmentation,
        }


def to_rgb_heatmap(attr: object) -> torch.Tensor:
    from captum.attr import LayerAttribution

    if not isinstance(attr, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(attr)}")

    if attr.dim() == 4:
        attr = attr.abs().mean(dim=1, keepdim=True)
    elif attr.dim() == 3:
        attr = attr.abs().mean(dim=0, keepdim=True)
        attr = attr.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected attribution shape: {tuple(attr.shape)}")

    attr = LayerAttribution.interpolate(
        attr,
        (224, 224),
        interpolate_mode="bilinear",
    )
    return attr.repeat(1, 3, 1, 1)


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


def _make_superpixel_runtime_kwargs(mask_fn: Callable[..., object], **seg_kwargs: object):
    def _runtime_kwargs(
        inputs: "TensorOrTupleOfTensorsGeneric", _target: object
    ) -> dict[str, "torch.Tensor"]:
        inputs_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
        if not isinstance(inputs_tensor, torch.Tensor):
            raise TypeError(
                "Lime runtime kwargs expected tensor inputs or tuple[Tensor, ...], "
                f"got {type(inputs_tensor)}."
            )
        return {
            "feature_mask": make_superpixel_mask(
                mask_function=mask_fn,
                img=inputs_tensor,
                **seg_kwargs,
            )
        }

    return _runtime_kwargs


def method_catalog() -> list[MethodCatalogEntry]:
    base_entries = [
        MethodCatalogEntry("Occlusion", "Occlusion", "perturbation"),
        MethodCatalogEntry("GuidedGradCam", "GuidedGradCam", "gradient", True),
        MethodCatalogEntry("GradientShap", "GradientShap", "gradient"),
        MethodCatalogEntry("Saliency", "Saliency", "gradient"),
        MethodCatalogEntry("IntegratedGradients", "IntegratedGradients", "gradient"),
        MethodCatalogEntry("LayerGradCam", "LayerGradCam", "gradient", True),
        MethodCatalogEntry("DeepLift", "DeepLift", "gradient"),
        MethodCatalogEntry("GuidedBackprop", "GuidedBackprop", "gradient"),
        MethodCatalogEntry("InputXGradient", "InputXGradient", "gradient"),
        MethodCatalogEntry("Deconvolution", "Deconvolution", "gradient"),
        MethodCatalogEntry("LayerIntegratedGradients", "LayerIntegratedGradients", "gradient", True),
    ]
    segmented_methods = ["Lime", "KernelShap", "ShapleyValueSampling"]
    segmentations = ["SLIC", "Quickshift", "KMeans"]
    for method_name in segmented_methods:
        for segmentation in segmentations:
            base_entries.append(
                MethodCatalogEntry(
                    id=f"{method_name} ({segmentation})",
                    label=f"{method_name} ({segmentation})",
                    category="perturbation",
                    segmentation=segmentation,
                )
            )
    return base_entries


def build_interp_methods(
    last_conv_layer: "nn.Module",
    device: "torch.device",
    to_rgb_heatmap: Callable[
        ["TensorOrTupleOfTensorsGeneric"], "TensorOrTupleOfTensorsGeneric"
    ],
) -> list[AttributionConfig]:
    from captum.attr import (
        Deconvolution,
        DeepLift,
        GradientShap,
        GuidedBackprop,
        GuidedGradCam,
        InputXGradient,
        IntegratedGradients,
        KernelShap,
        LayerAttribution,
        LayerGradCam,
        LayerIntegratedGradients,
        Lime,
        Occlusion,
        Saliency,
        ShapleyValueSampling,
    )
    from skimage.segmentation import quickshift, slic

    slic_medium = _make_superpixel_runtime_kwargs(
        slic,
        n_segments=100,
        compactness=10.0,
        start_label=0,
    )
    quickshift_medium = _make_superpixel_runtime_kwargs(
        quickshift,
        kernel_size=8,
        max_dist=15,
        ratio=0.8,
    )
    kmeans_medium = _make_superpixel_runtime_kwargs(
        kmeans_superpixels,
        n_clusters=16,
        add_xy=True,
        xy_weight=0.2,
        random_state=0,
        n_init=10,
    )

    return [
        AttributionConfig(
            Occlusion,
            sliding_window_shapes=(3, 15, 15),
            strides=(3, 8, 8),
        ),
        AttributionConfig(GuidedGradCam, layer=last_conv_layer),
        AttributionConfig(
            GradientShap,
            n_samples=50,
            stdevs=0.0001,
            baselines=torch.ones(1, 3, 224, 224, device=device),
        ),
        AttributionConfig(Saliency),
        AttributionConfig(IntegratedGradients, n_steps=50),
        AttributionConfig(
            LayerGradCam,
            layer=last_conv_layer,
            relu_attributions=True,
            callback=lambda attr: LayerAttribution.interpolate(
                attr,
                (224, 224),
                interpolate_mode="bilinear",
            ).repeat(1, 3, 1, 1),
        ),
        AttributionConfig(
            DeepLift,
            baselines=torch.ones(1, 3, 224, 224, device=device),
        ),
        AttributionConfig(GuidedBackprop),
        AttributionConfig(InputXGradient),
        AttributionConfig(Deconvolution),
        AttributionConfig(Lime, runtime_kwargs_fn=slic_medium, n_samples=50, suffix="(SLIC)"),
        AttributionConfig(
            Lime,
            runtime_kwargs_fn=quickshift_medium,
            n_samples=50,
            suffix="(Quickshift)",
        ),
        AttributionConfig(
            Lime,
            runtime_kwargs_fn=kmeans_medium,
            n_samples=50,
            suffix="(KMeans)",
        ),
        AttributionConfig(
            KernelShap,
            runtime_kwargs_fn=slic_medium,
            n_samples=50,
            suffix="(SLIC)",
        ),
        AttributionConfig(
            KernelShap,
            runtime_kwargs_fn=quickshift_medium,
            n_samples=50,
            suffix="(Quickshift)",
        ),
        AttributionConfig(
            KernelShap,
            runtime_kwargs_fn=kmeans_medium,
            n_samples=50,
            suffix="(KMeans)",
        ),
        AttributionConfig(
            ShapleyValueSampling,
            runtime_kwargs_fn=slic_medium,
            n_samples=15,
            perturbations_per_eval=8,
            suffix="(SLIC)",
        ),
        AttributionConfig(
            ShapleyValueSampling,
            runtime_kwargs_fn=quickshift_medium,
            n_samples=15,
            perturbations_per_eval=8,
            suffix="(Quickshift)",
        ),
        AttributionConfig(
            ShapleyValueSampling,
            runtime_kwargs_fn=kmeans_medium,
            n_samples=15,
            perturbations_per_eval=8,
            suffix="(KMeans)",
        ),
        AttributionConfig(
            LayerIntegratedGradients,
            layer=last_conv_layer,
            baselines=torch.ones(1, 3, 224, 224, device=device),
            n_steps=50,
            attribute_to_layer_input=False,
            callback=to_rgb_heatmap,
        ),
    ]

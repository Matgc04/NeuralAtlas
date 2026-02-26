from typing import Callable

import numpy as np
from torch import nn
import torch
from captum.attr import LayerAttribution

def disable_inplace_relu(model: nn.Module) -> nn.Module:
    """Recursively set inplace=False on all ReLU modules in the model.
    This is necessary for layer attribution methods (e.g. LayerIntegratedGradients)
    that require gradient computation through
    activations.
    
    Functionally identical to the original model — only affects memory usage.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def to_rgb_heatmap(attr):
    """"Convert an N channel attribution map to a 3-channel RGB heatmap for visualization."""

    if not isinstance(attr, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(attr)}")

    # Expect NCHW or CHW
    if attr.dim() == 4:            # [N, C, H, W]
        # collapse channels -> [N, 1, H, W]
        attr = attr.abs().mean(dim=1, keepdim=True)
    elif attr.dim() == 3:          # [C, H, W]
        attr = attr.abs().mean(dim=0, keepdim=True)   # [1, H, W]
        attr = attr.unsqueeze(0)                      # [1, 1, H, W]
    else:
        raise ValueError(f"Unexpected attribution shape: {tuple(attr.shape)}")

    # Upsample single-channel map to input resolution
    attr = LayerAttribution.interpolate(attr, (224, 224), interpolate_mode="bilinear")  # [N,1,224,224]

    # Make 3-channel for visualization code that expects RGB-like
    return attr.repeat(1, 3, 1, 1)  # [N,3,224,224]



def make_superpixel_mask(mask_function: Callable, img: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    img: (1, 3, H, W) float in [0,1] on CPU or GPU.
    returns feature_mask: (1, 1, H, W) long on same device as img
    """
    # slic works on CPU numpy, so move a copy to CPU
    img_np = img.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()  # (H,W,3)

    seg = mask_function(img_np, **kwargs)
    seg_t = torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
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
    """
    Build a superpixel-like label map with KMeans.

    img_np: (H, W, 3) float image in [0, 1]
    returns: (H, W) integer labels
    """
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

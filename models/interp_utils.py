from typing import Callable

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
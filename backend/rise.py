from __future__ import annotations

import math

import torch
from captum._utils.common import ExpansionTypes, _expand_target, _select_targets
from captum._utils.typing import TargetType
from captum.attr import Attribution
from torch.nn import functional as F


def centered_saliency(
    weighted_scores: torch.Tensor,
    mask_sum: torch.Tensor,
    score_sum: torch.Tensor,
    samples: int,
    channels: int,
) -> torch.Tensor:
    """Normalize by observed mask weight and subtract the mean masked score.

    Unlike published RISE, this produces signed maps. Observed mask weights
    account for border coverage; dividing across channels avoids multiplying
    the attribution when infidelity sums over channels.
    """
    saliency = weighted_scores / mask_sum.clamp_min(
        torch.finfo(weighted_scores.dtype).eps
    )
    saliency -= (score_sum / samples)[:, None, None]
    return saliency[:, None].expand(-1, channels, -1, -1) / channels


class RISE(Attribution):
    """Randomized Input Sampling for Explanation, centered.

    RISE estimates spatial importance by averaging random masks weighted by the
    model score they preserve. Masks are generated and evaluated in batches so
    memory use does not grow with ``n_masks``. The map is finished by
    `centered_saliency`, which documents how it departs from the paper.
    """

    @staticmethod
    def _generate_masks(
        count: int,
        height: int,
        width: int,
        grid_size: int,
        probability: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cell_height = math.ceil(height / grid_size)
        cell_width = math.ceil(width / grid_size)
        upsampled_height = (grid_size + 1) * cell_height
        upsampled_width = (grid_size + 1) * cell_width

        # Keep seeded draws on CPU; upsample and gather on the model device.
        coarse = (
            torch.rand(
                count,
                1,
                grid_size,
                grid_size,
                generator=generator,
            )
            < probability
        ).float().to(device=device)
        upsampled = F.interpolate(
            coarse,
            size=(upsampled_height, upsampled_width),
            mode="bilinear",
            align_corners=False,
        )

        y = torch.randint(cell_height, (count,), generator=generator).to(device)
        x = torch.randint(cell_width, (count,), generator=generator).to(device)
        rows = y[:, None, None] + torch.arange(height, device=device)[None, :, None]
        columns = x[:, None, None] + torch.arange(width, device=device)[None, None, :]
        batch = torch.arange(count, device=device)[:, None, None]
        masks = upsampled[batch, 0, rows, columns].unsqueeze(1)
        return masks.to(dtype=dtype)

    def attribute(
        self,
        inputs: torch.Tensor,
        target: TargetType = None,
        *,
        n_masks: int = 2048,
        grid_size: int = 7,
        probability: float = 0.5,
        mask_batch_size: int = 128,
        baselines: float | torch.Tensor = 0.0,
        seed: int = 0,
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"RISE expects BCHW inputs, got shape {tuple(inputs.shape)}.")
        if n_masks <= 0:
            raise ValueError("n_masks must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1].")
        if mask_batch_size <= 0:
            raise ValueError("mask_batch_size must be positive.")

        input_batch, channels, height, width = inputs.shape
        baseline = torch.as_tensor(
            baselines,
            device=inputs.device,
            dtype=inputs.dtype,
        ).broadcast_to(inputs.shape)
        saliency = inputs.new_zeros(input_batch, height, width)
        mask_sum = inputs.new_zeros(height, width)
        score_sum = inputs.new_zeros(input_batch)
        generator = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            for start in range(0, n_masks, mask_batch_size):
                count = min(mask_batch_size, n_masks - start)
                masks = self._generate_masks(
                    count,
                    height,
                    width,
                    grid_size,
                    probability,
                    generator,
                    inputs.device,
                    inputs.dtype,
                )
                masked = (
                    inputs[:, None] * masks[None]
                    + baseline[:, None] * (1.0 - masks[None])
                ).reshape(input_batch * count, channels, height, width)
                output = self.forward_func(masked)
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "RISE expects the model to return a tensor, "
                        f"got {type(output)}."
                    )

                expanded_target = _expand_target(
                    target,
                    count,
                    ExpansionTypes.repeat_interleave,
                )
                scores = _select_targets(torch.softmax(output, dim=-1), expanded_target)
                if scores.numel() != input_batch * count:
                    raise ValueError(
                        "RISE target must select one scalar score per masked input."
                    )
                batch_scores = scores.reshape(input_batch, count)
                saliency += torch.einsum("bm,mhw->bhw", batch_scores, masks[:, 0])
                mask_sum += masks[:, 0].sum(dim=0)
                score_sum += batch_scores.sum(dim=1)

        return centered_saliency(saliency, mask_sum, score_sum, n_masks, channels)

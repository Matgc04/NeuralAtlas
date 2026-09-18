from __future__ import annotations

import torch
from captum._utils.common import ExpansionTypes, _expand_target, _select_targets
from captum._utils.typing import TargetType
from captum.attr import Attribution
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

from backend.rise import centered_saliency


class CBRISE(Attribution):
    """RISE with convergence detection and blurred perturbations, centered.

    This implementation also applies prediction-relative normalization (PRN),
    the third component of CB-RISE. Model logits are converted to probabilities
    before PRN so the score ratios have the meaning defined by the method.

    PRN factors are non-negative, so the raw map inherits the sign problem of
    RISE, and the map is finished by the same `centered_saliency`. The
    convergence test is unaffected by the centering because it compares min-max
    normalized snapshots, which are invariant to a constant shift.
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
        # Only the coarse grid is drawn from the CPU generator, so a seed keeps
        # meaning the same masks; the upsampling to full resolution and the
        # normalization run on `device` (see `RISE._generate_masks`).
        coarse = (
            torch.rand(
                count,
                1,
                grid_size,
                grid_size,
                generator=generator,
            )
            > 1.0 - probability
        ).float().to(device=device)
        masks = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        minimum = masks.amin(dim=(1, 2, 3), keepdim=True)
        masks = masks - minimum
        maximum = masks.amax(dim=(1, 2, 3), keepdim=True)
        return (masks / (maximum + 1e-8)).to(dtype=dtype)

    @staticmethod
    def _blurred_inputs(inputs: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel_size = int((((sigma - 0.8) / 0.3) + 1.0) * 2.0 + 1.0)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return gaussian_blur(inputs, [kernel_size, kernel_size], [sigma, sigma])

    @staticmethod
    def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
        minimum = heatmap.amin(dim=(-2, -1), keepdim=True)
        normalized = heatmap - minimum
        maximum = normalized.amax(dim=(-2, -1), keepdim=True)
        return normalized / maximum.clamp_min(torch.finfo(heatmap.dtype).eps)

    @staticmethod
    def _target_scores(
        output: torch.Tensor,
        target: TargetType,
        expansion_count: int = 1,
    ) -> torch.Tensor:
        expanded_target = _expand_target(
            target,
            expansion_count,
            ExpansionTypes.repeat_interleave,
        )
        scores = _select_targets(torch.softmax(output, dim=-1), expanded_target)
        if scores.numel() != output.shape[0]:
            raise ValueError(
                "CBRISE target must select one scalar score per model output."
            )
        return scores.reshape(output.shape[0])

    def attribute(
        self,
        inputs: torch.Tensor,
        target: TargetType = None,
        *,
        n_masks: int = 4096,
        grid_size: int = 7,
        probability: float = 0.5,
        mask_batch_size: int = 128,
        sigma: float = 10.0,
        patience: int = 64,
        epsilon: float = 1e-3,
        threshold: float = 0.3,
        seed: int = 0,
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                f"CBRISE expects BCHW inputs, got shape {tuple(inputs.shape)}."
            )
        if n_masks <= 0:
            raise ValueError("n_masks must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1].")
        if mask_batch_size <= 0:
            raise ValueError("mask_batch_size must be positive.")
        if sigma <= 0.8:
            raise ValueError("sigma must be greater than 0.8.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1).")

        input_batch, channels, height, width = inputs.shape
        blurred = self._blurred_inputs(inputs, sigma)
        heatmap = inputs.new_zeros(input_batch, height, width)
        mask_sum = inputs.new_zeros(height, width)
        factor_sum = inputs.new_zeros(input_batch)
        running_mean = inputs.new_zeros(input_batch, height, width)
        running_sum_squares = inputs.new_zeros(input_batch, height, width)
        previous_variance = inputs.new_zeros(input_batch, height, width)
        generator = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            original_output = self.forward_func(inputs)
            if not isinstance(original_output, torch.Tensor):
                raise TypeError(
                    "CBRISE expects the model to return a tensor, "
                    f"got {type(original_output)}."
                )
            original_scores = self._target_scores(original_output, target)
            original_scores = original_scores.clamp_min(
                torch.finfo(original_scores.dtype).eps
            )

            processed = 0
            while processed < n_masks:
                until_checkpoint = patience - processed % patience
                count = min(
                    mask_batch_size,
                    n_masks - processed,
                    until_checkpoint,
                )
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
                    + blurred[:, None] * (1.0 - masks[None])
                ).reshape(input_batch * count, channels, height, width)
                output = self.forward_func(masked)
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "CBRISE expects the model to return a tensor, "
                        f"got {type(output)}."
                    )

                scores = self._target_scores(output, target, count).reshape(
                    input_batch, count
                )
                factors = torch.minimum(scores, original_scores[:, None])
                factors = factors / original_scores[:, None]
                heatmap += torch.einsum("bm,mhw->bhw", factors, masks[:, 0])
                mask_sum += masks[:, 0].sum(dim=0)
                factor_sum += factors.sum(dim=1)
                processed += count

                if processed % patience != 0:
                    continue

                current = self._normalize_heatmap(
                    heatmap / mask_sum.clamp_min(torch.finfo(inputs.dtype).eps)
                )
                checkpoints = processed // patience
                delta = current - running_mean
                running_mean += delta / checkpoints
                running_sum_squares += delta * (current - running_mean)
                if checkpoints < 2:
                    continue

                variance = running_sum_squares / (checkpoints - 1)
                stable = (variance - previous_variance).abs() < epsilon
                if stable.float().mean() > 1.0 - threshold:
                    break
                previous_variance = variance

        return centered_saliency(heatmap, mask_sum, factor_sum, processed, channels)

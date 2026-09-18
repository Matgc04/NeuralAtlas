"""Fidelity and calibrated spatial predictive improvement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from backend.metrics.metrics import Metric


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    """The noisy baseline of Yeh et al. (2019), for local explanations.

    Every pixel gets i.i.d. noise, which probes the sensitivity of the function
    around ``x`` -- what a local explanation reports.

    A local attribution is a sensitivity per unit of input, so the change it
    predicts is the first-order term ``delta^T a`` and the weight is ``delta``
    itself.
    """

    std: float

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("std must be positive")

    def sample(
        self,
        inputs: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        removed = torch.randn(
            (count,) + tuple(inputs.shape),
            device=inputs.device,
            dtype=inputs.dtype,
            generator=generator,
        ) * self.std
        return removed, removed


@dataclass(frozen=True, slots=True)
class SquareRemoval:
    """Square removal of Yeh et al. (2019), for global explanations.

    A uniformly placed square is replaced by ``baseline``, so the perturbation
    is zero outside the patch. That asks the question a global explanation
    claims to answer -- how much does the logit drop if this region is removed
    -- and moves the output enough to carry signal, unlike small i.i.d. noise.

    The weight here is the patch mask, not the perturbation: a global
    attribution already carries the displacement from the baseline inside it
    (Captum spells this `multiply_by_inputs`, after the local/global split of
    Ancona et al., 2018), so the change it predicts is the attribution summed
    over the removed region. Weighting by the perturbation as well would apply
    the displacement twice; on a linear model with a zero baseline that turns
    the exact prediction ``sum_i m_i w_i x_i`` into ``sum_i m_i w_i x_i^2``,
    penalising an attribution that is right.
    """

    size: int
    baseline: float

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be positive")

    def sample(
        self,
        inputs: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = inputs.shape
        if self.size > min(height, width):
            raise ValueError("size must not exceed the smaller input side")

        removed = inputs - torch.as_tensor(
            self.baseline,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        rows = torch.arange(height, device=inputs.device)
        columns = torch.arange(width, device=inputs.device)
        tops = torch.randint(
            height - self.size + 1,
            (count, batch),
            device=inputs.device,
            generator=generator,
        )
        lefts = torch.randint(
            width - self.size + 1,
            (count, batch),
            device=inputs.device,
            generator=generator,
        )
        in_rows = (rows >= tops.unsqueeze(-1)) & (
            rows < (tops + self.size).unsqueeze(-1)
        )
        in_columns = (columns >= lefts.unsqueeze(-1)) & (
            columns < (lefts + self.size).unsqueeze(-1)
        )
        # (count, batch, 1, height, width); the channel axis broadcasts, so the
        # weighted sum runs over every channel of the attribution.
        mask = (
            in_rows.unsqueeze(-1) & in_columns.unsqueeze(-2)
        ).unsqueeze(2).to(inputs.dtype)
        return mask * removed.unsqueeze(0), mask


@dataclass(frozen=True, slots=True)
class SuperpixelRemoval:
    """Remove one uniformly sampled feature per image, with replacement.

    Reuse the attribution's segmentation. Each draw removes a whole feature,
    irrespective of its area; this is a separate fidelity experiment from squares.
    """

    feature_mask: torch.Tensor
    baseline: float

    def sample(
        self,
        inputs: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The same segmentation covers every sample in the batch, so the feature
        # ids are found once and the draws broadcast over (count, batch).
        features = self.feature_mask.unique()
        selected = features[
            torch.randint(
                features.numel(),
                (count, inputs.shape[0], 1, 1, 1),
                device=inputs.device,
                generator=generator,
            )
        ]
        # (count, batch, 1, height, width); the channel axis broadcasts, as above.
        mask = (self.feature_mask == selected).to(inputs.dtype)
        return mask * (inputs - self.baseline).unsqueeze(0), mask


Perturbation = GaussianNoise | SquareRemoval | SuperpixelRemoval


class FidelityScore(Metric):
    """Relative reduction in infidelity over a zero attribution.

    For each sampled perturbation ``delta``, the attribution predicts an output
    change and the model supplies the observed target-logit change
    ``f(x) - f(x - delta)``. The perturbation also supplies the weight the
    attribution is summed against, because that differs by explanation family:
    ``delta`` for a local attribution, the removal mask for a global one. The
    score is

    ``1 - E[(predicted - observed)^2] / E[observed^2]``.

    One is perfect, zero matches the zero-attribution baseline, and negative
    values are worse than that baseline. A zero baseline error makes the score
    undefined and is represented as ``NaN``.

    Yeh et al. (2019) sample ``delta`` differently for each explanation family
    (section 2.5), so the caller passes the perturbation: `GaussianNoise` for
    local explanations, `SquareRemoval` for global ones. Scores from the two
    measure different things and must not be ranked against each other.

    With ``calibrate=True``, fit an intercept and nonnegative slope on separate
    samples and compare against their mean output change instead of zero.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attributions: torch.Tensor,
        targets: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.inputs = inputs.detach()
        self.attributions = attributions.detach()
        self.targets = targets.detach()
        self._original_scores: torch.Tensor | None = None
        self._validate_inputs()
        if feature_mask is not None:
            self.attributions = self._share_over_features(
                self.attributions, feature_mask
            )

    @staticmethod
    def _share_over_features(
        attributions: torch.Tensor, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        """Divide a per-feature attribution by the entries it was repeated over.

        Lime and KernelShap give one coefficient per superpixel, which Captum repeats
        across every pixel and channel of that segment. Since `update` sums over
        pixels, the coefficient would be counted once per entry (~1568x3 for a 224x224
        image with 32 segments), inflating the predicted change. Sharing it evenly
        over its entries makes the sum count it once, weighted by the fraction of the
        segment the perturbation covers.
        """
        ids = feature_mask.expand_as(attributions)
        # Counted on one sample, since every sample repeats the same segments.
        entries_per_feature = torch.bincount(ids[0].reshape(-1))
        return attributions / entries_per_feature[ids].to(attributions.dtype)

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self) -> None:
        self.validate_inputs(self.inputs, self.targets)
        if self.attributions.shape != self.inputs.shape:
            raise ValueError(
                "Attributions must have the same shape as inputs for fidelity"
            )
        if self.attributions.device != self.inputs.device:
            raise ValueError("Inputs and attributions must be on the same device")
        if self.inputs.dim() != 4:
            raise ValueError("Fidelity expects BCHW inputs")

    @staticmethod
    def _target_scores(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return outputs.gather(1, targets.view(-1, 1)).squeeze(1)

    def update(
        self,
        perturbation: Perturbation,
        n_perturb_samples: int = 25,
        max_examples_per_batch: int = 5,
        random_seed: int = 0,
        calibrate: bool = False,
    ) -> None:
        """Sample perturbations and reduce them to one score per image.

        ``calibrate`` draws ``n_perturb_samples`` twice: the first set fits the
        intercept and slope, the second is scored against them.
        """
        if n_perturb_samples < 1:
            raise ValueError("n_perturb_samples must be positive")
        if max_examples_per_batch < 1:
            raise ValueError("max_examples_per_batch must be positive")

        batch_size = self.inputs.shape[0]
        predictions = []
        observations = []
        sample_count = n_perturb_samples * (2 if calibrate else 1)
        generator = torch.Generator(device=self.inputs.device).manual_seed(random_seed)

        with torch.no_grad():
            if self._original_scores is None:
                self._original_scores = self._target_scores(
                    self.model(self.inputs), self.targets
                )
            original_scores = self._original_scores

            sampled = 0
            while sampled < sample_count:
                count = min(max_examples_per_batch, sample_count - sampled)
                perturbations, weights = perturbation.sample(
                    self.inputs, count, generator
                )
                perturbed_inputs = self.inputs.unsqueeze(0) - perturbations

                flat_inputs = perturbed_inputs.flatten(0, 1)
                repeated_targets = self.targets.repeat(count)
                perturbed_scores = self._target_scores(
                    self.model(flat_inputs), repeated_targets
                ).view(count, batch_size)

                predicted_changes = (
                    weights * self.attributions.unsqueeze(0)
                ).flatten(2).sum(dim=2)
                observed_changes = original_scores.unsqueeze(0) - perturbed_scores
                predictions.append(predicted_changes)
                observations.append(observed_changes)
                sampled += count

        predicted = torch.cat(predictions)
        observed = torch.cat(observations)
        baseline_prediction = self.inputs.new_zeros(batch_size)
        if calibrate:
            z, predicted = predicted.split(n_perturb_samples)
            y, observed = observed.split(n_perturb_samples)
            z_mean = z.mean(dim=0)
            baseline_prediction = y.mean(dim=0)
            z = z - z_mean
            y = y - baseline_prediction
            # A zero denominator means z is constant, so the numerator is zero
            # too; the floor just keeps the division finite.
            denominator = z.square().sum(dim=0).clamp_min(torch.finfo(z.dtype).eps)
            scale = ((z * y).sum(dim=0) / denominator).clamp_min(0)
            predicted = baseline_prediction + scale * (predicted - z_mean)

        attribution_error_sum = (predicted - observed).square().sum(dim=0)
        baseline_error_sum = (observed - baseline_prediction).square().sum(dim=0)
        result = torch.full_like(attribution_error_sum, torch.nan)
        defined = baseline_error_sum > 0
        result[defined] = (
            1 - attribution_error_sum[defined] / baseline_error_sum[defined]
        )
        self.result = result

    def compute(self) -> torch.Tensor:
        if self.result is None:
            raise RuntimeError("Must run update() before computing fidelity")
        return self.result

    def reset(self) -> None:
        self.result = None

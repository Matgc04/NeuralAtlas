from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

MetricValue: TypeAlias = str | int | float | bool | None
InterpretabilityMetricsPayload: TypeAlias = dict[str, dict[str, MetricValue]]


@dataclass(frozen=True, slots=True)
class AttributionFailure:
    code: str
    feature_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {"code": self.code, "feature_count": self.feature_count}

    @classmethod
    def from_dict(cls, data: object) -> "AttributionFailure | None":
        if not isinstance(data, Mapping):
            return None
        code = data.get("code")
        feature_count = data.get("feature_count")
        if not isinstance(code, str) or not isinstance(feature_count, int):
            return None
        return cls(code=code, feature_count=feature_count)


@dataclass(slots=True)
class PredictionRecord:
    predicted_class_id: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "predicted_class_id": self.predicted_class_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "PredictionRecord | None":
        if data is None:
            return None
        try:
            confidence_value = data["confidence"]
            if not isinstance(confidence_value, (int, float, str)):
                return None
            return cls(
                predicted_class_id=str(data["predicted_class_id"]),
                confidence=float(confidence_value),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(slots=True)
class ImageRecord:
    model: str
    dataset: str
    class_id: str
    image_id: str
    source_filename: str
    original_url: str | None
    prediction: PredictionRecord | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    attribution_failures: dict[str, AttributionFailure] = field(default_factory=dict)
    interpretability_metrics: InterpretabilityMetricsPayload = field(
        default_factory=dict
    )

    @property
    def sample_key(self) -> tuple[str, str]:
        return self.class_id, self.source_filename

    def completed_methods(
        self,
        image_ext: str,
        metrics: set[str],
        extra_metrics: Mapping[str, set[str]] | None = None,
    ) -> set[str]:
        """Methods that need no rerun: a registered failure, or an output of the right
        format whose faithfulness metrics were all persisted.

        `extra_metrics` names the keys a given method emits on top of `metrics` --
        superpixel methods also score `fidelity_superpixel` -- so a record written
        before that key existed still counts as incomplete.
        """
        target_ext = f".{image_ext.lower()}"
        completed = set(self.attribution_failures)
        for method, url in self.outputs.items():
            required = metrics | (extra_metrics or {}).get(method, set())
            persisted = self.interpretability_metrics.get(method, {})
            if url.lower().endswith(target_ext) and persisted.keys() >= required:
                completed.add(method)
        return completed

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "image_id": self.image_id,
            "source_filename": self.source_filename,
            "original_url": self.original_url,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "outputs": dict(sorted(self.outputs.items())),
            "attribution_failures": {
                method: failure.to_dict()
                for method, failure in sorted(self.attribution_failures.items())
            },
            "interpretability_metrics": self.interpretability_metrics,
        }

    @classmethod
    def from_dict(
        cls,
        model: str,
        dataset: str,
        data: dict[str, object],
    ) -> "ImageRecord":
        source_filename = data.get("source_filename")
        if not source_filename:
            if not data.get("original_url"):
                raise ValueError("Image record needs source_filename or original_url")
            source_filename = Path(str(data["original_url"])).name

        prediction_data = data.get("prediction")
        prediction = PredictionRecord.from_dict(
            prediction_data if isinstance(prediction_data, Mapping) else None
        )

        outputs_data = data.get("outputs")
        outputs: dict[str, str] = {}
        if isinstance(outputs_data, Mapping):
            outputs = {str(key): str(value) for key, value in outputs_data.items()}

        failures_data = data.get("attribution_failures")
        attribution_failures: dict[str, AttributionFailure] = {}
        if isinstance(failures_data, Mapping):
            for method, failure_data in failures_data.items():
                failure = AttributionFailure.from_dict(failure_data)
                if failure is not None:
                    attribution_failures[str(method)] = failure

        metrics_data = data.get("interpretability_metrics")
        interpretability_metrics: InterpretabilityMetricsPayload = {}
        if isinstance(metrics_data, Mapping):
            interpretability_metrics = {
                str(method): {
                    str(metric_name): metric_value
                    for metric_name, metric_value in metric_values.items()
                }
                for method, metric_values in metrics_data.items()
                if isinstance(metric_values, Mapping)
            }

        return cls(
            model=model,
            dataset=dataset,
            class_id=str(data["class_id"]),
            image_id=str(data["image_id"]),
            source_filename=str(source_filename),
            original_url=(
                None
                if data.get("original_url") in {None, ""}
                else str(data.get("original_url"))
            ),
            prediction=prediction,
            outputs=outputs,
            attribution_failures=attribution_failures,
            interpretability_metrics=interpretability_metrics,
        )

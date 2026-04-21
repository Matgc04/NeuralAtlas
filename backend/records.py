from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

MetricValue: TypeAlias = str | int | float | bool | None
InterpretabilityMetricsPayload: TypeAlias = dict[str, dict[str, MetricValue]]


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
    original_url: str | None
    prediction: PredictionRecord | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    interpretability_metrics: InterpretabilityMetricsPayload = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "image_id": self.image_id,
            "original_url": self.original_url,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "outputs": dict(sorted(self.outputs.items())),
            "interpretability_metrics": self.interpretability_metrics,
        }

    @classmethod
    def from_dict(
        cls,
        model: str,
        dataset: str,
        data: dict[str, object],
    ) -> "ImageRecord":
        prediction_data = data.get("prediction")
        prediction = PredictionRecord.from_dict(
            prediction_data if isinstance(prediction_data, Mapping) else None
        )

        outputs_data = data.get("outputs")
        outputs: dict[str, str] = {}
        if isinstance(outputs_data, Mapping):
            outputs = {str(key): str(value) for key, value in outputs_data.items()}

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
            original_url=(
                None
                if data.get("original_url") in {None, ""}
                else str(data.get("original_url"))
            ),
            prediction=prediction,
            outputs=outputs,
            interpretability_metrics=interpretability_metrics,
        )

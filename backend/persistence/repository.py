from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from backend import config
from backend.methods import MethodCatalogEntry, extra_metric_keys, method_catalog
from backend.records import ImageRecord


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    id: str
    label: str
    family: str | None = None
    parameter_count: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "parameter_count": self.parameter_count,
        }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        Path(temp_path).replace(path)
    except Exception:
        try:
            Path(temp_path).unlink(missing_ok=True)
        finally:
            raise


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        with path.open() as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


@dataclass(slots=True)
class ClassMetricsBucket:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0


@dataclass(slots=True)
class MetricsBucket:
    total: int = 0
    correct: int = 0
    missing_predictions: int = 0
    per_class: dict[str, ClassMetricsBucket] = field(default_factory=dict)


def _get_per_class_bucket(bucket: MetricsBucket, class_id: str) -> ClassMetricsBucket:
    if class_id not in bucket.per_class:
        bucket.per_class[class_id] = ClassMetricsBucket()
    return bucket.per_class[class_id]


def _record_prediction(
    bucket: MetricsBucket,
    actual_class_id: str,
    predicted_class_id: str | None,
) -> None:
    actual_bucket = _get_per_class_bucket(bucket, actual_class_id)
    bucket.total += 1
    actual_bucket.support += 1

    if predicted_class_id is None or predicted_class_id == "":
        bucket.missing_predictions += 1
        actual_bucket.fn += 1
        return

    if predicted_class_id == actual_class_id:
        bucket.correct += 1
        actual_bucket.tp += 1
        return

    actual_bucket.fn += 1
    _get_per_class_bucket(bucket, predicted_class_id).fp += 1


def _finalize_metrics_bucket(bucket: MetricsBucket) -> dict[str, int | float]:
    class_count = len(bucket.per_class)

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for class_bucket in bucket.per_class.values():
        tp = class_bucket.tp
        fp = class_bucket.fp
        fn = class_bucket.fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    total = bucket.total
    correct = bucket.correct
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "missingPredictions": bucket.missing_predictions,
        "classCount": class_count,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "macroPrecision": (precision_sum / class_count) if class_count > 0 else 0.0,
        "macroRecall": (recall_sum / class_count) if class_count > 0 else 0.0,
        "macroF1": (f1_sum / class_count) if class_count > 0 else 0.0,
    }


class OutputRepository:
    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = output_root or config.OUTPUT_ROOT
        self.public_root = self.output_root.parent

    def _manifest_path(self) -> Path:
        return self.output_root / "manifest.json"

    def _models_catalog_path(self) -> Path:
        return self.output_root / "catalogs" / "models.json"

    def _methods_catalog_path(self) -> Path:
        return self.output_root / "catalogs" / "methods.json"

    def _run_dir(self, model: str, dataset: str) -> Path:
        return self.output_root / "runs" / model / dataset

    def _run_images_path(self, model: str, dataset: str) -> Path:
        return self._run_dir(model, dataset) / "images.json"

    def _run_summary_path(self, model: str, dataset: str) -> Path:
        return self._run_dir(model, dataset) / "summary.json"

    def vlm_descriptions_path(self, model: str, dataset: str, class_id: str) -> Path:
        return self._run_dir(model, dataset) / "vlm" / f"{class_id}.json"

    def load_vlm_descriptions(
        self,
        model: str,
        dataset: str,
        class_id: str,
    ) -> dict[str, Any] | None:
        """Load a class shard, checking its identity and image mapping shape."""
        path = self.vlm_descriptions_path(model, dataset, class_id)
        if not path.exists():
            return None
        with path.open() as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid VLM description shard: {path}")
        identity = (payload.get("model"), payload.get("dataset"), payload.get("class_id"))
        if identity != (model, dataset, class_id):
            raise ValueError(f"VLM description shard identity does not match its path: {path}")
        images = payload.get("images")
        if not isinstance(images, dict) or not all(
            isinstance(entry, dict) for entry in images.values()
        ):
            raise ValueError(f"Invalid VLM description shard images: {path}")
        return payload

    def upsert_vlm_description(
        self,
        model: str,
        dataset: str,
        class_id: str,
        image_id: str,
        method: str,
        generator: Mapping[str, object],
        description: Mapping[str, str],
        *,
        force: bool = False,
    ) -> Path:
        payload = self.load_vlm_descriptions(model, dataset, class_id)
        stale = payload is None or payload.get("generator") != generator
        if stale and payload is not None and not force:
            raise ValueError(
                f"VLM generator mismatch for {model}/{dataset}/{class_id}; use force to replace the shard"
            )
        if stale:
            payload = {
                "schema_version": 1,
                "model": model,
                "dataset": dataset,
                "class_id": class_id,
                "generator": dict(generator),
                "images": {},
            }

        payload["images"].setdefault(image_id, {})[method] = dict(description)

        path = self.vlm_descriptions_path(model, dataset, class_id)
        _atomic_write_json(path, payload)
        return path

    def load_images(self, model: str, dataset: str) -> list[ImageRecord]:
        payload = _read_json(self._run_images_path(model, dataset), {"model": model, "dataset": dataset, "images": []})
        images = payload.get("images", []) if isinstance(payload, dict) else []
        if not isinstance(images, list):
            return []
        return [
            ImageRecord.from_dict(model=model, dataset=dataset, data=item)
            for item in images
            if isinstance(item, dict)
        ]

    def _records_by_key(
        self,
        model: str,
        dataset: str,
    ) -> dict[tuple[str, str], ImageRecord]:
        return {
            (record.class_id, record.image_id): record
            for record in self.load_images(model, dataset)
        }

    def _completed_by_key(
        self,
        model: str,
        dataset: str,
        keys: list[tuple[str, str]],
        image_ext: str,
        metrics: set[str],
    ) -> Iterator[set[str]]:
        """Completed methods per key, in order; a key with no record completed none.

        Completion is decided per image rather than by a global output count: the same
        method can be done for one window and missing in the next.
        """
        records = self._records_by_key(model, dataset)
        extra = extra_metric_keys(metrics)
        for key in keys:
            record = records.get(key)
            yield (
                record.completed_methods(image_ext, metrics, extra) if record else set()
            )

    def methods_complete_for_all(
        self,
        model: str,
        dataset: str,
        keys: list[tuple[str, str]],
        image_ext: str,
        metrics: set[str],
    ) -> set[str]:
        """Methods already complete for every one of `keys`, so they can be skipped."""
        completed = list(self._completed_by_key(model, dataset, keys, image_ext, metrics))
        return set.intersection(*completed) if completed else set()

    def first_incomplete_sample(
        self,
        model: str,
        dataset: str,
        keys: list[tuple[str, str]],
        image_ext: str,
        metrics: set[str],
        methods: set[str] | None = None,
    ) -> int:
        """Length of the leading run of `keys` that has every catalog method complete."""
        required = methods or {entry.id for entry in method_catalog()}
        completed_by_key = self._completed_by_key(model, dataset, keys, image_ext, metrics)
        for index, completed in enumerate(completed_by_key):
            if not required <= completed:
                return index
        return len(keys)

    def upsert_image_records(
        self,
        model: str,
        dataset: str,
        records: list[ImageRecord],
    ) -> None:
        existing = self._records_by_key(model, dataset)
        for record in records:
            key = (record.class_id, record.image_id)
            current = existing.get(key)
            if current is None:
                existing[key] = record
                continue
            if record.original_url:
                current.original_url = record.original_url
            if record.prediction is not None:
                current.prediction = record.prediction
            current.outputs.update(record.outputs)
            for method_name in record.outputs:
                current.attribution_failures.pop(method_name, None)
            for method_name, failure in record.attribution_failures.items():
                current.outputs.pop(method_name, None)
                current.interpretability_metrics.pop(method_name, None)
                current.attribution_failures[method_name] = failure
            for method_name, metric_values in record.interpretability_metrics.items():
                current.interpretability_metrics.setdefault(method_name, {}).update(metric_values)

        merged = sorted(
            existing.values(),
            key=lambda record: (
                int(record.class_id) if record.class_id.isdigit() else record.class_id,
                int(record.image_id) if record.image_id.isdigit() else record.image_id,
            ),
        )
        self._write_run_bundle(model, dataset, merged)

    def prune_stale_artifacts(
        self,
        model: str,
        dataset: str,
        image_ext: str,
    ) -> tuple[int, int]:
        removed_json_entries = self._prune_stale_outputs(model, dataset, image_ext)
        removed_files = self._prune_stale_image_files(model, dataset, image_ext)
        return removed_json_entries, removed_files

    def _prune_stale_outputs(self, model: str, dataset: str, image_ext: str) -> int:
        target_ext = f".{image_ext.lower()}"
        known_methods = {entry.id for entry in method_catalog()}
        removed = 0
        remaining: list[ImageRecord] = []
        for record in self.load_images(model, dataset):
            stale_methods = {
                method_name
                for method_name, url in record.outputs.items()
                if not url.lower().endswith(target_ext)
            }
            orphan_metrics = set(record.interpretability_metrics) - set(record.outputs)
            # Methods dropped from the catalog keep older runs looking complete.
            retired_methods = (
                set(record.outputs)
                | set(record.interpretability_metrics)
                | set(record.attribution_failures)
            ) - known_methods
            pruned = stale_methods | orphan_metrics | retired_methods
            for method_name in pruned:
                record.outputs.pop(method_name, None)
                record.interpretability_metrics.pop(method_name, None)
                record.attribution_failures.pop(method_name, None)
            removed += len(pruned)
            if (
                record.outputs
                or record.attribution_failures
                or record.prediction is not None
                or record.original_url
                or record.interpretability_metrics
            ):
                remaining.append(record)
        if removed > 0:
            self._write_run_bundle(model, dataset, remaining)
        return removed

    def _prune_stale_image_files(
        self,
        model: str,
        dataset: str,
        image_ext: str,
    ) -> int:
        target_ext = f".{image_ext.lower()}"
        prefix = f"{model}__{dataset}__"
        known_methods = {entry.id for entry in method_catalog()}
        removed = 0
        images_dir = self.output_root / "images"
        if not images_dir.exists():
            return 0
        for path in images_dir.iterdir():
            if not (path.is_file() and path.name.startswith(prefix)):
                continue
            if path.suffix.lower() != target_ext or path.stem.split("__")[-1] not in known_methods:
                path.unlink()
                removed += 1
        return removed

    def write_catalogs(
        self,
        model_entries: list[ModelCatalogEntry],
        method_entries: list[MethodCatalogEntry] | None = None,
        run_base_urls: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        existing_models_payload = _read_json(self._models_catalog_path(), {"models": []})
        existing_models = {}
        if isinstance(existing_models_payload, dict):
            for item in existing_models_payload.get("models", []):
                if isinstance(item, dict) and "id" in item:
                    existing_models[str(item["id"])] = item
        for entry in model_entries:
            existing_models[entry.id] = entry.to_dict()

        sorted_models = [
            existing_models[model_id]
            for model_id in sorted(existing_models)
        ]
        _atomic_write_json(self._models_catalog_path(), {"models": sorted_models})

        methods = method_entries if method_entries is not None else method_catalog()
        _atomic_write_json(
            self._methods_catalog_path(),
            {"methods": [entry.to_dict() for entry in methods]},
        )
        self.refresh_manifest(run_base_urls)

    def refresh_manifest(
        self,
        run_base_urls: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        runs: dict[str, dict[str, dict[str, str]]] = {}
        datasets_by_model: dict[str, list[str]] = {}
        runs_root = self.output_root / "runs"
        if runs_root.exists():
            for model_dir in sorted(runs_root.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                runs[model] = {}
                datasets_by_model[model] = []
                for dataset_dir in sorted(model_dir.iterdir()):
                    if not dataset_dir.is_dir():
                        continue
                    dataset = dataset_dir.name
                    images_path = self._run_images_path(model, dataset)
                    summary_path = self._run_summary_path(model, dataset)
                    if not images_path.exists():
                        continue
                    datasets_by_model[model].append(dataset)
                    run = {
                        "images": str(images_path.relative_to(self.public_root)).replace("\\", "/"),
                        "summary": str(summary_path.relative_to(self.public_root)).replace("\\", "/"),
                    }
                    base_url = run_base_urls.get((model, dataset)) if run_base_urls else None
                    if base_url:
                        run["base_url"] = base_url
                    runs[model][dataset] = run

        manifest = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "attribution_encoding": dict(config.ATTRIBUTION_ENCODING),
            "catalogs": {
                "models": str(self._models_catalog_path().relative_to(self.public_root)).replace("\\", "/"),
                "methods": str(self._methods_catalog_path().relative_to(self.public_root)).replace("\\", "/"),
            },
            "models": sorted(runs),
            "datasets_by_model": {model: datasets_by_model[model] for model in sorted(datasets_by_model)},
            "runs": runs,
        }
        _atomic_write_json(self._manifest_path(), manifest)


    def _write_run_bundle(
        self,
        model: str,
        dataset: str,
        records: list[ImageRecord],
    ) -> None:
        payload = {
            "model": model,
            "dataset": dataset,
            "images": [record.to_dict() for record in records],
        }
        _atomic_write_json(self._run_images_path(model, dataset), payload)
        _atomic_write_json(self._run_summary_path(model, dataset), self._build_summary(model, dataset, records))
        self.refresh_manifest()

    def _build_summary(
        self,
        model: str,
        dataset: str,
        records: list[ImageRecord],
    ) -> dict[str, object]:
        bucket = MetricsBucket()
        class_ids = set()
        methods = set()
        for record in records:
            class_ids.add(record.class_id)
            methods.update(record.outputs)
            methods.update(record.attribution_failures)
            predicted_class_id = (
                record.prediction.predicted_class_id if record.prediction is not None else None
            )
            _record_prediction(bucket, record.class_id, predicted_class_id)
        return {
            "model": model,
            "dataset": dataset,
            "attribution_encoding": dict(config.ATTRIBUTION_ENCODING),
            "imageCount": len(records),
            "classCount": len(class_ids),
            "methodCount": len(methods),
            "metrics": _finalize_metrics_bucket(bucket),
        }

import json
import os
import tempfile
from pathlib import Path


def _create_metrics_bucket() -> dict:
    return {
        "total": 0,
        "correct": 0,
        "missingPredictions": 0,
        "perClass": {},
    }


def _get_per_class_bucket(bucket: dict, class_id: str) -> dict:
    if class_id not in bucket["perClass"]:
        bucket["perClass"][class_id] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    return bucket["perClass"][class_id]


def _record_prediction(bucket: dict, actual_class_id: str, predicted_class_id: str | None) -> None:
    actual_id = str(actual_class_id)
    predicted_id = None if predicted_class_id is None else str(predicted_class_id)
    actual_bucket = _get_per_class_bucket(bucket, actual_id)

    bucket["total"] += 1
    actual_bucket["support"] += 1

    if predicted_id is None or predicted_id == "":
        bucket["missingPredictions"] += 1
        actual_bucket["fn"] += 1
        return

    if predicted_id == actual_id:
        bucket["correct"] += 1
        actual_bucket["tp"] += 1
        return

    actual_bucket["fn"] += 1
    _get_per_class_bucket(bucket, predicted_id)["fp"] += 1


def _finalize_metrics_bucket(bucket: dict) -> dict:
    per_class_entries = list(bucket["perClass"].values())
    class_count = len(per_class_entries)

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for class_bucket in per_class_entries:
        tp = class_bucket["tp"]
        fp = class_bucket["fp"]
        fn = class_bucket["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    return {
        "total": bucket["total"],
        "correct": bucket["correct"],
        "incorrect": bucket["total"] - bucket["correct"],
        "missingPredictions": bucket["missingPredictions"],
        "classCount": class_count,
        "accuracy": (bucket["correct"] / bucket["total"]) if bucket["total"] > 0 else 0.0,
        "macroPrecision": (precision_sum / class_count) if class_count > 0 else 0.0,
        "macroRecall": (recall_sum / class_count) if class_count > 0 else 0.0,
        "macroF1": (f1_sum / class_count) if class_count > 0 else 0.0,
    }


def build_metrics(structure: dict) -> dict:
    model_buckets: dict = {}
    dataset_buckets: dict = {}

    for model, model_dict in structure.get("models", {}).items():
        model_buckets[model] = _create_metrics_bucket()
        dataset_buckets[model] = {}

        datasets = model_dict.get("datasets", {})
        for dataset, dataset_dict in datasets.items():
            dataset_buckets[model][dataset] = _create_metrics_bucket()
            classes = dataset_dict.get("classes", {})

            for class_id, class_dict in classes.items():
                images = class_dict.get("images", {})
                for image_dict in images.values():
                    prediction = image_dict.get("prediction")
                    predicted_class_id = None
                    if isinstance(prediction, dict):
                        predicted_class_id = prediction.get("predicted_class_id")
                    _record_prediction(model_buckets[model], str(class_id), predicted_class_id)
                    _record_prediction(dataset_buckets[model][dataset], str(class_id), predicted_class_id)

    return {
        "byModel": {
            model: _finalize_metrics_bucket(bucket)
            for model, bucket in model_buckets.items()
        },
        "byModelAndDataset": {
            model: {
                dataset: _finalize_metrics_bucket(bucket)
                for dataset, bucket in buckets.items()
            }
            for model, buckets in dataset_buckets.items()
        },
    }


def enrich_structure(structure: dict) -> dict:
    metrics = build_metrics(structure)
    by_model = metrics.get("byModel", {})
    by_model_and_dataset = metrics.get("byModelAndDataset", {})

    structure.pop("metrics", None)
    for model_name, model_dict in structure.get("models", {}).items():
        if not isinstance(model_dict, dict):
            continue

        model_dict["metrics"] = by_model.get(model_name, _finalize_metrics_bucket(_create_metrics_bucket()))
        dataset_metrics = by_model_and_dataset.get(model_name, {})

        datasets = model_dict.get("datasets", {})
        if not isinstance(datasets, dict):
            continue

        for dataset_name, dataset_dict in datasets.items():
            if not isinstance(dataset_dict, dict):
                continue
            dataset_dict["metrics"] = dataset_metrics.get(
                dataset_name,
                _finalize_metrics_bucket(_create_metrics_bucket()),
            )

    return structure


def build_structure(records: list[dict]) -> dict:
    structure: dict = {"models": {}}

    for record in records:
        model = record["model"]
        dataset = record["dataset"]
        class_id = record["class_id"]
        image_id = record["image_id"]

        model_dict = structure["models"].setdefault(model, {"datasets": {}})
        dataset_dict = model_dict["datasets"].setdefault(dataset, {"classes": {}})
        class_dict = dataset_dict["classes"].setdefault(class_id, {"images": {}})
        image_dict = class_dict["images"].setdefault(image_id, {"outputs": {}})

        if "method" in record:
            image_dict["outputs"][record["method"]] = record["url"]
        if "prediction" in record:
            image_dict["prediction"] = record["prediction"]

    return structure


def _merge_outputs(existing: dict, new: dict) -> dict:
    for key, value in new.items():
        if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
            _merge_outputs(existing[key], value)
        else:
            existing[key] = value
    return existing


def write_structure(structure: dict, output_file: str | Path) -> None:
    structure = enrich_structure(structure)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f"{output_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as temp_file:
            json.dump(structure, temp_file, indent=2, sort_keys=True)
        Path(temp_path).replace(output_path)
    except Exception:
        try:
            Path(temp_path).unlink(missing_ok=True)
        finally:
            raise


def export_to_json(records: list[dict], output_file: str | Path) -> None:
    new_structure = build_structure(records)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with output_path.open() as f:
            existing = json.load(f)
    else:
        existing = {"models": {}}

    merged = _merge_outputs(existing, new_structure)
    write_structure(merged, output_path)


def prune_stale_structure_outputs(
    structure: dict,
    model: str,
    dataset: str,
    image_ext: str,
) -> int:
    removed = 0
    model_dict = structure.get("models", {}).get(model, {})
    dataset_dict = model_dict.get("datasets", {}).get(dataset, {})
    classes = dataset_dict.get("classes", {})

    empty_images: list[str] = []
    empty_classes: list[str] = []

    for class_id, class_dict in classes.items():
        images = class_dict.get("images", {})
        empty_images.clear()

        for image_id, image_dict in images.items():
            outputs = image_dict.get("outputs", {})
            stale_methods = [
                method_name
                for method_name, output_url in outputs.items()
                if isinstance(output_url, str)
                and not output_url.lower().endswith(f".{image_ext.lower()}")
            ]
            for method_name in stale_methods:
                del outputs[method_name]
                removed += 1

            if not outputs:
                image_dict.pop("outputs", None)

            if not image_dict:
                empty_images.append(image_id)

        for image_id in empty_images:
            del images[image_id]

        if not images:
            empty_classes.append(class_id)

    for class_id in empty_classes:
        del classes[class_id]

    return removed

class OutputExporter:
    def build_structure(self, records: list[dict]) -> dict:
        return build_structure(records)

    def build_metrics(self, structure: dict) -> dict:
        return build_metrics(structure)

    def export_to_json(self, records: list[dict], output_file: str | Path) -> None:
        export_to_json(records, output_file)

    def write_structure(self, structure: dict, output_file: str | Path) -> None:
        write_structure(structure, output_file)

    def prune_stale_structure_outputs(
        self,
        structure: dict,
        model: str,
        dataset: str,
        image_ext: str,
    ) -> int:
        return prune_stale_structure_outputs(structure, model, dataset, image_ext)

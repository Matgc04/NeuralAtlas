import json
import os
import tempfile
from pathlib import Path


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

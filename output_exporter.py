import json
import os
import tempfile
from pathlib import Path


def build_structure(records: list[dict[str, str]]) -> dict:
    structure: dict = {"models": {}}

    for record in records:
        model = record["model"]
        dataset = record["dataset"]
        class_id = record["class_id"]
        image_id = record["image_id"]
        method = record["method"]
        url = record["url"]

        model_dict = structure["models"].setdefault(model, {"datasets": {}})
        dataset_dict = model_dict["datasets"].setdefault(dataset, {"classes": {}})
        class_dict = dataset_dict["classes"].setdefault(class_id, {"images": {}})
        image_dict = class_dict["images"].setdefault(image_id, {"outputs": {}})
        image_dict["outputs"][method] = url

    return structure


def _merge_outputs(existing: dict, new: dict) -> dict:
    for key, value in new.items():
        if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
            _merge_outputs(existing[key], value)
        else:
            existing[key] = value
    return existing


def export_to_json(records: list[dict[str, str]], output_file: str | Path) -> None:
    new_structure = build_structure(records)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with output_path.open() as f:
            existing = json.load(f)
    else:
        existing = {"models": {}}

    merged = _merge_outputs(existing, new_structure)

    fd, temp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f"{output_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as temp_file:
            json.dump(merged, temp_file, indent=2, sort_keys=True)
        Path(temp_path).replace(output_path)
    except Exception:
        try:
            Path(temp_path).unlink(missing_ok=True)
        finally:
            raise


class OutputExporter:
    def build_structure(self, records: list[dict[str, str]]) -> dict:
        return build_structure(records)

    def export_to_json(self, records: list[dict[str, str]], output_file: str | Path) -> None:
        export_to_json(records, output_file)

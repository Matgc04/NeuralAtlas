from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from torchvision import datasets


def load_sample_selection(
    selection_path: Path,
    dataset: str,
) -> list[tuple[str, str, str]]:
    """Load generic (class_id, image_id, source_filename) sample selectors."""
    try:
        payload = json.loads(selection_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read sample selection {selection_path}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported sample selection schema in {selection_path}")
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"Sample selection dataset is {payload.get('dataset')!r}, expected {dataset!r}"
        )
    entries = payload.get("samples", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid samples list in {selection_path}")

    samples = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Invalid sample entry in {selection_path}")
        try:
            sample = (
                str(entry["class_id"]),
                str(entry["image_id"]),
                str(entry["source_filename"]),
            )
        except KeyError as error:
            raise ValueError(f"Incomplete sample entry in {selection_path}") from error
        samples.append(sample)
    if len(samples) != len(set(samples)):
        raise ValueError(f"Duplicate sample entries in {selection_path}")
    return samples


def resolve_sample_indices(
    data: datasets.ImageFolder,
    requested: list[tuple[str, str, str]],
) -> list[int]:
    """Resolve a sparse selection against the exact files currently on disk."""
    counters: dict[str, int] = {}
    available: dict[tuple[str, str], tuple[int, str]] = {}
    for index, (path, target) in enumerate(data.samples):
        class_id = data.classes[target]
        image_number = counters.get(class_id, 0)
        counters[class_id] = image_number + 1
        available[(class_id, Path(path).name)] = (index, str(image_number))

    indices = []
    for class_id, expected_image_id, source_filename in requested:
        resolved = available.get((class_id, source_filename))
        if resolved is None:
            raise ValueError(
                f"Selected sample is absent from the dataset: {class_id}/{source_filename}"
            )
        index, actual_image_id = resolved
        if actual_image_id != expected_image_id:
            raise ValueError(
                f"Selected sample {class_id}/{source_filename} has image_id "
                f"{actual_image_id}, expected {expected_image_id}"
            )
        indices.append(index)
    return indices

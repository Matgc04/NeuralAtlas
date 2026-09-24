"""Dataset descriptors: what a dataset under `public/<id>/` holds and what its classes mean.

A dataset is a folder with a `dataset.json` next to an ImageFolder tree:

    public/<id>/dataset.json
    public/<id>/<images_dir>/<class_folder>/<image>

The descriptor names the label space the classes live in -- the output space of the
models it can be run against, `public/label_spaces/<label_space>.json` -- and maps each
class folder to its index in that space. `classes` may be left out when the folders are
already named after their index ("0", "207", ...). Bringing images of known classes
(e.g. ImageNet) only needs a descriptor; bringing new classes needs a new label space
and a model that predicts it.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from backend import config

DESCRIPTOR_NAME = "dataset.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelSpace:
    id: str
    title: str
    labels: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"{config.LABEL_SPACES_DIR.name}/{self.id}.json"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    id: str
    title: str
    label_space: LabelSpace
    images_dir: str
    root: Path
    classes: Mapping[str, int] = field(default_factory=dict)

    @property
    def images_path(self) -> Path:
        return self.root / self.images_dir

    def target(self, class_id: str) -> int:
        """Index of a class folder in the label space."""
        if self.classes:
            if class_id not in self.classes:
                raise ValueError(
                    f"Class folder {class_id!r} of dataset {self.id!r} is not declared in its `classes`."
                )
            return self.classes[class_id]
        if not class_id.isdigit() or int(class_id) >= len(self.label_space.labels):
            raise ValueError(
                f"Class folder {class_id!r} of dataset {self.id!r} is not an index of label space "
                f"{self.label_space.id!r}; map it in the `classes` of {DESCRIPTOR_NAME}."
            )
        return int(class_id)

    def label(self, class_id: str) -> str:
        return self.label_space.labels[self.target(class_id)]

    def short_label(self, class_id: str) -> str:
        """The label up to its first comma: ImageNet labels list synonyms after the main name."""
        return self.label(class_id).split(",")[0].strip()

    def image_url(self, class_id: str, filename: str) -> str:
        return f"/{self.id}/{self.images_dir}/{class_id}/{filename}"

    def descriptor(self) -> dict[str, object]:
        """The `dataset.json` payload that loads back into this spec."""
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "title": self.title,
            "label_space": self.label_space.id,
            "images_dir": self.images_dir,
        }
        if self.classes:
            payload["classes"] = dict(sorted(self.classes.items()))
        return payload

    def catalog_entry(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "label_space": self.label_space.id,
            "labels": self.label_space.url,
            "images_dir": self.images_dir,
            "classes": dict(sorted(self.classes.items())),
        }


def _read(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must hold a JSON object.")
    return payload


def load_label_space(label_space_id: str, public_root: Path = config.BASE_PUBLIC_DIR) -> LabelSpace:
    path = public_root / config.LABEL_SPACES_DIR.name / f"{label_space_id}.json"
    payload = _read(path)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise SystemExit(f"{path}: `labels` must be a non-empty list of strings, indexed by model output.")
    return LabelSpace(
        id=label_space_id,
        title=str(payload.get("title", label_space_id)),
        labels=tuple(labels),
    )


def load_dataset(dataset_id: str, public_root: Path = config.BASE_PUBLIC_DIR) -> DatasetSpec:
    root = public_root / dataset_id
    path = root / DESCRIPTOR_NAME
    if not path.is_file():
        raise SystemExit(f"Dataset descriptor not found: {path}")
    payload = _read(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"{path}: unsupported schema_version {payload.get('schema_version')!r}.")
    label_space_id = payload.get("label_space")
    if not isinstance(label_space_id, str) or not label_space_id:
        raise SystemExit(f"{path}: `label_space` is required.")
    label_space = load_label_space(label_space_id, public_root)

    raw_classes = payload.get("classes", {})
    if not isinstance(raw_classes, dict):
        raise SystemExit(f"{path}: `classes` must map class folder -> label space index.")
    classes: dict[str, int] = {}
    for folder, target in raw_classes.items():
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < len(label_space.labels):
            raise SystemExit(
                f"{path}: class {folder!r} maps to {target!r}, outside label space {label_space_id!r} "
                f"(0..{len(label_space.labels) - 1})."
            )
        classes[str(folder)] = target

    images_dir = payload.get("images_dir")
    if not isinstance(images_dir, str) or not images_dir.strip("/."):
        raise SystemExit(f"{path}: `images_dir` must name the ImageFolder root, e.g. \"val\".")

    return DatasetSpec(
        id=dataset_id,
        title=str(payload.get("title", dataset_id)),
        label_space=label_space,
        images_dir=images_dir.strip("/"),
        root=root,
        classes=classes,
    )


def dataset_ids(public_root: Path = config.BASE_PUBLIC_DIR) -> list[str]:
    """Every dataset published under `public_root`, i.e. every folder with a descriptor."""
    return sorted(path.parent.name for path in public_root.glob(f"*/{DESCRIPTOR_NAME}"))

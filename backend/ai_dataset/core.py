"""Data shapes, role abstractions, prompt/schema and shared IO helpers."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAPTION_PROMPT = """You are describing a real photograph so a text-to-image model can recreate it as faithfully as possible.
The main subject is a {label}.

Rules:
- Describe ONLY what is visibly present. Do not guess context or invent details that are not in the image.
- Preserve the exact number of subjects and their colors, materials, pose and orientation.
- Preserve the camera viewpoint and how the subject is framed and cropped.
- It is a realistic photograph: do not stylize, beautify, or add artistic effects.
- regeneration_prompt must be a single literal, detailed paragraph combining the other fields so the photo can be regenerated faithfully.
- Do not mention ImageNet, datasets, captions, or class labels.
"""

# Gemini structured-output JSON Schema. type is lowercase; propertyOrdering forces
# the model to emit the descriptive fields before synthesising regeneration_prompt last.
CAPTION_FIELDS = [
    "main_object",
    "count",
    "pose",
    "scene",
    "background",
    "viewpoint",
    "framing",
    "visual_attributes",
    "style",
    "regeneration_prompt",
]
CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        name: ({"type": "array", "items": {"type": "string"}} if name == "visual_attributes" else {"type": "string"})
        for name in CAPTION_FIELDS
    },
    "required": CAPTION_FIELDS,
    "propertyOrdering": CAPTION_FIELDS,
}


# data shapes


@dataclass
class Caption:
    main_object: str
    count: str
    pose: str
    scene: str
    background: str
    viewpoint: str
    framing: str
    visual_attributes: list[str]
    style: str
    regeneration_prompt: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Caption":
        fields = {name: str(data.get(name, "")) for name in CAPTION_FIELDS}
        fields["visual_attributes"] = [str(item) for item in data.get("visual_attributes", [])]
        return cls(**fields)


@dataclass
class GeneratedImage:
    data: bytes
    mime_type: str

    @property
    def extension(self) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(self.mime_type, mimetypes.guess_extension(self.mime_type) or ".png")


# roles


class Captioner(ABC):
    """Turns an image into a structured, reconstruction-oriented caption."""

    name: str
    model: str

    @abstractmethod
    def caption(self, image_bytes: bytes, mime_type: str, label: str) -> tuple[Caption, str]:
        """Return the structured caption and the raw model text it was parsed from."""


class ImageGenerator(ABC):
    """Turns a prompt into an image."""

    name: str
    model: str

    @abstractmethod
    def generate_image(self, prompt: str) -> GeneratedImage:
        ...

    def describe(self) -> dict[str, Any]:
        """Provider metadata persisted into records and the manifest.

        Subclasses override the fields that apply to them; unused fields stay None
        so every record/manifest carries the same key set regardless of provider.
        """
        return {
            "image_provider": self.name,
            "image_model": self.model,
            "image_size": None,
            "image_width": None,
            "image_height": None,
            "image_steps": None,
        }


# helpers


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def load_labels(path: Path) -> dict[str, str]:
    """Map class_id -> primary class name (first synonym) from an id2label file."""
    if not path.exists():
        return {}
    raw: dict[str, Any] = json.loads(path.read_text())
    return {str(key): str(value).split(",")[0].strip() for key, value in raw.items()}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    if default is not None:
        return default
    raise FileNotFoundError(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def loads_fenced_json(text: str) -> Any:
    """Parse JSON that an LLM may have wrapped in a markdown code fence."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean)


def sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

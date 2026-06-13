#!/usr/bin/env python3
"""Generate an AI-regenerated dataset paired 1:1 with a source ``public/<dataset>`` tree.

For every source image we (1) caption it with a vision model, anchored on the known
ImageNet class label, then (2) feed the resulting ``regeneration_prompt`` to an image
model and save the result under a parallel ``public/<target>`` tree with the same
class/index layout. The caption prompt optimises for *faithful reconstruction* so the
regenerated image can be compared against its source (paired mode).

Captioning and image generation are separate roles, each with its own ``--*-provider``,
so they can be mixed (e.g. caption with one backend, generate with another). Per image we
caption, persist the caption, then generate — a generation failure never loses the caption.
Only Gemini ships today.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
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


# gemini


class GeminiClient:
    """Shared low-level access to the Gemini generateContent endpoint."""

    def __init__(self, api_key: str, api_version: str) -> None:
        self.api_key = api_key
        self.api_version = api_version

    def generate_content(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/{self.api_version}/models/{model}:generateContent"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc


class GeminiCaptioner(Captioner):
    name = "gemini"

    def __init__(self, client: GeminiClient, model: str) -> None:
        self.client = client
        self.model = model

    def caption(self, image_bytes: bytes, mime_type: str, label: str) -> tuple[Caption, str]:
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": _b64(image_bytes)}},
                    {"text": CAPTION_PROMPT.format(label=label or "object")},
                ],
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": CAPTION_SCHEMA,
                "temperature": 0,
            },
        }
        response = self.client.generate_content(self.model, payload)
        raw_text = _extract_text(response)
        return Caption.from_dict(_loads_json(raw_text)), raw_text


class GeminiImageGenerator(ImageGenerator):
    name = "gemini"

    def __init__(self, client: GeminiClient, model: str, image_size: str = "", aspect_ratio: str = "") -> None:
        self.client = client
        self.model = model
        self.image_size = image_size  # "512" (3.1 only), "1K", "2K", "4K"; cheaper tiers cost less per image
        self.aspect_ratio = aspect_ratio

    def generate_image(self, prompt: str) -> GeneratedImage:
        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        image_config = {}
        if self.image_size:
            image_config["imageSize"] = self.image_size
        if self.aspect_ratio:
            image_config["aspectRatio"] = self.aspect_ratio
        if image_config:
            payload["generationConfig"] = {"imageConfig": image_config}
        response = self.client.generate_content(self.model, payload)
        inline = _extract_image(response)
        return GeneratedImage(data=base64.b64decode(inline["data"]), mime_type=inline["mime_type"])


def _gemini_client() -> GeminiClient:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    return GeminiClient(api_key, os.getenv("GEMINI_API_VERSION", "v1beta"))


def build_captioner(name: str, args: argparse.Namespace) -> Captioner:
    if name == "gemini":
        return GeminiCaptioner(_gemini_client(), os.getenv("GEMINI_CAPTION_MODEL", args.caption_model))
    raise SystemExit(f"unknown caption provider: {name}")


def build_image_generator(name: str, args: argparse.Namespace) -> ImageGenerator:
    if name == "gemini":
        return GeminiImageGenerator(
            _gemini_client(),
            os.getenv("GEMINI_IMAGE_MODEL", args.image_model),
            image_size=args.image_size,
            aspect_ratio=args.aspect_ratio,
        )
    raise SystemExit(f"unknown image provider: {name}")


# generator


class Generator:
    def __init__(
        self,
        args: argparse.Namespace,
        labels: dict[str, str],
        captioner: Captioner,
        image_generator: ImageGenerator,
    ) -> None:
        self.args = args
        self.labels = labels
        self.captioner = captioner
        self.image_generator = image_generator

        public_dir = Path(args.public_dir)
        self.source_dir = public_dir / args.source
        self.target_dir = public_dir / args.target
        self.structure_path = self.target_dir / f"{args.target}_structure.json"
        self.captions_path = self.target_dir / "captions.json"
        self.manifest_path = self.target_dir / "manifest.json"

        self.source_structure: dict[str, list[str]] = read_json(self.source_dir / f"{args.source}_structure.json")
        self.captions: dict[str, Any] = read_json(self.captions_path, default={"images": []})
        self.structure: dict[str, list[Any]] = read_json(self.structure_path, default={})

    def run(self) -> None:
        processed = 0
        for class_id in sorted(self.source_structure, key=sort_key):
            self.structure.setdefault(class_id, [])
            for image_index, filename in enumerate(self.source_structure[class_id]):
                image_id = str(image_index)
                existing = self._find(class_id, image_id)
                if not self.args.force and existing and existing.get("generated_filename"):
                    continue
                if self.args.limit is not None and processed >= self.args.limit:
                    self._flush()
                    print(f"done: processed {processed}")
                    return
                try:
                    record = self._caption(class_id, image_id, filename, existing)
                    self._upsert(record)
                    self._flush()  # caption persisted before we spend an image generation
                    self._generate(record, image_index)
                    self._flush()
                except Exception as exc:
                    handle_error(self.args, f"{class_id}/{filename}: {exc}")
                    continue
                processed += 1
                print(f"{processed}: {class_id}/{filename} -> {record['generated_url']}")
                if self.args.sleep:
                    time.sleep(self.args.sleep)
        self._flush()
        print(f"done: processed {processed}")

    def _find(self, class_id: str, image_id: str) -> dict[str, Any] | None:
        for item in self.captions.get("images", []):
            if item.get("class_id") == class_id and item.get("image_id") == image_id:
                return item
        return None

    def _caption(self, class_id: str, image_id: str, filename: str, existing: dict[str, Any] | None) -> dict[str, Any]:
        if not self.args.force and existing and existing.get("generation_prompt"):
            return existing  # resume: caption already done, reuse it for generation

        source_image = self.source_dir / "val" / class_id / filename
        if not source_image.exists():
            raise FileNotFoundError(f"missing source image: {source_image}")

        mime_type = mimetypes.guess_type(source_image.name)[0] or "image/webp"
        label = self.labels.get(class_id, "")
        caption, raw_caption = self.captioner.caption(source_image.read_bytes(), mime_type, label)
        return {
            "class_id": class_id,
            "image_id": image_id,
            "label": label,
            "source_filename": filename,
            "source_url": f"/{self.args.source}/val/{class_id}/{filename}",
            "caption": asdict(caption),
            "raw_caption": raw_caption,
            "generation_prompt": caption.regeneration_prompt or raw_caption,
            "caption_provider": self.captioner.name,
            "caption_model": self.captioner.model,
            "captioned_at": now_iso(),
        }

    def _generate(self, record: dict[str, Any], image_index: int) -> None:
        class_id = record["class_id"]
        image = self.image_generator.generate_image(record["generation_prompt"])
        generated_filename = f"{Path(record['source_filename']).stem}__ai{image.extension}"
        output_dir = self.target_dir / "val" / class_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / generated_filename).write_bytes(image.data)

        record.update({
            "generated_filename": generated_filename,
            "generated_url": f"/{self.args.target}/val/{class_id}/{generated_filename}",
            "image_provider": self.image_generator.name,
            "image_model": self.image_generator.model,
            "image_size": self.args.image_size,
            "generated_at": now_iso(),
        })

        # Keep the target index aligned with the source so paired comparison stays 1:1.
        values = self.structure.setdefault(class_id, [])
        while len(values) <= image_index:
            values.append(None)
        values[image_index] = generated_filename

    def _upsert(self, record: dict[str, Any]) -> None:
        images: list[dict[str, Any]] = self.captions.setdefault("images", [])
        if record in images:
            return
        images[:] = [
            item for item in images
            if not (item.get("class_id") == record["class_id"] and item.get("image_id") == record["image_id"])
        ]
        images.append(record)

    def _flush(self) -> None:
        self.captions["dataset"] = self.args.target
        self.captions["source_dataset"] = self.args.source
        self.captions["mode"] = "paired"
        self.captions["updated_at"] = now_iso()
        self.captions["images"] = sorted(
            self.captions.get("images", []),
            key=lambda item: (sort_key(item["class_id"]), sort_key(item["image_id"])),
        )
        manifest = {
            "schema_version": 3,
            "dataset": self.args.target,
            "source_dataset": self.args.source,
            "mode": "paired",
            "caption_provider": self.captioner.name,
            "caption_model": self.captioner.model,
            "image_provider": self.image_generator.name,
            "image_model": self.image_generator.model,
            "image_size": self.args.image_size,
            "structure": f"{self.args.target}/{self.args.target}_structure.json",
            "captions": f"{self.args.target}/captions.json",
            "generated_at": self.captions["updated_at"],
        }
        write_json(self.structure_path, {key: self.structure[key] for key in sorted(self.structure, key=sort_key)})
        write_json(self.captions_path, self.captions)
        write_json(self.manifest_path, manifest)


# helpers


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _extract_parts(response: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for candidate in response.get("candidates", []):
        parts.extend(candidate.get("content", {}).get("parts", []))
    return parts


def _extract_text(response: dict[str, Any]) -> str:
    return "\n".join(part["text"] for part in _extract_parts(response) if part.get("text")).strip()


def _extract_image(response: dict[str, Any]) -> dict[str, str]:
    for part in _extract_parts(response):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {
                "mime_type": inline.get("mimeType") or inline.get("mime_type") or "image/png",
                "data": inline["data"],
            }
    raise RuntimeError("response did not include an image")


def _loads_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean)


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


def handle_error(args: argparse.Namespace, message: str) -> None:
    if args.continue_on_error:
        print(f"error: {message}")
        return
    raise SystemExit(message)


def sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an AI-regenerated dataset paired with a public/<dataset> tree.")
    parser.add_argument("--caption-provider", default="gemini", help="captioning backend")
    parser.add_argument("--image-provider", default="gemini", help="image-generation backend")
    parser.add_argument("--source", default="imagenet-pico")
    parser.add_argument("--target", default="imagenet-pico-ai")
    parser.add_argument("--public-dir", default="interpretability-viewer/public")
    parser.add_argument("--id2label", default="imagenet-mini/imagenet-1k-id2label.json",
                        help="path (under public-dir) mapping class_id -> label, used to anchor captions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="regenerate even if a caption already exists")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--caption-model", default="gemini-3.5-flash")
    parser.add_argument("--image-model", default="gemini-3.1-flash-image")
    parser.add_argument("--image-size", default="512",
                        help="output resolution tier: 512 (gemini-3.1 only), 1K, 2K, 4K. Smaller is cheaper per image")
    parser.add_argument("--aspect-ratio", default="1:1", help="output aspect ratio, e.g. 1:1, 4:3, 16:9")
    return parser.parse_args()


def main() -> None:
    load_env(Path(".env"))
    args = parse_args()
    labels = load_labels(Path(args.public_dir) / args.id2label)
    captioner = build_captioner(args.caption_provider, args)
    image_generator = build_image_generator(args.image_provider, args)
    Generator(args, labels, captioner, image_generator).run()


if __name__ == "__main__":
    main()

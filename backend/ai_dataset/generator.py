"""Paired-generation orchestration: caption every source image, then regenerate it."""
from __future__ import annotations

import argparse
import mimetypes
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder

from backend.datasets import DESCRIPTOR_NAME, DatasetSpec

from .core import Captioner, ImageGenerator, now_iso, read_json, sort_key, write_json


class Generator:
    def __init__(
        self,
        args: argparse.Namespace,
        source: DatasetSpec,
        captioner: Captioner,
        image_generator: ImageGenerator,
    ) -> None:
        self.args = args
        self.source = source
        self.captioner = captioner
        self.image_generator = image_generator
        self.only = self._parse_only(args.only)
        self.exclude = [self._parse_only(value) for value in (args.exclude or [])]

        self.target_dir = Path(args.public_dir) / args.target
        self.captions_path = self.target_dir / "captions.json"
        self.manifest_path = self.target_dir / "manifest.json"

        # Class folder -> filenames in ImageFolder order, so image ids match the pipeline's.
        self.source_images: dict[str, list[str]] = {}
        for path, _ in ImageFolder(str(source.images_path)).samples:
            self.source_images.setdefault(Path(path).parent.name, []).append(Path(path).name)
        self.captions: dict[str, Any] = read_json(self.captions_path, default={"images": []})

    def run(self) -> None:
        # A --only target is always (re)generated; --stage decides whether its caption
        # is reused (image) or redone (full). Without --only we keep the resume behaviour.
        forced = self.args.force or self.only is not None
        recaption = self.args.stage == "full"
        if self.only is not None:
            print(f"targeting {self.args.only} (stage={self.args.stage})")

        total = self._count_pending(forced)
        print(f"{total} image(s) to generate", flush=True)

        processed = matched = 0
        for class_id in sorted(self.source_images, key=sort_key):
            for image_index, filename in enumerate(self.source_images[class_id]):
                image_id = str(image_index)
                if not self._targeted(class_id, image_id) or self._excluded(class_id, image_id):
                    continue
                matched += 1
                existing = self._find(class_id, image_id)
                if not forced and existing and existing.get("generated_filename"):
                    continue
                if self.args.limit is not None and processed >= self.args.limit:
                    self._flush()
                    print(f"done: processed {processed}")
                    return
                prefix = f"[{processed + 1}/{total}] {class_id}/{filename}"
                started = time.monotonic()
                try:
                    record = self._caption(class_id, image_id, filename, existing, recaption, prefix)
                    self._upsert(record)
                    self._flush()  # caption persisted before we spend an image generation
                    self._generate(record, prefix)
                    self._flush()
                except Exception as exc:
                    self._handle_error(f"{class_id}/{filename}: {exc}")
                    continue
                processed += 1
                print(f"{prefix} done in {time.monotonic() - started:.1f}s -> {record['generated_url']}", flush=True)
                if self.args.sleep:
                    time.sleep(self.args.sleep)
        self._flush()
        if self.only is not None and matched == 0:
            print(f"warning: --only {self.args.only!r} matched no source image")
        print(f"done: processed {processed}")

    @staticmethod
    def _parse_only(value: str | None) -> tuple[str, str | None] | None:
        if value is None:
            return None
        class_id, _, index = value.partition("/")
        class_id = class_id.strip()
        if not class_id:
            raise SystemExit(f"invalid --only selector: {value!r} (use CLASS or CLASS/INDEX)")
        index = index.strip()
        return (class_id, index or None)

    def _count_pending(self, forced: bool) -> int:
        """How many images this run will actually generate — the progress denominator.

        Mirrors the run() filters (targeted, already-generated skip, --limit) so the
        ``[i/total]`` counter matches what gets processed.
        """
        pending = 0
        for class_id, filenames in self.source_images.items():
            for image_index in range(len(filenames)):
                if not self._targeted(class_id, str(image_index)) or self._excluded(class_id, str(image_index)):
                    continue
                existing = self._find(class_id, str(image_index))
                if not forced and existing and existing.get("generated_filename"):
                    continue
                pending += 1
                if self.args.limit is not None and pending >= self.args.limit:
                    return self.args.limit
        return pending

    @staticmethod
    def _matches(selector: tuple[str, str | None], class_id: str, image_id: str) -> bool:
        want_class, want_image = selector
        return class_id == want_class and (want_image is None or image_id == want_image)

    def _targeted(self, class_id: str, image_id: str) -> bool:
        return self.only is None or self._matches(self.only, class_id, image_id)

    def _excluded(self, class_id: str, image_id: str) -> bool:
        return any(self._matches(selector, class_id, image_id) for selector in self.exclude)

    def _find(self, class_id: str, image_id: str) -> dict[str, Any] | None:
        for item in self.captions.get("images", []):
            if item.get("class_id") == class_id and item.get("image_id") == image_id:
                return item
        return None

    def _caption(self, class_id: str, image_id: str, filename: str, existing: dict[str, Any] | None,
                 recaption: bool, prefix: str) -> dict[str, Any]:
        if not recaption and existing and existing.get("generation_prompt"):
            return existing  # reuse the saved caption (resume, or --stage image) for generation

        source_image = self.source.images_path / class_id / filename
        if not source_image.exists():
            raise FileNotFoundError(f"missing source image: {source_image}")

        mime_type = mimetypes.guess_type(source_image.name)[0] or "image/webp"
        label = self.source.short_label(class_id)
        print(f"{prefix} captioning...", flush=True)
        caption, raw_caption = self.captioner.caption(source_image.read_bytes(), mime_type, label)
        return {
            "class_id": class_id,
            "image_id": image_id,
            "label": label,
            "source_filename": filename,
            "source_url": self.source.image_url(class_id, filename),
            "caption": asdict(caption),
            "raw_caption": raw_caption,
            "generation_prompt": caption.regeneration_prompt or raw_caption,
            "caption_provider": self.captioner.name,
            "caption_model": self.captioner.model,
            "captioned_at": now_iso(),
        }

    def _generate(self, record: dict[str, Any], prefix: str) -> None:
        class_id = record["class_id"]
        print(f"{prefix} generating image...", flush=True)
        image = self.image_generator.generate_image(record["generation_prompt"])
        stem = f"{Path(record['source_filename']).stem}__ai"
        generated_filename = stem + image.extension
        output_dir = self.target_dir / self.source.images_dir / class_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Drop the previous file when regenerating into a different extension, else it orphans.
        for previous in output_dir.glob(f"{stem}.*"):
            previous.unlink()
        (output_dir / generated_filename).write_bytes(image.data)

        record.update({
            "generated_filename": generated_filename,
            "generated_url": f"/{self.args.target}/{self.source.images_dir}/{class_id}/{generated_filename}",
            **self.image_generator.describe(),
            "generated_at": now_iso(),
        })

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
            **self.image_generator.describe(),
            "captions": f"{self.args.target}/captions.json",
            "generated_at": self.captions["updated_at"],
        }
        write_json(self.captions_path, self.captions)
        write_json(self.manifest_path, manifest)
        # Paired images keep the source classes, so the target is labelled exactly like it; where
        # its images are hosted is its own, so an existing descriptor is left as is.
        if not (self.target_dir / DESCRIPTOR_NAME).exists():
            write_json(self.target_dir / DESCRIPTOR_NAME,
                       replace(self.source, title=self.args.target, images_base_url=None).descriptor())

    def _handle_error(self, message: str) -> None:
        if self.args.continue_on_error:
            print(f"error: {message}")
            return
        raise SystemExit(message)

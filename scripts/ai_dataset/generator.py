"""Paired-generation orchestration: caption every source image, then regenerate it."""
from __future__ import annotations

import argparse
import mimetypes
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .core import Captioner, ImageGenerator, now_iso, read_json, sort_key, write_json


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
        self.only = self._parse_only(args.only)
        self.exclude = [self._parse_only(value) for value in (args.exclude or [])]

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
        # A --only target is always (re)generated; --stage decides whether its caption
        # is reused (image) or redone (full). Without --only we keep the resume behaviour.
        forced = self.args.force or self.only is not None
        recaption = self.args.stage == "full"
        if self.only is not None:
            print(f"targeting {self.args.only} (stage={self.args.stage})")

        total = self._count_pending(forced)
        print(f"{total} image(s) to generate", flush=True)

        processed = matched = 0
        for class_id in sorted(self.source_structure, key=sort_key):
            self.structure.setdefault(class_id, [])
            for image_index, filename in enumerate(self.source_structure[class_id]):
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
                    self._generate(record, image_index, prefix)
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
        for class_id, filenames in self.source_structure.items():
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

        source_image = self.source_dir / "val" / class_id / filename
        if not source_image.exists():
            raise FileNotFoundError(f"missing source image: {source_image}")

        mime_type = mimetypes.guess_type(source_image.name)[0] or "image/webp"
        label = self.labels.get(class_id, "")
        print(f"{prefix} captioning...", flush=True)
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

    def _generate(self, record: dict[str, Any], image_index: int, prefix: str) -> None:
        class_id = record["class_id"]
        print(f"{prefix} generating image...", flush=True)
        image = self.image_generator.generate_image(record["generation_prompt"])
        generated_filename = f"{Path(record['source_filename']).stem}__ai{image.extension}"
        output_dir = self.target_dir / "val" / class_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Drop the previous file when regenerating into a different extension, else it orphans.
        slot = self.structure.get(class_id, [])
        previous = slot[image_index] if image_index < len(slot) else None
        if previous and previous != generated_filename:
            (output_dir / previous).unlink(missing_ok=True)
        (output_dir / generated_filename).write_bytes(image.data)

        record.update({
            "generated_filename": generated_filename,
            "generated_url": f"/{self.args.target}/val/{class_id}/{generated_filename}",
            **self.image_generator.describe(),
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
            **self.image_generator.describe(),
            "structure": f"{self.args.target}/{self.args.target}_structure.json",
            "captions": f"{self.args.target}/captions.json",
            "generated_at": self.captions["updated_at"],
        }
        write_json(self.structure_path, {key: self.structure[key] for key in sorted(self.structure, key=sort_key)})
        write_json(self.captions_path, self.captions)
        write_json(self.manifest_path, manifest)

    def _handle_error(self, message: str) -> None:
        if self.args.continue_on_error:
            print(f"error: {message}")
            return
        raise SystemExit(message)

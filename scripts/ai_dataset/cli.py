"""Command-line entry point: argument parsing, provider registry, ``main``."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .cloudflare import CloudflareWorkersAIImageGenerator, cloudflare_workers_ai_client
from .codex import codex_image_generator_from_env
from .core import Captioner, ImageGenerator, load_env, load_labels
from .gemini import GeminiCaptioner, GeminiImageGenerator, gemini_client
from .generator import Generator


def build_captioner(name: str, args: argparse.Namespace) -> Captioner:
    if name == "gemini":
        return GeminiCaptioner(gemini_client(), os.getenv("GEMINI_CAPTION_MODEL", args.caption_model))
    raise SystemExit(f"unknown caption provider: {name}")


def build_image_generator(name: str, args: argparse.Namespace) -> ImageGenerator:
    if name == "gemini":
        return GeminiImageGenerator(
            gemini_client(),
            os.getenv("GEMINI_IMAGE_MODEL", args.image_model),
            image_size=args.image_size,
            aspect_ratio=args.aspect_ratio,
        )
    if name in {"cloudflare", "cloudflare-workers-ai", "workers-ai"}:
        return CloudflareWorkersAIImageGenerator(
            cloudflare_workers_ai_client(),
            os.getenv("CLOUDFLARE_IMAGE_MODEL", args.cloudflare_image_model),
            width=args.cloudflare_width,
            height=args.cloudflare_height,
            steps=args.cloudflare_steps,
            seed=args.cloudflare_seed,
        )
    if name == "codex":
        return codex_image_generator_from_env(args.codex_image_model, args.codex_timeout)
    raise SystemExit(f"unknown image provider: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an AI-regenerated dataset paired with a public/<dataset> tree.")
    parser.add_argument("--caption-provider", default="gemini", help="captioning backend")
    parser.add_argument("--image-provider", default="cloudflare", help="image-generation backend")
    parser.add_argument("--source", default="imagenet-pico")
    parser.add_argument("--target", default="imagenet-pico-ai")
    parser.add_argument("--public-dir", default="interpretability-viewer/public")
    parser.add_argument("--id2label", default="imagenet-mini/imagenet-1k-id2label.json",
                        help="path (under public-dir) mapping class_id -> label, used to anchor captions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--force", action="store_true",
                        help="reprocess items that are already generated (otherwise they are skipped); "
                             "what gets redone is governed by --stage")
    parser.add_argument("--only", metavar="CLASS[/INDEX]",
                        help="regenerate just one target and force it: a class_id (e.g. 1) for the whole class, "
                             "or class_id/index (e.g. 1/0) for a single image")
    parser.add_argument("--exclude", metavar="CLASS[/INDEX]", action="append",
                        help="skip a class_id or class_id/index; repeatable (e.g. --exclude 1 --exclude 5/0)")
    parser.add_argument("--stage", choices=["image", "full"], default="image",
                        help="what to (re)generate per processed item: 'image' reuses the saved caption and only "
                             "regenerates the picture (cheap, no vision call); 'full' re-captions then regenerates")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--caption-model", default="gemini-3.5-flash")
    parser.add_argument("--image-model", default="gemini-3.1-flash-image")
    parser.add_argument("--image-size", default="512",
                        help="output resolution tier: 512 (gemini-3.1 only), 1K, 2K, 4K. Smaller is cheaper per image")
    parser.add_argument("--aspect-ratio", default="1:1", help="output aspect ratio, e.g. 1:1, 4:3, 16:9")
    parser.add_argument("--cloudflare-image-model", default="@cf/black-forest-labs/flux-2-klein-4b")
    parser.add_argument("--cloudflare-width", type=int, default=512)
    parser.add_argument("--cloudflare-height", type=int, default=512)
    parser.add_argument("--cloudflare-steps", type=int, default=25)
    parser.add_argument("--cloudflare-seed", type=int)
    parser.add_argument("--codex-image-model", default="",
                        help="optional Codex model override for image generation; empty uses Codex default")
    parser.add_argument("--codex-timeout", type=int, default=900,
                        help="seconds to wait for each Codex image generation subprocess")
    return parser.parse_args()


def main() -> None:
    load_env(Path(".env"))
    args = parse_args()
    labels = load_labels(Path(args.public_dir) / args.id2label)
    captioner = build_captioner(args.caption_provider, args)
    image_generator = build_image_generator(args.image_provider, args)
    Generator(args, labels, captioner, image_generator).run()

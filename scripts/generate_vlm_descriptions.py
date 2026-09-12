#!/usr/bin/env python
"""Describe attribution maps with a VLM served by llama-server.

Descriptions are stored as one resumable JSON shard per ImageNet class and
uploaded to the model's existing Hugging Face dataset repository.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import config  # noqa: E402
from backend.ai_dataset.core import load_env, load_labels, sort_key  # noqa: E402
from backend.hf import attributions_base_repo, model_repo_id  # noqa: E402
from backend.hf import with_retries as _with_retries  # noqa: E402
from backend.persistence import OutputRepository  # noqa: E402
from backend.records import ImageRecord  # noqa: E402
from backend.vlm import (  # noqa: E402
    OVERLAY_VERSION,
    PROMPT_VERSION,
    LlamaVlmClient,
    model_view,
    overlay,
    vlm_data_url,
)

DEFAULT_VLM_MODEL = "Qwen3-VL-8B-Instruct"
T = TypeVar("T")


def with_retries(label: str, action: Callable[[], T], attempts: int = 3) -> T:
    return _with_retries(
        label,
        action,
        attempts,
        base_delay=2,
        log=lambda message: print(message, flush=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="classifier model whose run should be described")
    parser.add_argument("--dataset", default="imagenet-pico")
    parser.add_argument("--repo", help="HF dataset repo; defaults to $HF_ATTRIBUTIONS_REPO-<model>")
    parser.add_argument("--server-url", default=os.getenv("VLM_SERVER_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--vlm-model", default=os.getenv("VLM_MODEL", DEFAULT_VLM_MODEL))
    parser.add_argument("--quantization", default=os.getenv("VLM_QUANTIZATION", "Q4_K_M"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", help="only describe these attribution methods")
    parser.add_argument("--only", metavar="CLASS[/IMAGE]", help="only describe one class or image")
    parser.add_argument("--limit", type=int, help="maximum number of new descriptions")
    parser.add_argument(
        "--commit-every",
        type=int,
        default=0,
        metavar="CLASSES",
        help="upload after this many modified class shards; 0 uploads once at the end",
    )
    parser.add_argument("--force", action="store_true", help="regenerate existing descriptions")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="write local shards without committing to HF",
    )
    parser.add_argument(
        "--id2label",
        default="imagenet-mini/imagenet-1k-id2label.json",
        help="label map relative to interpretability-viewer/public",
    )
    return parser.parse_args()


def read_run(api: Any, repo_id: str, revision: str, model: str, dataset: str) -> list[ImageRecord]:
    remote_path = f"runs/{model}/{dataset}/images.json"
    path = Path(
        api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=remote_path,
        )
    )
    payload = json.loads(path.read_text())
    if payload.get("model") != model or payload.get("dataset") != dataset:
        raise ValueError(f"Run identity does not match {repo_id}/{remote_path}")
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Run has no images list: {repo_id}/{remote_path}")
    return [
        ImageRecord.from_dict(model, dataset, image)
        for image in images
        if isinstance(image, dict)
    ]


def restore_shard(
    api: Any,
    repo_id: str,
    revision: str,
    repository: OutputRepository,
    model: str,
    dataset: str,
    class_id: str,
    remote: set[str],
) -> bool:
    """Restore the remote shard and report whether a newer local copy needs upload."""
    local_path = repository.vlm_descriptions_path(model, dataset, class_id)
    remote_path = local_path.relative_to(repository.output_root).as_posix()
    if remote_path not in remote:
        return local_path.exists()
    cached = Path(
        api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=remote_path,
        )
    )
    if local_path.exists():
        return local_path.read_bytes() != cached.read_bytes()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, local_path)
    return False


def original_path(record: ImageRecord) -> Path:
    if not record.original_url:
        raise ValueError("Image record has no original_url")
    path = REPO_ROOT / config.BASE_PUBLIC_DIR / record.original_url.lstrip("/")
    if not path.is_file():
        raise FileNotFoundError(f"Original image is not available locally: {path}")
    return path


def local_heatmap(output_url: str) -> Path:
    return REPO_ROOT / config.OUTPUT_IMAGES_DIR / Path(output_url).name


def remote_heatmap(dataset: str, class_id: str, output_url: str) -> str:
    return f"images/{dataset}/{class_id}/{Path(output_url).name}"


def prefetch(api: Any, repo_id: str, revision: str, filenames: list[str]) -> None:
    """Cache a class's heatmaps in parallel at the pinned commit."""
    if not filenames:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [
            pool.submit(
                api.hf_hub_download,
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=filename,
            )
            for filename in filenames
        ]:
            future.result()


def heatmap_path(
    api: Any,
    repo_id: str,
    revision: str,
    dataset: str,
    class_id: str,
    output_url: str,
) -> Path:
    local = local_heatmap(output_url)
    if local.is_file():
        return local
    return Path(
        api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=remote_heatmap(dataset, class_id, output_url),
        )
    )


def upload_shards(api: Any, repo_id: str, output_root: Path, paths: set[Path]) -> int:
    from huggingface_hub import CommitOperationAdd

    ordered_paths = sorted(paths)
    if not ordered_paths:
        return 0
    operations = [
        CommitOperationAdd(
            path_in_repo=path.relative_to(output_root).as_posix(),
            path_or_fileobj=path,
        )
        for path in ordered_paths
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"add VLM descriptions for {len(ordered_paths)} classes",
    )
    return len(ordered_paths)


def in_scope(record: ImageRecord, only: str | None) -> bool:
    if only is None:
        return True
    class_id, separator, image_id = only.partition("/")
    return record.class_id == class_id and (not separator or record.image_id == image_id)


def grouped_by_class(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_id].append(record)
    for images in grouped.values():
        images.sort(key=lambda record: sort_key(record.image_id))
    return grouped


def main() -> None:
    load_env(REPO_ROOT / ".env")
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.commit_every < 0:
        raise SystemExit("--commit-every cannot be negative")
    limit = args.limit or math.inf
    commit_every = args.commit_every or math.inf

    token = os.getenv("HF_TOKEN")
    if not args.no_upload and not token:
        raise SystemExit("HF_TOKEN is required unless --no-upload is used")

    from huggingface_hub import HfApi

    repo_id = args.repo or model_repo_id(attributions_base_repo(), args.model)
    api = HfApi(token=token)
    revision = str(api.repo_info(repo_id=repo_id, repo_type="dataset").sha)
    records = read_run(api, repo_id, revision, args.model, args.dataset)
    records = [record for record in records if in_scope(record, args.only)]
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))

    labels = load_labels(REPO_ROOT / config.BASE_PUBLIC_DIR / args.id2label)
    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)
    client = LlamaVlmClient(args.server_url, args.vlm_model, seed=args.seed)
    generator = {
        "model": args.vlm_model,
        "quantization": args.quantization,
        "prompt_version": PROMPT_VERSION,
        "overlay_version": OVERLAY_VERSION,
        "seed": args.seed,
    }
    selected_methods = set(args.methods) if args.methods else None

    print(
        f"Source: {repo_id}@{revision[:12]} | {args.model}/{args.dataset} | "
        f"VLM: {args.vlm_model} ({args.quantization})",
        flush=True,
    )

    generated = 0
    skipped = 0
    pending: set[Path] = set()

    def flush(threshold: float) -> None:
        if args.no_upload or len(pending) < threshold:
            return
        count = with_retries(
            "upload VLM shards",
            lambda: upload_shards(api, repo_id, repository.output_root, pending),
        )
        print(f"Uploaded {count} class shards", flush=True)
        pending.clear()

    groups = grouped_by_class(records)
    for class_id in sorted(groups, key=sort_key):
        if restore_shard(
            api, repo_id, revision, repository, args.model, args.dataset, class_id, remote
        ):
            pending.add(repository.vlm_descriptions_path(args.model, args.dataset, class_id))
        shard = repository.load_vlm_descriptions(args.model, args.dataset, class_id)
        if shard is not None and shard.get("generator") != generator and not args.force:
            raise SystemExit(
                f"Generator mismatch in class {class_id}; use --force to replace that shard"
            )

        # Plan the class before fetching its heatmaps in parallel.
        completed = shard["images"] if shard is not None else {}
        plan = {}
        for record in groups[class_id]:
            methods = set(record.outputs)
            if selected_methods is not None:
                methods &= selected_methods
            todo = methods if args.force else methods - completed.get(record.image_id, {}).keys()
            plan[record.image_id] = sorted(todo)
            skipped += len(methods) - len(todo)
        prefetch(
            api,
            repo_id,
            revision,
            [
                remote_heatmap(args.dataset, class_id, record.outputs[method])
                for record in groups[class_id]
                for method in plan[record.image_id]
                if not local_heatmap(record.outputs[method]).is_file()
            ],
        )

        for record in groups[class_id]:
            image_id = record.image_id
            methods = plan[image_id]
            if not methods:
                continue

            with Image.open(original_path(record)) as source:
                crop = model_view(source)
                original_url = vlm_data_url(crop)
                label = labels.get(class_id, class_id)
                for method in methods:
                    map_path = heatmap_path(
                        api, repo_id, revision, args.dataset, class_id, record.outputs[method]
                    )
                    with Image.open(map_path) as heatmap:
                        overlay_url = vlm_data_url(overlay(crop, heatmap))
                    print(f"{class_id}/{image_id} · {method}", flush=True)
                    description = with_retries(
                        f"{class_id}/{image_id}/{method}",
                        lambda: client.describe(original_url, overlay_url, label),
                    )
                    path = repository.upsert_vlm_description(
                        args.model,
                        args.dataset,
                        class_id,
                        image_id,
                        method,
                        generator,
                        description.to_dict(),
                        force=args.force,
                    )
                    pending.add(path)
                    generated += 1
                    if generated >= limit:
                        break
            if generated >= limit:
                break

        flush(commit_every)
        if generated >= limit:
            break

    flush(1)
    print(f"Done: generated={generated}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()

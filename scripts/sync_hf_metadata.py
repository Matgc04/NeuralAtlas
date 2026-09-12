#!/usr/bin/env python
"""Consolidate per-model Hugging Face run metadata into the local global index.

The sweep writes each model to a separate HF dataset so concurrent workers never
touch shared JSON. This script is the single-writer reduction step: it discovers
those datasets, downloads their run JSON at immutable commit revisions, then rebuilds
the catalogs and global manifest consumed by the GitHub Pages frontend.

Configuration comes from ``.env`` (see ``.env.example``):
  HF_TOKEN                 optional for public repos, required for private repos
  HF_ATTRIBUTIONS_REPO     repo prefix, default Matgc04/neuralatlas-attributions

Examples:
  uv run python scripts/sync_hf_metadata.py --dry-run
  uv run python scripts/sync_hf_metadata.py
  uv run python scripts/sync_hf_metadata.py --repos org/attrs-model_a org/attrs-model_b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import config  # noqa: E402
from backend.ai_dataset.core import load_env  # noqa: E402
from backend.hf import (  # noqa: E402
    DEFAULT_ATTRIBUTIONS_REPO,
    attributions_base_repo,
    model_repo_id,
)
from backend.persistence import ModelCatalogEntry, OutputRepository  # noqa: E402

RUN_FILENAMES = frozenset({"images.json", "summary.json"})


@dataclass(frozen=True, slots=True)
class RemoteRun:
    repo_id: str
    revision: str
    model: str
    dataset: str
    files: tuple[str, str]


def discover_repos(api, base_repo: str) -> list[str]:
    """Find model repos by exact prefix without hardcoding model names."""
    try:
        owner, repo_name = base_repo.split("/", 1)
    except ValueError as error:
        raise ValueError(f"Expected OWNER/REPO for --base-repo, got {base_repo!r}") from error

    prefix = model_repo_id(f"{owner}/{repo_name}", "")
    return sorted(
        str(info.id)
        for info in api.list_datasets(author=owner)
        if str(info.id).startswith(prefix)
    )


def parse_run_path(path: str) -> tuple[str, str, str] | None:
    parts = Path(path).parts
    if len(parts) != 4 or parts[0] != "runs" or parts[3] not in RUN_FILENAMES:
        return None
    _, model, dataset, filename = parts
    if model in {"", ".", ".."} or dataset in {"", ".", ".."}:
        return None
    return model, dataset, filename


def inspect_repo(api, repo_id: str) -> list[RemoteRun]:
    """Snapshot and enumerate complete runs without listing the large image tree."""
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    revision = str(info.sha)
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo="runs",
        recursive=True,
    ):
        parsed = parse_run_path(str(entry.path))
        if parsed is None:
            continue
        model, dataset, filename = parsed
        grouped.setdefault((model, dataset), {})[filename] = str(entry.path)

    runs = []
    for (model, dataset), files in sorted(grouped.items()):
        missing = RUN_FILENAMES - files.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"Incomplete run in {repo_id} at {revision}: {model}/{dataset} misses {names}")
        runs.append(
            RemoteRun(
                repo_id=repo_id,
                revision=revision,
                model=model,
                dataset=dataset,
                files=(files["images.json"], files["summary.json"]),
            )
        )
    return runs


def inspect_repos(api, repo_ids: Iterable[str]) -> list[RemoteRun]:
    runs: list[RemoteRun] = []
    owners: dict[tuple[str, str], str] = {}
    for repo_id in repo_ids:
        for run in inspect_repo(api, repo_id):
            key = (run.model, run.dataset)
            if key in owners:
                raise RuntimeError(
                    f"Run {run.model}/{run.dataset} exists in both {owners[key]} and {repo_id}"
                )
            owners[key] = repo_id
            runs.append(run)
    return runs


def validate_payload(path: Path, run: RemoteRun, filename: str) -> None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid JSON downloaded from {run.repo_id}: {filename}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {run.repo_id}: {filename}")
    if payload.get("model") != run.model or payload.get("dataset") != run.dataset:
        raise RuntimeError(
            f"Payload identity does not match runs/{run.model}/{run.dataset}: "
            f"{run.repo_id}/{filename}"
        )
    if filename == "images.json" and not isinstance(payload.get("images"), list):
        raise RuntimeError(f"Expected an images list in {run.repo_id}: {filename}")


def download_runs(api, runs: Iterable[RemoteRun], output_root: Path) -> None:
    """Validate every remote file before replacing any local run metadata."""
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neuralatlas-metadata-", dir=output_root.parent) as temp:
        staging = Path(temp)
        staged: list[tuple[Path, Path]] = []
        for run in runs:
            for remote_path in run.files:
                filename = Path(remote_path).name
                cached = Path(
                    api.hf_hub_download(
                        repo_id=run.repo_id,
                        repo_type="dataset",
                        revision=run.revision,
                        filename=remote_path,
                        force_download=True,
                    )
                )
                source = staging / remote_path
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, source)
                validate_payload(source, run, filename)
                staged.append((source, output_root / remote_path))

        for source, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)


def known_model_ids(output_root: Path) -> set[str]:
    path = output_root / "catalogs" / "models.json"
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return set()
    return {
        str(entry["id"])
        for entry in payload["models"]
        if isinstance(entry, dict) and "id" in entry
    }


def run_base_url(run: RemoteRun) -> str:
    return (
        f"https://huggingface.co/datasets/{run.repo_id}"
        f"/resolve/{run.revision}/images/{run.dataset}"
    )


def run_base_urls(runs: Iterable[RemoteRun]) -> dict[tuple[str, str], str]:
    return {(run.model, run.dataset): run_base_url(run) for run in runs}


def runs_to_download(runs: list[RemoteRun], output_root: Path) -> list[RemoteRun]:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest_runs = manifest.get("runs", {})
    return [
        run
        for run in runs
        if manifest_runs.get(run.model, {}).get(run.dataset, {}).get("base_url")
        != run_base_url(run)
        or any(not (output_root / path).is_file() for path in run.files)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-repo",
        help=f"HF repo prefix used for discovery (default: $HF_ATTRIBUTIONS_REPO or {DEFAULT_ATTRIBUTIONS_REPO})",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        help="Explicit HF dataset repos; skips automatic prefix discovery",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / config.OUTPUT_ROOT,
        help="Local outputs directory to update",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect and validate remote run layout without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(REPO_ROOT / ".env")

    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    base_repo = attributions_base_repo(args.base_repo)
    repos = sorted(set(args.repos or discover_repos(api, base_repo)))
    if not repos:
        raise SystemExit(f"No HF dataset repos found with prefix {base_repo}-")

    print(f"Found {len(repos)} repositories:")
    for repo_id in repos:
        print(f"  {repo_id}")

    runs = inspect_repos(api, repos)
    if not runs:
        raise SystemExit("The discovered repositories contain no complete runs/*/* metadata.")
    print(f"Found {len(runs)} runs:")
    for run in runs:
        print(f"  {run.model}/{run.dataset} <- {run.repo_id}@{run.revision[:12]}")

    output_root = args.output_root.resolve()
    pending_runs = runs_to_download(runs, output_root)
    print(
        f"Metadata: {len(pending_runs)} to download, "
        f"{len(runs) - len(pending_runs)} unchanged"
    )
    if args.dry_run:
        return

    if pending_runs:
        download_runs(api, pending_runs, output_root)

    existing_models = known_model_ids(output_root)
    new_models = sorted({run.model for run in runs} - existing_models)
    repository = OutputRepository(output_root)
    repository.write_catalogs(
        [ModelCatalogEntry(id=model, label=model, family=model) for model in new_models],
        run_base_urls=run_base_urls(runs),
    )
    print(f"Consolidated {len(runs)} runs into {output_root}")


if __name__ == "__main__":
    main()

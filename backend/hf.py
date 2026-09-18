"""Shared conventions for the Hugging Face attribution repos.

The sweep, the VLM captioner and the metadata reducer all talk to the same set
of per-model dataset repos. Keeping the naming scheme and the retry policy here
means `sync_hf_metadata` discovers exactly the repos `run_sweep` creates.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

DEFAULT_ATTRIBUTIONS_REPO = "Matgc04/neuralatlas-attributions"


def attributions_base_repo(override: str | None = None) -> str:
    """The repo prefix every per-model repo is derived from."""
    return override or os.getenv("HF_ATTRIBUTIONS_REPO", DEFAULT_ATTRIBUTIONS_REPO)


def model_repo_id(base_repo: str, model: str) -> str:
    """The dataset repo holding one model's run. Discovery relies on this shape."""
    return f"{base_repo}-{model}"


def with_retries(
    label: str,
    action: Callable[[], T],
    attempts: int = 4,
    base_delay: float = 15.0,
    log: Callable[[str], None] = print,
) -> T:
    """Retry a network action with exponential backoff; a multi-day run will hit blips."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log(f"warn: {label} failed ({error!r}); retrying in {delay:g}s")
            time.sleep(delay)
    raise AssertionError("unreachable")

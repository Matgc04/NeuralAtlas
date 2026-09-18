"""Codex CLI image generation provider.

Codex writes into an isolated temporary directory for each request. The dataset
pipeline still owns the final deterministic filename.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .core import GeneratedImage, ImageGenerator


class CodexImageGenerator(ImageGenerator):
    name = "codex"

    def __init__(
        self,
        command: str,
        model: str,
        timeout: int,
        image_size: str,
        output_format: str,
        output_name: str = "result.jpg",
    ) -> None:
        self.command = command
        self.model = model
        self.timeout = timeout
        self.image_size = image_size
        self.output_format = output_format
        self.output_name = output_name

    def generate_image(self, prompt: str) -> GeneratedImage:
        with tempfile.TemporaryDirectory(prefix="neuralatlas-codex-image-") as tmp:
            workdir = Path(tmp)
            output_path = workdir / self.output_name
            status_path = workdir / "codex-response.txt"
            command = [
                self.command,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--cd",
                str(workdir),
                "--output-last-message",
                str(status_path),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append(_image_prompt(prompt, output_path.name, self.image_size, self.output_format))

            result = subprocess.run(
                command,
                cwd=workdir,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(_format_failure(result))

            image_path = output_path if output_path.exists() else _find_generated_image(workdir)
            if image_path is None:
                details = result.stdout.strip()
                if not details and status_path.exists():
                    details = status_path.read_text(errors="replace").strip()
                raise RuntimeError(f"Codex did not write an image to {output_path.name}. {details}".strip())

            data = image_path.read_bytes()
            return GeneratedImage(data=data, mime_type=_sniff_mime(data, image_path))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "image_model": self.model or "codex-default"}


def codex_image_generator_from_env(model: str, timeout: int) -> CodexImageGenerator:
    return CodexImageGenerator(
        command=os.getenv("CODEX_BIN", "codex"),
        model=model,
        timeout=timeout,
        image_size=os.getenv("CODEX_IMAGE_SIZE", "1024x1024"),
        output_format=os.getenv("CODEX_IMAGE_FORMAT", "jpeg"),
        output_name=os.getenv("CODEX_IMAGE_OUTPUT", "result.jpg"),
    )


def _image_prompt(prompt: str, output_name: str, image_size: str, output_format: str) -> str:
    return f"""$imagegen Generate exactly one raster image from this prompt:

{prompt}

Use output size {image_size} and output format {output_format}.
Save the generated image in the current working directory as `{output_name}`.
Do not create additional images. Do not edit repository files. Your final response should only confirm the saved filename.
"""


def _find_generated_image(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _format_failure(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return f"Codex image generation failed with exit code {result.returncode}: {output}"


def _sniff_mime(data: bytes, path: Path) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")

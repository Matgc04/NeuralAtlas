"""Cloudflare Workers AI image generation.

The flux image models require ``multipart/form-data`` input (sending JSON gets a 400
``required properties at '/' are 'multipart'``), so we encode the fields as a form.
The reply comes back in one of two shapes: raw ``image/*`` bytes, or a JSON envelope
``{"result": {"image": "<base64>"}}``. We branch on the response Content-Type and sniff
the decoded bytes for the concrete mime.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

from .core import GeneratedImage, ImageGenerator


class CloudflareWorkersAIClient:
    """Low-level client for Cloudflare Workers AI model execution."""

    def __init__(self, account_id: str, api_token: str) -> None:
        self.account_id = account_id
        self.api_token = api_token

    def run_model(self, model: str, fields: dict[str, str]) -> tuple[bytes, str]:
        """POST the fields as multipart/form-data; return the raw body and Content-Type."""
        boundary = f"----NeuralAtlas{uuid.uuid4().hex}"
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model}"
        request = urllib.request.Request(
            url,
            data=_multipart_form_data(fields, boundary),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read(), response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cloudflare Workers AI HTTP {exc.code}: {body}") from exc


class CloudflareWorkersAIImageGenerator(ImageGenerator):
    name = "cloudflare-workers-ai"

    def __init__(
        self,
        client: CloudflareWorkersAIClient,
        model: str,
        width: int,
        height: int,
        steps: int | None,
        seed: int | None,
    ) -> None:
        self.client = client
        self.model = model
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed

    def generate_image(self, prompt: str) -> GeneratedImage:
        fields = {"prompt": prompt, "width": str(self.width), "height": str(self.height)}
        if self.steps is not None:
            fields["steps"] = str(self.steps)
        if self.seed is not None:
            fields["seed"] = str(self.seed)

        body, content_type = self.client.run_model(self.model, fields)
        data = body if content_type.startswith("image/") else base64.b64decode(_image_from_envelope(body))
        return GeneratedImage(data=data, mime_type=_sniff_mime(data))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "image_width": self.width, "image_height": self.height, "image_steps": self.steps}


def cloudflare_workers_ai_client() -> CloudflareWorkersAIClient:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise SystemExit("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN in .env")
    return CloudflareWorkersAIClient(account_id, api_token)


def _multipart_form_data(fields: dict[str, str], boundary: str) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
            value.encode("utf-8"),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def _image_from_envelope(body: bytes) -> str:
    envelope = json.loads(body.decode("utf-8"))
    if not envelope.get("success", True):
        raise RuntimeError(f"Cloudflare Workers AI error: {envelope.get('errors') or envelope}")
    image_b64 = envelope.get("result", {}).get("image")
    if not image_b64:
        raise RuntimeError(f"Cloudflare Workers AI response did not include an image: {envelope}")
    return image_b64


def _sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"

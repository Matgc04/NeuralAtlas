"""Gemini captioning and image generation via the generateContent endpoint."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .core import (
    CAPTION_PROMPT,
    CAPTION_SCHEMA,
    Caption,
    Captioner,
    GeneratedImage,
    ImageGenerator,
    _b64,
    loads_fenced_json,
)


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
        return Caption.from_dict(loads_fenced_json(raw_text)), raw_text


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

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "image_size": self.image_size}


def gemini_client() -> GeminiClient:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    return GeminiClient(api_key, os.getenv("GEMINI_API_VERSION", "v1beta"))


# response parsing


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

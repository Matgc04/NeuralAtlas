from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, cast, get_args

import numpy as np
from PIL import Image

from backend.ai_dataset.core import loads_fenced_json
from backend.models import MODEL_VIEW

PROMPT_VERSION = "attribution-description-v1"
OVERLAY_VERSION = "jet-gamma-v1"
VLM_IMAGE_SIZE = 512
Focus = Literal["subject", "background", "mixed", "unclear"]
FOCUS_VALUES = frozenset(get_args(Focus))

PROMPT = """Rojo y amarillo indican mayor magnitud de atribución, verde una
magnitud intermedia, y azul o transparente una magnitud menor.

Describí en una oración qué objetos o partes visibles coinciden con las zonas
de mayor atribución. No describas toda la imagen ni afirmes que una región
favorece la clase, porque el mapa usa valores absolutos. Si no es claro, decilo.

Clase objetivo: {target_label}.

Respondé en español como JSON:
{{"description":"...","focus":"subject|background|mixed|unclear"}}"""


@dataclass(frozen=True, slots=True)
class VlmDescription:
    description: str
    focus: Focus

    @classmethod
    def from_dict(cls, value: object) -> "VlmDescription":
        if not isinstance(value, dict):
            raise ValueError("VLM response must be a JSON object")
        description = value.get("description")
        focus = value.get("focus")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("VLM response has no description")
        if focus not in FOCUS_VALUES:
            raise ValueError(f"Invalid VLM focus: {focus!r}")
        return cls(description=description.strip(), focus=cast(Focus, focus))

    def to_dict(self) -> dict[str, str]:
        return {"description": self.description, "focus": self.focus}


def model_view(original: Image.Image) -> Image.Image:
    """The 224x224 crop the classifier was fed."""
    return MODEL_VIEW(original.convert("RGB"))


def vlm_data_url(image: Image.Image) -> str:
    """PNG data URL of `image` upscaled to the size the VLM is shown.

    The crop does not depend on the attribution method, so a caller describing
    several methods for one image encodes it once and reuses the string.
    """
    upscaled = image.resize(
        (VLM_IMAGE_SIZE, VLM_IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )
    buffer = io.BytesIO()
    upscaled.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class LlamaVlmClient:
    def __init__(
        self,
        server_url: str,
        model: str,
        *,
        seed: int = 0,
        timeout: int = 300,
    ) -> None:
        self.url = f"{server_url.rstrip('/')}/chat/completions"
        self.model = model
        self.seed = seed
        self.timeout = timeout

    def describe(
        self,
        original_url: str,
        overlay_url: str,
        target_label: str,
    ) -> VlmDescription:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Recorte original recibido por el clasificador:"},
                        {"type": "image_url", "image_url": {"url": original_url}},
                        {"type": "text", "text": "El mismo recorte con el mapa de atribución superpuesto:"},
                        {"type": "image_url", "image_url": {"url": overlay_url}},
                        {"type": "text", "text": PROMPT.format(target_label=target_label)},
                    ],
                }
            ],
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {error.code}: {details}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("llama-server response has no message content") from error
        if not isinstance(content, str):
            raise RuntimeError("llama-server message content is not text")
        return VlmDescription.from_dict(loads_fenced_json(content))


def overlay(crop: Image.Image, heatmap: Image.Image) -> Image.Image:
    """Composite a jet-colored attribution map over the classifier's crop."""
    values = np.asarray(
        heatmap.convert("L").resize(crop.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    colors = np.stack(
        [
            np.interp(values, [0, 0.35, 0.66, 0.89, 1], [0, 0, 1, 1, 0.5]),
            np.interp(values, [0, 0.125, 0.375, 0.64, 0.91, 1], [0, 0, 1, 1, 0, 0]),
            np.interp(values, [0, 0.11, 0.34, 0.65, 1], [0.5, 1, 1, 0, 0]),
        ],
        axis=-1,
    ) * 255.0
    # sqrt is the gamma of applyJet in App.jsx and 0.8 the default opacity of
    # its OverlayContext, so this is the overlay the viewer shows by default.
    alpha = (np.sqrt(values) * 0.8)[..., None]
    pixels = np.asarray(crop, dtype=np.float32)
    combined = pixels * (1.0 - alpha) + colors * alpha
    return Image.fromarray(np.clip(combined, 0, 255).round().astype(np.uint8), mode="RGB")



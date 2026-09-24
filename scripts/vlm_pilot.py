#!/usr/bin/env python
"""Does the VLM read the attribution map, or is it describing the photograph?

Describes the same image several times changing only the overlaid map. If the
description does not move, the map is decorative. Writes one JSONL line per
call; nothing is uploaded and no run artifact is touched.

    uv run python scripts/vlm_pilot.py --images-root /ruta/a/imagenet-pico/val
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.datasets import load_dataset  # noqa: E402
from backend.hf import attributions_base_repo, model_repo_id  # noqa: E402
from backend.records import ImageRecord  # noqa: E402
from backend.vlm import (  # noqa: E402
    OVERLAY_VERSION,
    PROMPT_VERSION,
    LlamaVlmClient,
    model_view,
    overlay,
    vlm_data_url,
)

CONDITIONS = ("sin_mapa", "correcto", "intercambiado", "invertido", "vacio",
              "uniforme", "gaussiana", "ruido")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-root", type=Path, required=True,
                        help="directorio 'val' de imagenet-pico (una carpeta por clase)")
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--dataset", default="imagenet-pico")
    parser.add_argument("--methods", nargs="+", default=["LayerGradCam", "Saliency"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--images", type=int, default=30, help="cuantas imagenes, de clases distintas")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--vlm-model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--quantization", default="Q4_K_M", help="solo se registra en el JSONL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.25, help="desvio de la gaussiana, en fraccion del lado")
    parser.add_argument("--out", type=Path, default=Path("vlm_pilot.jsonl"))
    return parser.parse_args()


def load_run(model: str, dataset: str) -> tuple[list[ImageRecord], object, str, str]:
    """Read the run from the public attributions repo. No token needed."""
    from huggingface_hub import HfApi

    repo_id = model_repo_id(attributions_base_repo(), model)
    api = HfApi()
    revision = str(api.repo_info(repo_id=repo_id, repo_type="dataset").sha)
    path = Path(api.hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                    filename=f"runs/{model}/{dataset}/images.json"))
    payload = json.loads(path.read_text())
    print(f"Corrida: {repo_id}@{revision[:12]}", flush=True)
    return ([ImageRecord.from_dict(model, dataset, image) for image in payload["images"]
             if isinstance(image, dict)], api, repo_id, revision)


def pick(records: list[ImageRecord], methods: list[str], count: int, seed: int) -> list[ImageRecord]:
    """One image per class, deterministic, only where every method has a map."""
    usable = [record for record in records if all(method in record.outputs for method in methods)]
    by_class: dict[str, ImageRecord] = {}
    for record in sorted(usable, key=lambda r: (r.class_id, r.image_id)):
        by_class.setdefault(record.class_id, record)
    chosen = sorted(by_class.values(), key=lambda r: r.class_id)
    random.Random(seed).shuffle(chosen)
    return chosen[:count]


def fetch_map(api, repo_id: str, revision: str, dataset: str, record: ImageRecord, method: str) -> Image.Image:
    name = Path(record.outputs[method]).name
    local = ROOT / config.OUTPUT_IMAGES_DIR / name
    path = local if local.is_file() else Path(api.hf_hub_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        filename=f"images/{dataset}/{record.class_id}/{name}"))
    with Image.open(path) as handle:
        return handle.convert("L").copy()


def source_image(images_root: Path, record: ImageRecord) -> Image.Image:
    path = images_root / record.class_id / record.source_filename
    if not path.is_file():
        matches = sorted((images_root / record.class_id).glob(Path(record.source_filename).stem + ".*"))
        if not matches:
            raise FileNotFoundError(path)
        path = matches[0]
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def synthetic(name: str, size: tuple[int, int], sigma: float, seed: int) -> Image.Image:
    width, height = size
    if name == "vacio":
        values = np.zeros((height, width), dtype=np.float32)
    elif name == "uniforme":
        values = np.full((height, width), 255.0, dtype=np.float32)
    elif name == "gaussiana":
        y = (np.arange(height) - (height - 1) / 2) / height
        x = (np.arange(width) - (width - 1) / 2) / width
        values = 255.0 * np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    elif name == "ruido":
        values = np.random.default_rng(seed).uniform(0, 255, (height, width)).astype(np.float32)
    else:
        raise ValueError(name)
    return Image.fromarray(values.round().astype(np.uint8), mode="L")


def main() -> None:
    args = parse_args()
    if not args.images_root.is_dir():
        raise SystemExit(f"No existe el directorio de imagenes: {args.images_root}")
    if args.out.exists():
        raise SystemExit(f"{args.out} ya existe; borralo o elegi otro --out")

    records, api, repo_id, revision = load_run(args.model, args.dataset)
    chosen = pick(records, args.methods, args.images, args.seed)
    if len(chosen) < 2:
        raise SystemExit("Hacen falta al menos dos imagenes para el control intercambiado")
    dataset = load_dataset(args.dataset, ROOT / config.BASE_PUBLIC_DIR)
    client = LlamaVlmClient(args.server_url, args.vlm_model, seed=args.seed)

    total = len(chosen) * len(args.methods) * len(args.conditions)
    print(f"{len(chosen)} imagenes x {len(args.methods)} metodos x "
          f"{len(args.conditions)} condiciones = {total} llamadas", flush=True)

    done = 0
    with args.out.open("w") as sink:
        sink.write(json.dumps({"type": "protocolo", "model": args.model, "dataset": args.dataset,
                               "vlm_model": args.vlm_model, "quantization": args.quantization,
                               "prompt_version": PROMPT_VERSION, "overlay_version": OVERLAY_VERSION,
                               "seed": args.seed, "sigma": args.sigma, "methods": args.methods,
                               "conditions": args.conditions, "source": f"{repo_id}@{revision}",
                               "images": [f"{r.class_id}/{r.image_id}" for r in chosen]},
                              ensure_ascii=False) + "\n")
        for index, record in enumerate(chosen):
            crop = model_view(source_image(args.images_root, record))
            clean_url = vlm_data_url(crop)
            # The label is always the real one, so only the map varies.
            label = dataset.short_label(record.class_id)
            other = chosen[(index + 1) % len(chosen)]
            for method in args.methods:
                real = fetch_map(api, repo_id, revision, args.dataset, record, method)
                for condition in args.conditions:
                    if condition == "sin_mapa":
                        url = clean_url
                    else:
                        if condition == "correcto":
                            heatmap = real
                        elif condition == "intercambiado":
                            heatmap = fetch_map(api, repo_id, revision, args.dataset, other, method)
                        elif condition == "invertido":
                            heatmap = Image.eval(real, lambda value: 255 - value)
                        else:
                            heatmap = synthetic(condition, real.size, args.sigma, args.seed + index)
                        url = vlm_data_url(overlay(crop, heatmap))

                    started = time.monotonic()
                    try:
                        result = client.describe(clean_url, url, label)
                        row = {"class_id": record.class_id, "image_id": record.image_id,
                               "label": label, "method": method, "condition": condition,
                               "description": result.description, "focus": result.focus,
                               "elapsed_s": round(time.monotonic() - started, 2)}
                    except Exception as error:  # a failed call must not lose the run
                        row = {"class_id": record.class_id, "image_id": record.image_id,
                               "label": label, "method": method, "condition": condition,
                               "error": repr(error)}
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    done += 1
                    print(f"[{done}/{total}] {record.class_id}/{record.image_id} · {method} · {condition}",
                          flush=True)

    summarize(args.out)
    print(f"Listo: {args.out}", flush=True)


def summarize(path: Path) -> None:
    """Agreement of `focus` between `correcto` and every other condition.

    A control that agrees with `correcto` almost always is a control the VLM
    could not tell apart, which is the whole question this pilot asks.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    scored = {(row["class_id"], row["image_id"], row["method"], row["condition"]): row["focus"]
              for row in rows if row.get("type") != "protocolo" and "focus" in row}
    keys = {key[:3] for key in scored}
    conditions = sorted({key[3] for key in scored} - {"correcto"})
    print("\nAcuerdo de focus contra 'correcto':")
    for condition in conditions:
        pairs = [(scored[key + ("correcto",)], scored[key + (condition,)]) for key in keys
                 if key + ("correcto",) in scored and key + (condition,) in scored]
        if not pairs:
            continue
        same = sum(1 for left, right in pairs if left == right)
        print(f"  {condition:<14} {same}/{len(pairs)}  ({100 * same / len(pairs):.0f} %)")
    print("Cuanto mas cerca del 100 %, menos esta leyendo el mapa.")


if __name__ == "__main__":
    main()

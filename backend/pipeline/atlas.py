from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric
from captum.attr import visualization as viz
from PIL import Image, ImageChops
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

from attr_config import AttributionConfig
from backend import config
from backend.records import ImageRecord, PredictionRecord


def build_output_filename(
    model_name: str,
    dataset_name: str,
    class_id: str,
    image_id: str,
    method_name: str,
    image_ext: str,
) -> str:
    return f"{model_name}__{dataset_name}__{class_id}__{image_id}__{method_name}.{image_ext}"


@dataclass(slots=True)
class AttributionRenderer:
    out_px: int = 224
    dpi: int = 112

    @property
    def side_inches(self) -> float:
        return self.out_px / self.dpi

    @staticmethod
    def _crop_uniform_background(image: Image.Image) -> Image.Image:
        background_color = image.getpixel((0, 0))
        background = Image.new(image.mode, image.size, background_color)
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        return image if bbox is None else image.crop(bbox)

    def _save_rendered_attr(
        self,
        fig: Figure,
        output_path: Path,
        image_ext: str,
    ) -> None:
        with BytesIO() as buffer:
            fig.savefig(
                buffer,
                format="png",
                dpi=self.dpi,
                bbox_inches=None,
                pad_inches=0,
            )
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB").copy()

        cropped_image = self._crop_uniform_background(image)
        save_kwargs: dict[str, Any] = {"format": image_ext.upper()}
        if image_ext == "webp":
            save_kwargs.update({"quality": 85, "method": 6, "lossless": False})
        elif image_ext in {"jpg", "jpeg"}:
            save_kwargs.update({"quality": 85, "optimize": True, "progressive": True})
        elif image_ext == "avif":
            save_kwargs.update({"quality": 75, "speed": 1})
        cropped_image.save(output_path, **save_kwargs)

    def render(
        self,
        attr: TensorOrTupleOfTensorsGeneric,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        class_id: str,
        image_id: str,
        method_name: str,
        image_ext: str,
        **kwargs: Any,
    ) -> str:
        if not isinstance(attr, torch.Tensor):
            raise TypeError(
                f"Expected attribution tensor for visualization, got {type(attr)}"
            )

        fig, ax = plt.subplots(figsize=(self.side_inches, self.side_inches), dpi=self.dpi)
        _ = viz.visualize_image_attr(
            attr.permute(1, 2, 0).detach().cpu().numpy(),
            plt_fig_axis=(fig, ax),
            use_pyplot=False,
            title=None,
            **kwargs,
        )
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        filename = build_output_filename(
            model_name=model_name,
            dataset_name=dataset_name,
            class_id=class_id,
            image_id=image_id,
            method_name=method_name,
            image_ext=image_ext,
        )
        self._save_rendered_attr(fig, output_dir / filename, image_ext)
        plt.close(fig)
        return f"{config.OUTPUT_IMAGES_BASE_URL}/{filename}"


class AtlasRunner:
    def __init__(
        self,
        model: Module,
        data: str | datasets.VisionDataset,
        interp_methods: list[AttributionConfig],
        renderer: AttributionRenderer | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        if isinstance(data, str):
            self.data = datasets.ImageFolder(data, transform=kwargs.get("transform", None))
        else:
            self.data = data
        self.interp_methods = interp_methods
        self.renderer = renderer or AttributionRenderer()

    @staticmethod
    def _serialize_prediction(model_output: object) -> PredictionRecord:
        if isinstance(model_output, torch.Tensor):
            logits = model_output
        elif (
            isinstance(model_output, (list, tuple))
            and len(model_output) > 0
            and isinstance(model_output[0], torch.Tensor)
        ):
            logits = model_output[0]
        else:
            raise TypeError(
                "Expected model output tensor (or tuple/list with tensor first item), "
                f"got {type(model_output)}."
            )
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        sample_logits = logits[0].detach().cpu()
        predicted_class_id = int(torch.argmax(sample_logits).item())
        confidence = float(torch.softmax(sample_logits, dim=0)[predicted_class_id].item())
        return PredictionRecord(
            predicted_class_id=str(predicted_class_id),
            confidence=confidence,
        )

    def _resolve_original_url(
        self,
        dataset_name: str,
        class_id: str,
        sample_index: int,
    ) -> str | None:
        sample_path: str | None = None
        if hasattr(self.data, "samples"):
            samples = getattr(self.data, "samples")
            if isinstance(samples, list) and sample_index < len(samples):
                sample = samples[sample_index]
                if isinstance(sample, tuple) and sample:
                    sample_path = str(sample[0])
        if sample_path is None:
            return None
        filename = Path(sample_path).name
        return f"/{dataset_name}/val/{class_id}/{filename}"

    def stream(
        self,
        num_samples: int,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        image_ext: str = "webp",
        **kwargs: Any,
    ) -> Iterator[ImageRecord]:
        
        self.model.eval()
        dataloader = DataLoader(self.data, batch_size=1, shuffle=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_ext = image_ext.lstrip(".").lower()

        class_image_counters: defaultdict[str, int] = defaultdict(int)

        with tqdm(total=num_samples, desc="Interpreting + Saving") as pbar:
            for sample_index, (inputs, target) in enumerate(dataloader):
                if sample_index >= num_samples:
                    break

                target = target.to(inputs.device)
                class_id = str(target.item())
                image_id = str(class_image_counters[class_id])
                original_url = self._resolve_original_url(dataset_name, class_id, sample_index)

                with torch.no_grad():
                    prediction = self._serialize_prediction(self.model(inputs))

                record = ImageRecord(
                    model=model_name,
                    dataset=dataset_name,
                    class_id=class_id,
                    image_id=image_id,
                    original_url=original_url,
                    prediction=prediction,
                    interpretability_metrics={},
                )

                inputs.requires_grad = True
                with tqdm(self.interp_methods, leave=False) as method_pbar:
                    for interp_method in method_pbar:
                        method_name = str(interp_method)
                        method_pbar.set_description(f"Attribution {method_name}")
                        attribution = interp_method.attribute(self.model, inputs, target)
                        if not isinstance(attribution, torch.Tensor):
                            raise TypeError(
                                "Streaming visualization requires tensor attribution output; "
                                f"got {type(attribution)} from {method_name}."
                            )
                        output_url = self.renderer.render(
                            attr=attribution[0],
                            output_dir=output_dir,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            class_id=class_id,
                            image_id=image_id,
                            method_name=method_name,
                            image_ext=image_ext,
                            **kwargs,
                        )
                        record.outputs[method_name] = output_url

                class_image_counters[class_id] += 1
                pbar.update(1)
                yield record

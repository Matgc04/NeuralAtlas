from attr_config import AttributionConfig

from io import BytesIO
from typing import Any, Iterator, List, TypeAlias, Union
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from captum.attr import visualization as viz
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric

from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

PredictionPayload: TypeAlias = dict[str, str | float]
ExportRecord: TypeAlias = dict[str, str | PredictionPayload]


class NeuralAtlas:
    OUT_PX = 224 
    DPI = 112
    SIDE_IN = OUT_PX / DPI

    def __init__(
        self,
        model: Module,
        data: Union[str, datasets.VisionDataset],
        interp_methods: List[AttributionConfig],
        **kwargs,
    ):
        self.model = model
        if isinstance(data, str):
            self.data = datasets.ImageFolder(
                data, transform=kwargs.get("transform", None)
            )
        else:
            self.data = data
        self.interp_methods = interp_methods

    @staticmethod
    def _crop_uniform_background(image: Image.Image) -> Image.Image:
        background_color = image.getpixel((0, 0))
        background = Image.new(image.mode, image.size, background_color)
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        if bbox is None:
            return image
        return image.crop(bbox)

    def _save_rendered_attr(
        self,
        fig: plt.Figure,
        output_path: Path,
        image_ext: str,
    ) -> None:
        with BytesIO() as buffer:
            fig.savefig(
                buffer,
                format="png",
                dpi=self.DPI,
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
            save_kwargs.update(
                {
                    "quality": 85,
                    "optimize": True,
                    "progressive": True,
                }
            )
        elif image_ext == "avif":
            save_kwargs.update(
                {
                    "quality": 75,
                    "speed": 1,
                }
            )

        cropped_image.save(output_path, **save_kwargs)

    def _save_single_attr(
        self,
        attr: TensorOrTupleOfTensorsGeneric,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        class_id: str,
        image_id: str,
        method_name: str,
        base_url: str,
        image_ext: str,
        prediction: PredictionPayload | None = None,
        **kwargs,
    ) -> ExportRecord:
        if not isinstance(attr, torch.Tensor):
            raise TypeError(
                f"Expected attribution tensor for visualization, got {type(attr)}"
            )

        fig, ax = plt.subplots(figsize=(self.SIDE_IN, self.SIDE_IN), dpi=self.DPI)
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

        filename = (
            f"{model_name}__{dataset_name}__{class_id}"
            f"__{image_id}__{method_name}.{image_ext}"
        )
        self._save_rendered_attr(
            fig=fig,
            output_path=output_dir / filename,
            image_ext=image_ext,
        )
        plt.close(fig)

        record: ExportRecord = {
            "model": model_name,
            "dataset": dataset_name,
            "class_id": class_id,
            "image_id": image_id,
            "method": method_name,
            "url": f"{base_url}/{filename}",
        }
        if prediction is not None:
            record["prediction"] = prediction
        return record

    @staticmethod
    def _serialize_prediction(model_output: object) -> PredictionPayload:
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
        return {
            "predicted_class_id": str(predicted_class_id),
            "confidence": confidence,
        }

    def interpret(
        self,
        num_samples: int = 1,
    ) -> dict:
        self.model.eval()
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=False)
        attributions = defaultdict(lambda: defaultdict(list))
        with tqdm(total=num_samples) as pbar:
            for i, (inputs, target) in enumerate(self.dataloader):
                if i == num_samples:
                    break
                target = target.to(inputs.device)
                inputs.requires_grad = True
                for interp_method in tqdm(self.interp_methods, leave=False):
                    pbar.set_description(f"Attribution {interp_method}")
                    attribution = interp_method.attribute(self.model, inputs, target)
                    attributions[str(target.item())][str(interp_method)].append(
                        attribution
                    )
                pbar.update(1)
        return attributions

    def visualize(
        self,
        attributions: dict,         
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        base_url: str,
        image_ext: str = "webp",
        predictions: dict[str, dict[str, PredictionPayload]] | None = None,
        **kwargs,
    ) -> list[ExportRecord]:
        total_num_attributions = sum(
            [len(attr) for target in attributions.values() for attr in target.values()]
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        base_url = base_url.rstrip("/")
        image_ext = image_ext.lstrip(".").lower()

        records: list[ExportRecord] = []

        with tqdm(total=total_num_attributions, desc="Plotting Attributions") as pbar:
            for target, interp_methods in attributions.items():
                for interp_method, attr in interp_methods.items():
                    method_name = str(interp_method)
                    attr = torch.cat(attr, dim=0)
                    for i, single_attr in enumerate(attr):
                        records.append(
                            self._save_single_attr(
                                attr=single_attr,
                                output_dir=output_dir,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                class_id=str(target),
                                image_id=str(i),
                                method_name=method_name,
                                base_url=base_url,
                                image_ext=image_ext,
                                prediction=(
                                    predictions.get(str(target), {}).get(str(i))
                                    if predictions
                                    else None
                                ),
                                **kwargs,
                            )
                        )
                        pbar.update(1)
        return records

    def interpret_and_visualize_stream(
        self,
        num_samples: int,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        base_url: str,
        image_ext: str = "webp",
        **kwargs,
    ) -> Iterator[list[ExportRecord]]:
        self.model.eval()
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=False)

        output_dir.mkdir(parents=True, exist_ok=True)
        base_url = base_url.rstrip("/")
        image_ext = image_ext.lstrip(".").lower()

        class_image_counters: defaultdict[str, int] = defaultdict(int)

        with tqdm(total=num_samples, desc="Interpreting + Saving") as pbar:
            for sample_index, (inputs, target) in enumerate(self.dataloader):
                if sample_index >= num_samples:
                    break

                target = target.to(inputs.device)

                class_id = str(target.item())
                image_id = str(class_image_counters[class_id])

                with torch.no_grad():
                    prediction = self._serialize_prediction(self.model(inputs))

                image_records: list[ExportRecord] = [
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "class_id": class_id,
                        "image_id": image_id,
                        "prediction": prediction,
                    }
                ]

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

                        image_records.append(
                            self._save_single_attr(
                                attr=attribution[0],
                                output_dir=output_dir,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                class_id=class_id,
                                image_id=image_id,
                                method_name=method_name,
                                base_url=base_url,
                                image_ext=image_ext,
                                prediction=prediction,
                                **kwargs,
                            )
                        )

                class_image_counters[class_id] += 1
                pbar.update(1)
                yield image_records

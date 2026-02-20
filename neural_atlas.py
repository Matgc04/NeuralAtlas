from attr_config import AttributionConfig

from typing import Iterator, List, Union
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from captum.attr import visualization as viz
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric

from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


class NeuralAtlas:
    OUT_PX = 512
    DPI = 128
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
        **kwargs,
    ) -> dict[str, str]:
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

        filename = (
            f"{model_name}__{dataset_name}__{class_id}"
            f"__{image_id}__{method_name}.{image_ext}"
        )
        save_kwargs = {
            "format": image_ext,
            "dpi": self.DPI,
            "bbox_inches": "tight",
            "pad_inches": 0,
        }
        if image_ext in {"webp", "jpg", "jpeg"}:
            save_kwargs["pil_kwargs"] = {"quality": 95}

        fig.savefig(output_dir / filename, **save_kwargs)
        plt.close(fig)

        return {
            "model": model_name,
            "dataset": dataset_name,
            "class_id": class_id,
            "image_id": image_id,
            "method": method_name,
            "url": f"{base_url}/{filename}",
        }

    def interpret(
        self,
        num_samples: int = 1,
    ) -> dict[str, List[TensorOrTupleOfTensorsGeneric]]:
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
        attributions: dict[str, List[TensorOrTupleOfTensorsGeneric]],
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        base_url: str,
        image_ext: str = "webp",
        **kwargs,
    ) -> list[dict[str, str]]:
        total_num_attributions = sum(
            [len(attr) for target in attributions.values() for attr in target.values()]
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        base_url = base_url.rstrip("/")
        image_ext = image_ext.lstrip(".").lower()

        records: list[dict[str, str]] = []

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
    ) -> Iterator[list[dict[str, str]]]:
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
                inputs.requires_grad = True

                class_id = str(target.item())
                image_id = str(class_image_counters[class_id])
                image_records: list[dict[str, str]] = []

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
                                **kwargs,
                            )
                        )

                class_image_counters[class_id] += 1
                pbar.update(1)
                yield image_records

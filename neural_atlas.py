from attr_config import AttributionConfig

from typing import Union, List
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

        OUT_PX = 512
        DPI = 128
        SIDE_IN = OUT_PX / DPI

        output_dir.mkdir(parents=True, exist_ok=True)
        base_url = base_url.rstrip("/")
        image_ext = image_ext.lstrip(".").lower()

        records: list[dict[str, str]] = []

        with tqdm(total=total_num_attributions, desc="Plotting Attributions") as pbar:
            for target, interp_methods in attributions.items():
                for interp_method, attr in interp_methods.items():
                    method_name = str(interp_method)
                    attr = torch.cat(attr, dim=0)
                    for i, attr in enumerate(attr):
                        fig, ax = plt.subplots(figsize=(SIDE_IN, SIDE_IN), dpi=DPI)
                        _ = viz.visualize_image_attr(
                            attr.permute(1, 2, 0).detach().cpu().numpy(),
                            plt_fig_axis=(fig, ax),
                            use_pyplot=False,
                            title=None,           
                            **kwargs,
                        )

                        ax.axis("off")
                        class_id = str(target)
                        image_id = str(i)

                        filename = (
                            f"{model_name}__{dataset_name}__{class_id}"
                            f"__{image_id}__{method_name}.{image_ext}"
                        )
                        save_kwargs = {
                            "format": image_ext,
                            "dpi": DPI,
                            "bbox_inches": "tight",
                            "pad_inches": 0,
                        }
                        if image_ext in {"webp", "jpg", "jpeg"}:
                            save_kwargs["pil_kwargs"] = {"quality": 95}

                        fig.savefig(output_dir / filename, **save_kwargs)
                        plt.close()
                        records.append(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "class_id": class_id,
                                "image_id": image_id,
                                "method": method_name,
                                "url": f"{base_url}/{filename}",
                            }
                        )
                        pbar.update(1)
        return records

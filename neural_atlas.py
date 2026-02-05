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
        export_root_path: Path,
        **kwargs,
    ) -> None:
        total_num_attributions = sum(
            [len(attr) for target in attributions.values() for attr in target.values()]
        )

        OUT_PX = 512
        DPI = 128
        SIDE_IN = OUT_PX / DPI

        with tqdm(total=total_num_attributions, desc="Plotting Attributions") as pbar:
            for target, interp_methods in attributions.items():
                for interp_method, attr in interp_methods.items():
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
                        save_path = (
                            export_root_path / Path(target) / Path(interp_method)
                        )
                        save_path.mkdir(parents=True, exist_ok=True)
                        fig.savefig(
                            save_path / f"{i}.webp",
                            format="webp",
                            dpi=DPI,
                            bbox_inches="tight",
                            pad_inches=0,
                            pil_kwargs={"quality": 95},
                        )
                        plt.close()
                        pbar.update(1)

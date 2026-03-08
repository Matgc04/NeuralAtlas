from __future__ import annotations

import argparse
import json
import warnings

# from torchvision import datasets

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from captum._utils.typing import TensorOrTupleOfTensorsGeneric
    import torch
    from torch import nn
    from attr_config import AttributionConfig

warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations")
warnings.filterwarnings(
    "ignore",
    message=(
        r"Setting forward, backward hooks and attributes on non-linear\s+"
        r"activations\.\s+The hooks and attributes will be removed\s+"
        r"after the attribution is finished"
    ),
    category=UserWarning,
    module=r"captum\.log\.dummy_log",
)

DEFAULT_MODEL_NAME = "alexnet"
DATASET_NAME = "imagenet-pico"
BASE_PATH = "interpretability-viewer/public/"
OUTPUT_IMAGES_DIR = BASE_PATH + "outputs/images"
OUTPUT_STRUCTURE_PATH = BASE_PATH + "outputs/outputs_structure.json"
IMAGE_EXT = "webp"
DEFAULT_NUM_SAMPLES = 20
DEFAULT_EXPORT_BATCH_IMAGES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attribution outputs.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all methods regardless of existing outputs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Torchvision model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Number of dataset samples to process in order "
            f"(default: {DEFAULT_NUM_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--export-batch-images",
        type=int,
        default=DEFAULT_EXPORT_BATCH_IMAGES,
        help=(
            "How many processed images to buffer before updating JSON metadata "
            f"(default: {DEFAULT_EXPORT_BATCH_IMAGES})."
        ),
    )
    return parser.parse_args()


def load_outputs_structure(path: str | Path) -> dict:
    output_path = Path(path)
    if not output_path.exists():
        return {"models": {}}
    try:
        with output_path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"models": {}}


def count_method_outputs(structure: dict, model: str, dataset: str, method: str) -> int:
    count = 0
    model_dict = structure.get("models", {}).get(model, {})
    dataset_dict = model_dict.get("datasets", {}).get(dataset, {})
    classes = dataset_dict.get("classes", {})
    for class_dict in classes.values():
        images = class_dict.get("images", {})
        for image_dict in images.values():
            outputs = image_dict.get("outputs", {})
            if method in outputs:
                count += 1
    return count


def is_method_complete(count: int, num_samples: int) -> bool:
    return count >= num_samples


def build_interp_methods(
    last_conv_layer: nn.Module,
    device: torch.device,
    to_rgb_heatmap: Callable[
        [TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric
    ],
) -> list[AttributionConfig]:
    from captum.attr import (
        Occlusion,
        GuidedGradCam,
        GradientShap,
        Saliency,
        IntegratedGradients,
        LayerIntegratedGradients,
        LayerGradCam,
        LayerAttribution,
        DeepLift,
        GuidedBackprop,
        InputXGradient,
        Deconvolution,
        Lime,
        KernelShap,
        ShapleyValueSampling,
    )
    from attr_config import AttributionConfig
    from models.interp_utils import make_superpixel_mask, kmeans_superpixels
    from skimage.segmentation import slic, quickshift
    import torch

    occlusion = AttributionConfig(
        Occlusion,
        sliding_window_shapes=(3, 15, 15),
        strides=(3, 8, 8),
    )
    guided_gradcam = AttributionConfig(
        GuidedGradCam,
        layer=last_conv_layer,
    )
    gradient_shap = AttributionConfig(
        GradientShap,
        n_samples=50,
        stdevs=0.0001,
        baselines=torch.ones(1, 3, 224, 224, device=device),
    )
    saliency = AttributionConfig(
        Saliency,
    )
    integrated_gradients = AttributionConfig(
        IntegratedGradients,
        n_steps=50,
    )
    layer_gradcam = AttributionConfig(
        LayerGradCam,
        layer=last_conv_layer,
        relu_attributions=True,
        callback=lambda attr: LayerAttribution.interpolate(
            attr,
            (224, 224),
            interpolate_mode="bilinear",
        ).repeat(1, 3, 1, 1),
    )
    deep_lift = AttributionConfig(
        DeepLift,
        baselines=torch.ones(1, 3, 224, 224, device=device),
    )
    guided_backprop = AttributionConfig(
        GuidedBackprop,
    )
    input_x_gradient = AttributionConfig(
        InputXGradient,
    )
    deconvolution = AttributionConfig(
        Deconvolution,
    )

    def _make_superpixel_runtime_kwargs(mask_fn, **seg_kwargs):
        def _runtime_kwargs(
            inputs: TensorOrTupleOfTensorsGeneric, _target: object
        ) -> dict[str, torch.Tensor]:
            inputs_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
            if not isinstance(inputs_tensor, torch.Tensor):
                raise TypeError(
                    "Lime runtime kwargs expected tensor inputs or tuple[Tensor, ...], "
                    f"got {type(inputs_tensor)}."
                )
            return {"feature_mask": make_superpixel_mask(mask_function=mask_fn, img=inputs_tensor, **seg_kwargs)}
        return _runtime_kwargs

    slic_medium = _make_superpixel_runtime_kwargs(
        slic, n_segments=100, compactness=10.0, start_label=0
    )
    quickshift_medium = _make_superpixel_runtime_kwargs(
        quickshift, kernel_size=8, max_dist=15, ratio=0.8
    )
    kmeans_medium = _make_superpixel_runtime_kwargs(
        kmeans_superpixels,
        n_clusters=16,
        add_xy=True,
        xy_weight=0.2,
        random_state=0,
        n_init=10,
    )

    lime_slic = AttributionConfig(
        Lime,
        runtime_kwargs_fn=slic_medium,
        n_samples = 50,
        suffix="(SLIC)"
    )

    lime_quickshift = AttributionConfig(
        Lime,
        runtime_kwargs_fn=quickshift_medium,
        n_samples = 50,
        suffix="(Quickshift)"
    )

    lime_kmeans = AttributionConfig(
        Lime,
        runtime_kwargs_fn=kmeans_medium,
        n_samples=50,
        suffix="(KMeans)",
    )

    kernel_shap_slic = AttributionConfig(
        KernelShap,
        runtime_kwargs_fn=slic_medium,
        n_samples=50,
        suffix="(SLIC)",
    )

    kernel_shap_quickshift = AttributionConfig(
        KernelShap,
        runtime_kwargs_fn=quickshift_medium,
        n_samples=50,
        suffix="(Quickshift)",
    )

    kernel_shap_kmeans = AttributionConfig(
        KernelShap,
        runtime_kwargs_fn=kmeans_medium,
        n_samples=50,
        suffix="(KMeans)",
    )

    shapley_value_sampling_slic = AttributionConfig(
        ShapleyValueSampling,
        runtime_kwargs_fn=slic_medium,
        n_samples=15,
        perturbations_per_eval=8,
        suffix="(SLIC)",
    )

    shapley_value_sampling_quickshift = AttributionConfig(
        ShapleyValueSampling,
        runtime_kwargs_fn=quickshift_medium,
        n_samples=15,
        perturbations_per_eval=8,
        suffix="(Quickshift)",
    )

    shapley_value_sampling_kmeans = AttributionConfig(
        ShapleyValueSampling,
        runtime_kwargs_fn=kmeans_medium,
        n_samples=15,
        perturbations_per_eval=8,
        suffix="(KMeans)",
    )

    layer_integrated_gradients = AttributionConfig(
        LayerIntegratedGradients,
        layer=last_conv_layer,
        baselines=torch.ones(1, 3, 224, 224, device=device),
        n_steps=50,
        # internal_batch_size=1,
        attribute_to_layer_input=False,
        callback=to_rgb_heatmap,
    )
    

    return [
        occlusion,
        guided_gradcam,
        gradient_shap,
        saliency,
        integrated_gradients,
        layer_gradcam,
        deep_lift,
        guided_backprop,
        input_x_gradient,
        deconvolution,
        lime_slic,
        lime_quickshift,
        lime_kmeans,
        kernel_shap_slic,
        kernel_shap_quickshift,
        kernel_shap_kmeans,
        shapley_value_sampling_slic,
        shapley_value_sampling_quickshift,
        shapley_value_sampling_kmeans,
        layer_integrated_gradients,
    ]


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be a positive integer.")
    if args.export_batch_images <= 0:
        raise SystemExit("--export-batch-images must be a positive integer.")
    
    import torch
    from torch import nn
    from torchvision import models
    from torchvision import transforms

    from models.interp_resnet18 import InterpResnet18
    from models.interp_utils import (
        disable_inplace_relu,
        to_rgb_heatmap,
    )
    from neural_atlas import NeuralAtlas
    from output_exporter import OutputExporter

    torch.manual_seed(0)

    DEVICE, DTYPE = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.float32,
    )

    print(f"Using device: {DEVICE}, dtype: {DTYPE}")

    if not hasattr(models, args.model):
        raise SystemExit(f"Unknown model '{args.model}'.")

    if args.model == "resnet18":
        model = InterpResnet18(weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)
    else:
        model = getattr(models, args.model)(weights="DEFAULT").to(
            device=DEVICE, dtype=DTYPE
        )

    disable_inplace_relu(model)

    model = nn.Sequential(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        model,
        # nn.Softmax(dim=1),
    )

    # Obtain the last convolutional layer
    last_conv_layer = None
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

    if last_conv_layer is None:
        raise SystemExit("Could not determine last convolutional layer for Grad-CAM methods.")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device=DEVICE, dtype=DTYPE)),
        ]
    )
    data = BASE_PATH + f"{DATASET_NAME}/val"
    # data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model {type(model).__name__} total parameters: ", pytorch_total_params)

    interp_methods = build_interp_methods(last_conv_layer, DEVICE, to_rgb_heatmap)

    if not args.recompute:
        structure = load_outputs_structure(OUTPUT_STRUCTURE_PATH)
        filtered_methods = []
        for method in interp_methods:
            method_name = str(method)
            existing_count = count_method_outputs(
                structure, args.model, DATASET_NAME, method_name
            )
            if is_method_complete(existing_count, args.num_samples):
                print(
                    f"Skipping {method_name}: {existing_count} outputs already exist."
                )
            else:
                filtered_methods.append(method)
        interp_methods = filtered_methods

    if not interp_methods:
        print("No new methods to run; exporting model predictions only.")

    natlas = NeuralAtlas(
        model,
        data,
        interp_methods,
        transform=transform,
    )

    exporter = OutputExporter()
    records_buffer: list[dict] = []
    buffered_images = 0

    for image_records in natlas.interpret_and_visualize_stream(
        num_samples=args.num_samples,
        output_dir=Path(OUTPUT_IMAGES_DIR),
        model_name=args.model,
        dataset_name=DATASET_NAME,
        base_url="/outputs/images",
        image_ext=IMAGE_EXT,
        method="heat_map",
        sign="absolute_value",
        cmap="jet",
        show_colorbar=True,
    ):
        records_buffer.extend(image_records)
        buffered_images += 1

        if buffered_images >= args.export_batch_images:
            exporter.export_to_json(records_buffer, OUTPUT_STRUCTURE_PATH)
            records_buffer.clear()
            buffered_images = 0

    if records_buffer:
        exporter.export_to_json(records_buffer, OUTPUT_STRUCTURE_PATH)


if __name__ == "__main__":
    main()

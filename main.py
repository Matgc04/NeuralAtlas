from attr_config import AttributionConfig
from models.interp_resnet18 import InterpResnet18
from models.interp_utils import disable_inplace_relu
from neural_atlas import NeuralAtlas
from output_exporter import OutputExporter

import argparse
import json
import warnings

import torch
from torch import nn
from torchvision import models
from torchvision import transforms

# from torchvision import datasets

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
)

from pathlib import Path

warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations")

torch.manual_seed(0)

DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)

DEFAULT_MODEL_NAME = "alexnet"
DATASET_NAME = "imagenet-pico"
BASE_PATH = "interpretability-viewer/public/"
OUTPUT_IMAGES_DIR = BASE_PATH + "outputs/images"
OUTPUT_STRUCTURE_PATH = BASE_PATH + "outputs/outputs_structure.json"
IMAGE_EXT = "webp"

num_samples = 20


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


def main() -> None:
    args = parse_args()

    print(f"Using device: {DEVICE}, dtype: {DTYPE}")

    if not hasattr(models, args.model):
        raise SystemExit(f"Unknown model '{args.model}'.")

    if args.model == "resnet18":
        model = InterpResnet18(weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)
    else:
        model = getattr(models, args.model)(weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)

    disable_inplace_relu(model)

    model = nn.Sequential(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        model,
        # nn.Softmax(dim=1),
    )

    # Obtain the last convolutional layer
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

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
        baselines=torch.ones(num_samples, 3, 224, 224, device=DEVICE),
    )
    saliency = AttributionConfig(
        Saliency,
    )
    integrated_gradients = AttributionConfig(
        IntegratedGradients,
        n_steps=200,
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
        baselines=torch.ones(1, 3, 224, 224, device=DEVICE)
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
    layer_integrated_gradients = AttributionConfig(
        LayerIntegratedGradients,
        layer=last_conv_layer,
        baselines=torch.ones(1, 3, 224, 224, device=DEVICE),
        attribute_to_layer_input=False,
        callback=lambda attr: LayerAttribution.interpolate(
            attr,
            (224, 224),
            interpolate_mode="bilinear",
        ).repeat(1, 3, 1, 1)
    )

    interp_methods = [
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
        layer_integrated_gradients,
    ]

    if not args.recompute:
        structure = load_outputs_structure(OUTPUT_STRUCTURE_PATH)
        filtered_methods = []
        for method in interp_methods:
            method_name = str(method)
            existing_count = count_method_outputs(
                structure, args.model, DATASET_NAME, method_name
            )
            if is_method_complete(existing_count, num_samples):
                print(
                    f"Skipping {method_name}: {existing_count} outputs already exist."
                )
            else:
                filtered_methods.append(method)
        interp_methods = filtered_methods

    if not interp_methods:
        print("No new methods to run; all outputs already computed.")
        return

    natlas = NeuralAtlas(
        model,
        data,
        interp_methods,
        transform=transform,
    )
    attributions = natlas.interpret(num_samples=num_samples)
    records = natlas.visualize(
        attributions,
        output_dir=Path(OUTPUT_IMAGES_DIR),
        model_name=args.model,
        dataset_name=DATASET_NAME,
        base_url="/outputs/images",
        image_ext=IMAGE_EXT,
        method="heat_map",
        sign="absolute_value",
        cmap="jet",
        show_colorbar=True,
    )

    exporter = OutputExporter()
    exporter.export_to_json(records, OUTPUT_STRUCTURE_PATH)


if __name__ == "__main__":
    main()

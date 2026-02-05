from attr_config import AttributionConfig
from neural_atlas import NeuralAtlas
from output_exporter import OutputExporter

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
    LayerGradCam,
    LayerAttribution,
)

from pathlib import Path

warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations")

torch.manual_seed(0)

DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)

print(f"Using device: {DEVICE}, dtype: {DTYPE}")

MODEL_NAME = "alexnet"
BASE_PATH = "interpretability-viewer/public/"
OUTPUT_PATH = BASE_PATH + "outputs"

num_samples = 20

model = getattr(models, MODEL_NAME)(weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)
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
data = BASE_PATH + "imagenet-pico/val"
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

natlas = NeuralAtlas(
    model,
    data,
    [
        occlusion,
        guided_gradcam,
        gradient_shap,
        saliency,
        integrated_gradients,
        layer_gradcam,
    ],
    transform=transform,
)
attributions = natlas.interpret(num_samples=num_samples)
natlas.visualize(
    attributions,
    Path(f"{OUTPUT_PATH}/{MODEL_NAME}/{'imagenet-pico'}"),
    method="heat_map",
    sign="absolute_value",
    cmap="jet",
    show_colorbar=True,
)

exporter = OutputExporter(OUTPUT_PATH)
exporter.export_to_json("outputs_structure.json")

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Optional, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models, transforms
from torchvision.models._api import WeightsEnum
from torchvision.models.inception import BasicConv2d
from torchvision.models.resnet import (
    ResNet,
    ResNet18_Weights,
    ResNet101_Weights,
    conv1x1,
    conv3x3,
)

from backend import config
from backend.datasets import load_label_space

@dataclass(slots=True)
class ModelRuntime:
    model: "nn.Module"
    device: "torch.device"
    dtype: "torch.dtype"
    transform: Callable[[object], "Tensor"]
    last_conv_layer: "nn.Module"
    parameter_count: int
    # Id of the label space the model's outputs index; a dataset is run only
    # against models that predict its label space.
    label_space: str


def _overwrite_named_param_strict(
    kwargs: dict[str, Any],
    param: str,
    new_value: object,
) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead."
            )
    else:
        kwargs[param] = new_value


class InterpBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("InterpBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in InterpBasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)
        return out


class InterpBottleneck(nn.Module):
    """Torchvision `Bottleneck` with one ReLU module per activation site.

    The stock block calls a single `self.relu` three times, which DeepLift rejects:
    it stores one input/output pair per module. Splitting the call sites keeps the
    parameters (and therefore the pretrained state dict) identical.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu3(out + identity)


class InterpResnet(ResNet):
    """A torchvision ResNet whose blocks have one ReLU module per activation site."""

    block: type[nn.Module] = InterpBasicBlock
    layers: list[int] = [2, 2, 2, 2]
    weights_enum: Any = ResNet18_Weights

    def __init__(
        self,
        *,
        weights: Optional[Union[WeightsEnum, str]] = None,
        progress: bool = True,
        **kwargs: Any,
    ) -> None:
        verified_weights = self.weights_enum.verify(weights)
        if isinstance(verified_weights, str):
            if verified_weights != "DEFAULT":
                raise ValueError(f"Unsupported weights value: {verified_weights}")
            verified_weights = self.weights_enum.DEFAULT

        if verified_weights is not None:
            _overwrite_named_param_strict(
                kwargs,
                "num_classes",
                len(verified_weights.meta["categories"]),
            )

        super().__init__(
            block=cast(Any, self.block),
            layers=self.layers,
            **kwargs,
        )

        if verified_weights is not None:
            state_dict = verified_weights.get_state_dict(progress=progress, check_hash=True)
            self.load_state_dict(state_dict, strict=True)


class InterpResnet18(InterpResnet):
    block = InterpBasicBlock
    layers = [2, 2, 2, 2]
    weights_enum = ResNet18_Weights


class InterpResnet101(InterpResnet):
    block = InterpBottleneck
    layers = [3, 4, 23, 3]
    weights_enum = ResNet101_Weights


def disable_inplace_relu(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


class HookableReLU(nn.ReLU):
    """ReLU module for architectures that call `F.relu`, which Captum cannot hook.

    GuidedBackprop and Deconvolution only override modules that are `nn.ReLU`
    instances. DeepLift matches the exact type; `build_interp_methods` registers
    both stand-ins with its rescale rule."""


class HookableReLU6(nn.ReLU):
    """`nn.ReLU6` is not an `nn.ReLU`, so GuidedBackprop and Deconvolution skip it.
    Same forward; the backward override then clamps the gradient masked to 0 < x < 6."""

    def forward(self, input: Tensor) -> Tensor:
        return F.relu6(input)


class HookableGELU(nn.ReLU):
    """GELU exposed as `nn.ReLU` so GuidedBackprop and Deconvolution hook it.
    GuidedBackprop then yields relu(grad * gelu'(x)) and Deconvolution relu(grad).
    Our extension: neither method is defined for smooth activations. DeepLift is
    left unregistered (it reduces to InputXGradient here anyway)."""

    def __init__(self, approximate: str) -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)


class HookableSiLU(nn.ReLU):
    """SiLU counterpart of `HookableGELU` (EfficientNet, including its SE blocks)."""

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)


def _basic_conv2d_forward(self: nn.Module, x: Tensor) -> Tensor:
    return self.relu(self.bn(self.conv(x)))


def make_relus_hookable(model: nn.Module) -> nn.Module:
    """Expose every elementwise activation as an `nn.ReLU` instance without changing
    the forward."""
    for parent in list(model.modules()):
        if isinstance(parent, BasicConv2d):  # Inception v3: F.relu inside forward
            parent.relu = HookableReLU()
            parent.forward = MethodType(_basic_conv2d_forward, parent)
        for name, child in parent.named_children():
            if type(child) is nn.ReLU6:  # MobileNet v2
                setattr(parent, name, HookableReLU6())
            elif type(child) is nn.GELU:  # ConvNeXt
                setattr(parent, name, HookableGELU(child.approximate))
            elif type(child) is nn.SiLU:  # EfficientNet
                setattr(parent, name, HookableSiLU())
    return model


INTERP_RESNETS = {"resnet18": InterpResnet18, "resnet101": InterpResnet101}


@dataclass(frozen=True, slots=True)
class Preprocess:
    resize: int
    crop: int
    mean: list[float]
    std: list[float]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """`model_specs/<id>.json`. `architecture` is a torchvision builder (resnet18/101 build
    the Interp variants, which DeepLift needs); `weights` is "DEFAULT" or a state_dict path
    relative to model_specs/."""

    architecture: str
    weights: str
    label_space: str
    preprocess: Preprocess


def model_ids(specs_dir: Path = config.MODEL_SPECS_DIR) -> list[str]:
    return sorted(path.stem for path in specs_dir.glob("*.json"))


def load_model_spec(model_id: str, specs_dir: Path = config.MODEL_SPECS_DIR) -> ModelSpec:
    payload = json.loads((specs_dir / f"{model_id}.json").read_text())
    return ModelSpec(payload["architecture"], payload["weights"], payload["label_space"],
                     Preprocess(**payload["preprocess"]))


def _build_network(spec: ModelSpec, specs_dir: Path) -> nn.Module:
    builder = INTERP_RESNETS.get(spec.architecture) or partial(models.get_model, spec.architecture)
    if spec.weights == "DEFAULT":
        return builder(weights="DEFAULT")
    # A checkpoint carries its own head, sized to the label space it predicts.
    network = builder(weights=None, num_classes=len(load_label_space(spec.label_space).labels))
    state_dict = torch.load(specs_dir / spec.weights, map_location="cpu", weights_only=True)
    network.load_state_dict(state_dict, strict=True)
    return network


def build_model_runtime(model_id: str, specs_dir: Path = config.MODEL_SPECS_DIR) -> ModelRuntime:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    spec = load_model_spec(model_id, specs_dir)
    base_model = _build_network(spec, specs_dir).to(device=device, dtype=dtype)
    disable_inplace_relu(base_model)
    make_relus_hookable(base_model)

    model = nn.Sequential(
        transforms.Normalize(spec.preprocess.mean, spec.preprocess.std),
        base_model,
    )

    last_conv_layer = None
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

    if last_conv_layer is None:
        raise SystemExit(
            "Could not determine last convolutional layer for Grad-CAM methods."
        )

    transform = transforms.Compose(
        [
            transforms.Resize(spec.preprocess.resize),
            transforms.CenterCrop(spec.preprocess.crop),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device=device, dtype=dtype)),
        ]
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return ModelRuntime(
        model=model,
        device=device,
        dtype=dtype,
        transform=transform,
        last_conv_layer=last_conv_layer,
        parameter_count=parameter_count,
        label_space=spec.label_space,
    )

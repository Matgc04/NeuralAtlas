from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

import torch
from torch import Tensor, nn
from torchvision import models, transforms
from torchvision.models._api import WeightsEnum
from torchvision.models.resnet import (
    ResNet,
    ResNet18_Weights,
    ResNet101_Weights,
    conv1x1,
    conv3x3,
)

# The crop every model in the catalog is trained on. Shared with `backend.vlm`
# so the VLM is shown exactly the pixels the classifier saw.
MODEL_VIEW = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)


@dataclass(slots=True)
class ModelRuntime:
    model: "nn.Module"
    device: "torch.device"
    dtype: "torch.dtype"
    transform: Callable[[object], "Tensor"]
    last_conv_layer: "nn.Module"
    parameter_count: int


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


def build_model_runtime(model_name: str) -> ModelRuntime:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    if not hasattr(models, model_name):
        raise SystemExit(f"Unknown model '{model_name}'.")

    interp_resnets = {"resnet18": InterpResnet18, "resnet101": InterpResnet101}
    if model_name in interp_resnets:
        base_model = interp_resnets[model_name](weights="DEFAULT").to(
            device=device, dtype=dtype
        )
    else:
        base_model = getattr(models, model_name)(weights="DEFAULT").to(
            device=device, dtype=dtype
        )

    disable_inplace_relu(base_model)

    model = nn.Sequential(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
            MODEL_VIEW,
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
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING, Union, cast

import torch
from torch import Tensor, nn
from torchvision import models, transforms
from torchvision.models.resnet import ResNet, ResNet18_Weights, conv3x3

if TYPE_CHECKING:
    pass


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


class InterpResnet18(ResNet):
    def __init__(
        self,
        *,
        weights: Optional[Union[ResNet18_Weights, str]] = None,
        progress: bool = True,
        **kwargs: Any,
    ) -> None:
        verified_weights = ResNet18_Weights.verify(weights)
        if isinstance(verified_weights, str):
            if verified_weights != "DEFAULT":
                raise ValueError(f"Unsupported weights value: {verified_weights}")
            verified_weights = ResNet18_Weights.DEFAULT

        if verified_weights is not None:
            _overwrite_named_param_strict(
                kwargs,
                "num_classes",
                len(verified_weights.meta["categories"]),
            )

        super().__init__(
            block=cast(Any, InterpBasicBlock),
            layers=[2, 2, 2, 2],
            **kwargs,
        )

        if verified_weights is not None:
            state_dict = verified_weights.get_state_dict(progress=progress, check_hash=True)
            self.load_state_dict(state_dict, strict=True)


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

    if model_name == "resnet18":
        base_model = InterpResnet18(weights="DEFAULT").to(device=device, dtype=dtype)
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
            transforms.Resize(256),
            transforms.CenterCrop(224),
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

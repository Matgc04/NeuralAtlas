# interp_resnet18.py
from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import nn
from torchvision.models.resnet import ResNet, conv3x3
from torchvision.models.resnet import ResNet18_Weights



def _overwrite_named_param_strict(kwargs: dict[str, Any], param: str, new_value: object) -> None:
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
        # Mirror torchvision’s verify behavior
        weights = ResNet18_Weights.verify(weights)

        # Mirror torchvision’s num_classes overwrite when weights are provided
        if weights is not None:
            _overwrite_named_param_strict(kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(block=InterpBasicBlock, layers=[2, 2, 2, 2], **kwargs)

        if weights is not None:
            state_dict = weights.get_state_dict(progress=progress, check_hash=True)
            self.load_state_dict(state_dict, strict=True)

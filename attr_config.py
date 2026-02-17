from collections.abc import Callable
from typing import Any, cast

from captum.attr import Attribution
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric, TargetType

_Callback = Callable[[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric]


class AttributionConfig:
    def __init__(
        self,
        attribution_class: type[Attribution],
        callback: _Callback | None = None,
        **kwargs: Any,
    ) -> None:
        self.attribution_class = attribution_class
        self.config = kwargs
        self.callback: _Callback = callback if callback is not None else cast(_Callback, lambda x: x)
        self.layer = self.config.pop("layer", None)


    def attribute(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
    ) -> TensorOrTupleOfTensorsGeneric:
        attributor = self.attribution_class(
            model,
            **({"layer": self.layer} if self.layer else {}),
        )

        return self.callback(
            attributor.attribute(
                inputs=inputs,
                target=target,
                **self.config,
            )
        )

    def __str__(self) -> str:
        return f"{self.attribution_class.__name__}"

    def __repr__(self) -> str:
        return f"AttributionConfig(attribution_class={self.attribution_class}, config={self.config})"

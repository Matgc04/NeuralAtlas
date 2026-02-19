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

        self._attributor: Attribution | None = None
        self._bound_model_id: int | None = None

    def _get_attributor(self, model: Module) -> Attribution:
        mid = id(model)
        if self._attributor is None or self._bound_model_id != mid:
            self._attributor = self.attribution_class(
                model,
                **({"layer": self.layer} if self.layer else {}),
            )
            self._bound_model_id = mid
        return self._attributor

    def attribute(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
    ) -> TensorOrTupleOfTensorsGeneric:
        attributor = self._get_attributor(model)

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

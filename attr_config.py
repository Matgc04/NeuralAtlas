from collections.abc import Callable
from typing import Any, cast

from captum.attr import Attribution
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric, TargetType

_Callback = Callable[[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric]
_RuntimeKwargsFn = Callable[
    [TensorOrTupleOfTensorsGeneric, TargetType],
    dict[str, Any],
]


class AttributionConfig:
    def __init__(
        self,
        attribution_class: type[Attribution],
        callback: _Callback | None = None,
        runtime_kwargs_fn: _RuntimeKwargsFn | None = None,
        name: str | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.attribution_class = attribution_class
        self.config = kwargs
        self.callback: _Callback = callback if callback is not None else cast(_Callback, lambda x: x)
        self.name = name
        self.suffix = suffix
        self.runtime_kwargs_fn = runtime_kwargs_fn
        self.layer = self.config.pop("layer", None)
        # Kwargs of the most recent `attribute` call, so a caller that needs the
        # same inputs the attributor saw (a superpixel `feature_mask`, say) can
        # read them back instead of reaching into the provider.
        self.last_runtime_kwargs: dict[str, Any] = {}

        self._attributor: Attribution | None = None
        self._bound_model_id: int | None = None

    def _get_attributor(self, model: Module) -> Attribution:
        mid = id(model)
        if self._attributor is None or self._bound_model_id != mid:
            self._attributor = self.attribution_class(
                model,
                **({"layer": self.layer} if self.layer is not None else {}),
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
        runtime_kwargs = (
            self.runtime_kwargs_fn(inputs, target)
            if self.runtime_kwargs_fn is not None
            else {}
        )
        self.last_runtime_kwargs = runtime_kwargs

        return self.callback(
            attributor.attribute(
                inputs=inputs,
                target=target,
                **self.config,
                **runtime_kwargs,
            )
        )

    def __str__(self) -> str:
        suffix = f" {self.suffix}" if self.suffix else ""
        return f"{self.name or self.attribution_class.__name__}{suffix}"

    def __repr__(self) -> str:
        return f"AttributionConfig(attribution_class={self.attribution_class}, config={self.config}, name={self.name}, suffix={self.suffix})"

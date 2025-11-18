from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Tolerance:
    rtol: float = 1e-5
    atol: float = 1e-6


ParamDict = dict[str, Any]
StructDict = dict[str, Any]
NameMapping = dict[str, str]


class Layer(Protocol):
    def get_weights(self) -> list[Any]:
        ...

    def set_weights(self, weights: list[Any]) -> None:
        ...


TransferFn = Callable[[Layer, ParamDict], None]
VerifyFn = Callable[[Layer, ParamDict, Tolerance], tuple[bool, list[str]]]


@dataclass(frozen=True)
class ConversionResult:
    matches: list[str]
    mismatches: list[str]
    layers_transferred: int

    @property
    def success(self) -> bool:
        return len(self.mismatches) == 0


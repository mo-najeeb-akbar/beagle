from __future__ import annotations

from dataclasses import dataclass

import pytest

from beagle.conversions.dispatch import ConversionRegistry
from beagle.conversions.types import Layer, ParamDict, Tolerance


@dataclass
class MockLayer:
    name: str
    weights: list[float]

    def get_weights(self) -> list[float]:
        return self.weights

    def set_weights(self, weights: list[float]) -> None:
        self.weights = weights


def test_registry_register_transfer() -> None:
    registry = ConversionRegistry()

    def mock_transfer(layer: Layer, params: ParamDict) -> None:
        layer.set_weights([1.0, 2.0])

    registry.register_transfer("MockLayer", mock_transfer)

    assert registry.has_transfer("MockLayer")
    assert registry.get_transfer("MockLayer") is mock_transfer


def test_registry_register_verify() -> None:
    registry = ConversionRegistry()

    def mock_verify(
        layer: Layer, params: ParamDict, tolerance: Tolerance
    ) -> tuple[bool, list[str]]:
        return True, ["success"]

    registry.register_verify("MockLayer", mock_verify)

    assert registry.has_verify("MockLayer")
    assert registry.get_verify("MockLayer") is mock_verify


def test_registry_get_nonexistent() -> None:
    registry = ConversionRegistry()

    assert registry.get_transfer("NonExistent") is None
    assert registry.get_verify("NonExistent") is None
    assert not registry.has_transfer("NonExistent")
    assert not registry.has_verify("NonExistent")


def test_registry_multiple_types() -> None:
    registry = ConversionRegistry()

    def transfer1(layer: Layer, params: ParamDict) -> None:
        pass

    def transfer2(layer: Layer, params: ParamDict) -> None:
        pass

    registry.register_transfer("Type1", transfer1)
    registry.register_transfer("Type2", transfer2)

    assert registry.get_transfer("Type1") is transfer1
    assert registry.get_transfer("Type2") is transfer2


def test_registry_overwrite() -> None:
    registry = ConversionRegistry()

    def transfer1(layer: Layer, params: ParamDict) -> None:
        pass

    def transfer2(layer: Layer, params: ParamDict) -> None:
        pass

    registry.register_transfer("Type1", transfer1)
    registry.register_transfer("Type1", transfer2)

    assert registry.get_transfer("Type1") is transfer2


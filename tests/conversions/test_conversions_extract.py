from __future__ import annotations

from dataclasses import dataclass

import pytest

from beagle.conversions.extract import extract_layer_refs, extract_structure


@dataclass
class MockLayer:
    name: str
    weights: list[float]

    def get_weights(self) -> list[float]:
        return self.weights

    def set_weights(self, weights: list[float]) -> None:
        self.weights = weights


@dataclass
class MockModel:
    conv1: MockLayer
    conv2: MockLayer
    dense: MockLayer


@dataclass
class MockNestedModel:
    encoder: MockModel
    decoder: MockModel


@dataclass
class MockListModel:
    layers: list[MockLayer]


def is_mock_layer(x: any) -> bool:
    return isinstance(x, MockLayer)


def test_extract_structure_flat_model() -> None:
    model = MockModel(
        conv1=MockLayer("conv1", [1.0]),
        conv2=MockLayer("conv2", [2.0]),
        dense=MockLayer("dense", [3.0]),
    )

    result = extract_structure(model, is_mock_layer)

    assert "conv1" in result
    assert "conv2" in result
    assert "dense" in result
    assert result["conv1"] == {}
    assert result["conv2"] == {}
    assert result["dense"] == {}


def test_extract_structure_nested_model() -> None:
    encoder = MockModel(
        conv1=MockLayer("e_conv1", [1.0]),
        conv2=MockLayer("e_conv2", [2.0]),
        dense=MockLayer("e_dense", [3.0]),
    )
    decoder = MockModel(
        conv1=MockLayer("d_conv1", [4.0]),
        conv2=MockLayer("d_conv2", [5.0]),
        dense=MockLayer("d_dense", [6.0]),
    )
    model = MockNestedModel(encoder=encoder, decoder=decoder)

    result_encoder = extract_structure(encoder, is_mock_layer)
    result_decoder = extract_structure(decoder, is_mock_layer)

    assert "conv1" in result_encoder
    assert "conv2" in result_encoder
    assert "dense" in result_encoder
    assert "conv1" in result_decoder
    assert "conv2" in result_decoder
    assert "dense" in result_decoder


def test_extract_structure_with_list() -> None:
    @dataclass
    class SimpleModel:
        conv_layers: list[MockLayer]

    model = SimpleModel(
        conv_layers=[
            MockLayer("layer0", [1.0]),
            MockLayer("layer1", [2.0]),
            MockLayer("layer2", [3.0]),
        ]
    )

    result = extract_structure(model, is_mock_layer)

    assert "conv_layers.0" in result
    assert "conv_layers.1" in result
    assert "conv_layers.2" in result


def test_extract_layer_refs_flat_model() -> None:
    model = MockModel(
        conv1=MockLayer("conv1", [1.0]),
        conv2=MockLayer("conv2", [2.0]),
        dense=MockLayer("dense", [3.0]),
    )

    result = extract_layer_refs(model, is_mock_layer)

    assert "conv1" in result
    assert "conv2" in result
    assert "dense" in result
    assert result["conv1"].name == "conv1"
    assert result["conv2"].name == "conv2"
    assert result["dense"].name == "dense"


def test_extract_layer_refs_with_list() -> None:
    @dataclass
    class SimpleModel:
        conv_layers: list[MockLayer]

    model = SimpleModel(
        conv_layers=[
            MockLayer("layer0", [1.0]),
            MockLayer("layer1", [2.0]),
            MockLayer("layer2", [3.0]),
        ]
    )

    result = extract_layer_refs(model, is_mock_layer)

    assert "conv_layers.0" in result
    assert "conv_layers.1" in result
    assert "conv_layers.2" in result
    assert result["conv_layers.0"].name == "layer0"
    assert result["conv_layers.1"].name == "layer1"
    assert result["conv_layers.2"].name == "layer2"


def test_extract_structure_skips_private_attrs() -> None:
    model = MockModel(
        conv1=MockLayer("conv1", [1.0]),
        conv2=MockLayer("conv2", [2.0]),
        dense=MockLayer("dense", [3.0]),
    )
    model._private = MockLayer("private", [99.0])

    result = extract_structure(model, is_mock_layer)

    assert "_private" not in result
    assert "conv1" in result


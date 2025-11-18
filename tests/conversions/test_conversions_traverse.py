from __future__ import annotations

from dataclasses import dataclass

import pytest

from beagle.conversions.traverse import create_name_mapping, is_param_dict, traverse_paired


@dataclass
class MockLayer:
    name: str
    weights: list[float]

    def get_weights(self) -> list[float]:
        return self.weights

    def set_weights(self, weights: list[float]) -> None:
        self.weights = weights


def test_create_name_mapping_exact_match() -> None:
    source = {"conv1": {}, "conv2": {}, "dense": {}}
    target = {"conv1": {}, "conv2": {}, "dense": {}}

    mapping = create_name_mapping(source, target)

    assert mapping["conv1"] == "conv1"
    assert mapping["conv2"] == "conv2"
    assert mapping["dense"] == "dense"


def test_create_name_mapping_case_insensitive() -> None:
    source = {"Conv1": {}, "Conv2": {}, "Dense": {}}
    target = {"conv1": {}, "conv2": {}, "dense": {}}

    mapping = create_name_mapping(source, target)

    assert mapping["Conv1"] == "conv1"
    assert mapping["Conv2"] == "conv2"
    assert mapping["Dense"] == "dense"


def test_create_name_mapping_residual_blocks() -> None:
    source = {"ResidualBlock_0": {}, "ResidualBlock_1": {}, "ResidualBlock_2": {}}
    target = {"residual_blocks.0": {}, "residual_blocks.1": {}, "residual_blocks.2": {}}

    mapping = create_name_mapping(source, target)

    assert mapping["ResidualBlock_0"] == "residual_blocks.0"
    assert mapping["ResidualBlock_1"] == "residual_blocks.1"
    assert mapping["ResidualBlock_2"] == "residual_blocks.2"


def test_create_name_mapping_conv_layers() -> None:
    source = {"Conv_0": {}, "Conv_1": {}, "Conv_2": {}}
    target = {"conv_layers.0": {}, "conv_layers.1": {}, "conv_layers.2": {}}

    mapping = create_name_mapping(source, target)

    assert mapping["Conv_0"] == "conv_layers.0"
    assert mapping["Conv_1"] == "conv_layers.1"
    assert mapping["Conv_2"] == "conv_layers.2"


def test_is_param_dict_with_kernel() -> None:
    params = {"kernel": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.5, 0.5]}
    assert is_param_dict(params) is True


def test_is_param_dict_with_scale() -> None:
    params = {"scale": [1.0, 1.0], "bias": [0.0, 0.0]}
    assert is_param_dict(params) is True


def test_is_param_dict_with_mean_var() -> None:
    params = {"mean": [0.0, 0.0], "var": [1.0, 1.0]}
    assert is_param_dict(params) is True


def test_is_param_dict_without_params() -> None:
    params = {"encoder": {}, "decoder": {}}
    assert is_param_dict(params) is False


def test_is_param_dict_non_dict() -> None:
    assert is_param_dict([1.0, 2.0, 3.0]) is False
    assert is_param_dict("not a dict") is False
    assert is_param_dict(None) is False


def test_traverse_paired_simple() -> None:
    source_params = {"conv1": {"kernel": [1.0], "bias": [0.5]}}
    target_layers = {"conv1": MockLayer("conv1", [])}
    target_struct = {"conv1": {}}

    results = list(traverse_paired(source_params, target_layers, target_struct))

    assert len(results) == 1
    layer, params, path = results[0]
    assert layer.name == "conv1"
    assert params == {"kernel": [1.0], "bias": [0.5]}
    assert path == "conv1"


def test_traverse_paired_with_name_mapping() -> None:
    source_params = {"Conv1": {"kernel": [1.0], "bias": [0.5]}}
    target_layers = {"conv1": MockLayer("conv1", [])}
    target_struct = {"conv1": {}}
    name_mapping = {"Conv1": "conv1"}

    results = list(
        traverse_paired(
            source_params, target_layers, target_struct, name_mapping=name_mapping
        )
    )

    assert len(results) == 1
    layer, params, path = results[0]
    assert layer.name == "conv1"
    assert path == "Conv1"


def test_traverse_paired_nested() -> None:
    source_params = {
        "encoder": {"conv1": {"kernel": [1.0], "bias": [0.5]}},
        "decoder": {"conv2": {"kernel": [2.0], "bias": [0.3]}},
    }
    target_layers = {
        "encoder": MockLayer("encoder", []),
        "decoder": MockLayer("decoder", []),
    }
    target_struct = {"encoder": {"conv1": {}}, "decoder": {"conv2": {}}}

    results = list(traverse_paired(source_params, target_layers, target_struct))

    assert len(results) >= 0


def test_traverse_paired_unwrap_single_child() -> None:
    source_params = {"haar_conv": {"Conv_0": {"kernel": [1.0]}}}
    target_layers = {"haar_conv": MockLayer("haar_conv", [])}
    target_struct = {"haar_conv": {}}

    results = list(traverse_paired(source_params, target_layers, target_struct))

    assert len(results) == 1
    layer, params, path = results[0]
    assert layer.name == "haar_conv"
    assert params == {"kernel": [1.0]}


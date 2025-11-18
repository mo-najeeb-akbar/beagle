from __future__ import annotations

from typing import Any

from beagle.conversions.types import Layer, ParamDict, TransferFn, VerifyFn


class ConversionRegistry:
    def __init__(self) -> None:
        self._transfer: dict[str, TransferFn] = {}
        self._verify: dict[str, VerifyFn] = {}

    def register_transfer(self, layer_type: str, fn: TransferFn) -> None:
        self._transfer[layer_type] = fn

    def register_verify(self, layer_type: str, fn: VerifyFn) -> None:
        self._verify[layer_type] = fn

    def get_transfer(self, layer_type: str) -> TransferFn | None:
        return self._transfer.get(layer_type)

    def get_verify(self, layer_type: str) -> VerifyFn | None:
        return self._verify.get(layer_type)

    def has_transfer(self, layer_type: str) -> bool:
        return layer_type in self._transfer

    def has_verify(self, layer_type: str) -> bool:
        return layer_type in self._verify


def transfer_weights(
    target_model: Any,
    source_params: ParamDict,
    registry: ConversionRegistry,
    is_layer_fn: Any,
    extract_fn: Any,
    batch_stats: ParamDict | None = None,
) -> int:
    from beagle.conversions.extract import extract_layer_refs, extract_structure
    from beagle.conversions.traverse import traverse_paired

    if isinstance(source_params, dict) and "params" in source_params:
        params = source_params["params"]
        batch_stats = source_params.get("batch_stats", batch_stats)
    else:
        params = source_params

    target_struct = extract_structure(target_model, is_layer_fn)
    target_layers = extract_layer_refs(target_model, is_layer_fn)

    count = 0
    for layer, params_dict, path in traverse_paired(
        params, target_layers, target_struct, is_layer_fn=is_layer_fn
    ):
        count += 1
        layer_type = type(layer).__name__

        transfer_fn = registry.get_transfer(layer_type)
        if transfer_fn:
            transfer_fn(layer, params_dict)

    return count


from __future__ import annotations

import numpy as np

from beagle.conversions.types import (
    ConversionResult,
    ParamDict,
    Tolerance,
)


def verify_transfer(
    target_model: any,
    source_params: ParamDict,
    registry: any,
    is_layer_fn: any,
    tolerance: Tolerance = Tolerance(),
) -> ConversionResult:
    from beagle.conversions.extract import extract_layer_refs, extract_structure
    from beagle.conversions.traverse import traverse_paired

    if isinstance(source_params, dict) and "params" in source_params:
        params = source_params["params"]
    else:
        params = source_params

    target_struct = extract_structure(target_model, is_layer_fn)
    target_layers = extract_layer_refs(target_model, is_layer_fn)

    matches = []
    mismatches = []
    count = 0

    for layer, params_dict, path in traverse_paired(
        params, target_layers, target_struct, is_layer_fn=is_layer_fn
    ):
        count += 1
        layer_type = type(layer).__name__

        verify_fn = registry.get_verify(layer_type)
        if verify_fn:
            success, messages = verify_fn(layer, params_dict, tolerance)
            if success:
                matches.extend([f"{path} - {msg}" for msg in messages])
            else:
                mismatches.extend([f"{path} - {msg}" for msg in messages])

    return ConversionResult(
        matches=matches, mismatches=mismatches, layers_transferred=count
    )


def compare_arrays(
    actual: np.ndarray, expected: np.ndarray, name: str, tolerance: Tolerance
) -> tuple[bool, str]:
    try:
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance.rtol, atol=tolerance.atol
        )
        return True, name
    except AssertionError as e:
        return False, f"{name}: {str(e)}"


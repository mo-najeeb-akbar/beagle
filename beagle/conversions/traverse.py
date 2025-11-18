from __future__ import annotations

from typing import Any, Iterator

from beagle.conversions.types import Layer, NameMapping, ParamDict, StructDict


def create_name_mapping(source_dict: StructDict, target_dict: StructDict) -> NameMapping:
    mapping = {}
    source_keys = set(source_dict.keys())
    target_keys = set(target_dict.keys())

    for key in source_keys & target_keys:
        mapping[key] = key

    source_only = source_keys - target_keys
    target_only = target_keys - source_keys

    source_lower = {k.lower(): k for k in source_only}
    target_lower = {k.lower(): k for k in target_only}

    for lower_key in source_lower.keys() & target_lower.keys():
        mapping[source_lower[lower_key]] = target_lower[lower_key]

    for source_key in list(source_only):
        if source_key in mapping:
            continue

        if source_key.startswith("ResidualBlock_"):
            idx = source_key.split("_")[1]
            target_match = f"residual_blocks.{idx}"
            if target_match in target_only:
                mapping[source_key] = target_match
                continue

        if source_key.startswith("Conv_") and source_key[5:].isdigit():
            idx = source_key.split("_")[1]
            target_match = f"conv_layers.{idx}"
            if target_match in target_only:
                mapping[source_key] = target_match
                continue

    return mapping


def is_param_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    param_keys = {"kernel", "bias", "scale", "mean", "var"}
    return any(k in value for k in param_keys)


def traverse_paired(
    source_params: ParamDict,
    target_layers: dict[str, Layer],
    target_struct: StructDict,
    name_mapping: NameMapping | None = None,
    path: str = "",
    is_layer_fn: Any | None = None,
) -> Iterator[tuple[Layer, ParamDict, str]]:
    if name_mapping is None:
        name_mapping = create_name_mapping(source_params, target_struct)
    
    if is_layer_fn is None:
        is_layer_fn = lambda x: hasattr(x, "get_weights")

    for source_key, target_key in name_mapping.items():
        if target_key not in target_struct:
            continue

        current_path = f"{path}/{source_key}" if path else source_key
        source_value = source_params[source_key]
        target_value = target_struct[target_key]

        if is_param_dict(source_value):
            if target_key in target_layers:
                yield (target_layers[target_key], source_value, current_path)
        elif isinstance(source_value, dict):
            if (
                isinstance(target_value, dict)
                and not target_value
                and target_key in target_layers
                and len(source_value) == 1
            ):
                child_key = list(source_value.keys())[0]
                child_params = source_value[child_key]
                if is_param_dict(child_params):
                    yield (target_layers[target_key], child_params, current_path)
                else:
                    if isinstance(target_value, dict) and target_key in target_layers:
                        from beagle.conversions.extract import extract_layer_refs

                        nested_layers = extract_layer_refs(
                            target_layers[target_key],
                            is_layer_fn,
                        )
                        yield from traverse_paired(
                            source_value,
                            nested_layers,
                            target_value,
                            None,
                            current_path,
                            is_layer_fn,
                        )
            elif isinstance(target_value, dict) and target_key in target_layers:
                from beagle.conversions.extract import extract_layer_refs

                nested_layers = extract_layer_refs(
                    target_layers[target_key], is_layer_fn
                )
                yield from traverse_paired(
                    source_value, nested_layers, target_value, None, current_path, is_layer_fn
                )
        else:
            if target_key in target_layers:
                yield (target_layers[target_key], source_value, current_path)


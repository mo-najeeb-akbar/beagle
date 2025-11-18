from __future__ import annotations

from typing import Any, Callable

from beagle.conversions.types import Layer, StructDict


def extract_structure(
    model: Any,
    is_layer: Callable[[Any], bool],
    skip_attrs: set[str] | None = None,
) -> StructDict:
    if skip_attrs is None:
        skip_attrs = {"layers", "built", "trainable", "dtype"}

    result = {}

    attrs_to_check = []
    if hasattr(model, "__dataclass_fields__"):
        attrs_to_check = list(model.__dataclass_fields__.keys())
    else:
        attrs_to_check = [
            name for name in dir(model) if not name.startswith("_")
        ]

    for attr_name in attrs_to_check:
        if attr_name in skip_attrs:
            continue
        try:
            attr = getattr(model, attr_name)

            if is_layer(attr):
                nested = extract_structure(attr, is_layer, skip_attrs)
                result[attr_name] = nested if nested else {}
            elif isinstance(attr, (list, tuple)) and len(attr) > 0:
                if is_layer(attr[0]):
                    for i, layer in enumerate(attr):
                        nested = extract_structure(layer, is_layer, skip_attrs)
                        result[f"{attr_name}.{i}"] = nested if nested else {}
            elif hasattr(attr, "__iter__") and not isinstance(attr, (str, bytes)):
                try:
                    items = list(attr)
                    if items and is_layer(items[0]):
                        for i, layer in enumerate(items):
                            nested = extract_structure(layer, is_layer, skip_attrs)
                            result[f"{attr_name}.{i}"] = nested if nested else {}
                except:
                    pass
        except:
            pass

    return result


def extract_layer_refs(
    model: Any, is_layer: Callable[[Any], bool], skip_attrs: set[str] | None = None
) -> dict[str, Layer]:
    if skip_attrs is None:
        skip_attrs = {"layers", "built", "trainable", "dtype"}

    refs = {}
    
    attrs_to_check = []
    if hasattr(model, "__dataclass_fields__"):
        attrs_to_check = list(model.__dataclass_fields__.keys())
    else:
        attrs_to_check = [
            name for name in dir(model) if not name.startswith("_")
        ]
    
    for attr_name in attrs_to_check:
        if attr_name in skip_attrs:
            continue
        try:
            attr = getattr(model, attr_name)
            if is_layer(attr):
                refs[attr_name] = attr
            elif isinstance(attr, (list, tuple)) and len(attr) > 0:
                if is_layer(attr[0]):
                    for i, layer in enumerate(attr):
                        refs[f"{attr_name}.{i}"] = layer
            elif hasattr(attr, "__iter__") and not isinstance(attr, (str, bytes)):
                try:
                    items = list(attr)
                    if items and is_layer(items[0]):
                        for i, layer in enumerate(items):
                            refs[f"{attr_name}.{i}"] = layer
                except:
                    pass
        except:
            pass

    return refs


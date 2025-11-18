from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf


def export_model_to_savedmodel(
    model: tf.keras.Model,
    output_path: str | Path,
) -> None:
    """Export a Keras model to TensorFlow SavedModel format (I/O side effect).
    
    Args:
        model: Keras model to export
        output_path: Directory path where the SavedModel will be saved
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path) + '.keras')


def export_model_to_tfjs(
    model: tf.keras.Model,
    output_path: str | Path,
    quantization_dtype: str | None = None,
) -> None:
    """Export a Keras model to TensorFlow.js format (I/O side effect).
    
    WARNING: tensorflowjs has dependency conflicts (tensorflow-decision-forests).
    Consider using export_model_to_savedmodel() instead and converting with CLI:
        tensorflowjs_converter --input_format=tf_saved_model saved_model_dir/ output_dir/
    
    Args:
        model: Keras model to export
        output_path: Directory path where the TFJS model will be saved
        quantization_dtype: Optional quantization ('uint8' or 'uint16')
    """
    import tensorflowjs as tfjs
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    kwargs: dict[str, Any] = {}
    if quantization_dtype:
        kwargs["quantization_dtype"] = quantization_dtype
    
    tfjs.converters.save_keras_model(model, str(output_path), **kwargs)


def extract_submodel(
    model: tf.keras.Model,
    layer_name: str,
    input_shape: tuple[int, ...] | None = None,
) -> tf.keras.Model:
    """Extract a sublayer as a standalone model (pure function).
    
    Args:
        model: Parent Keras model containing the sublayer
        layer_name: Name of the layer to extract
        input_shape: Optional input shape for the sublayer. If provided, creates
            a new model by calling the layer. If None, tries to use existing
            layer input/output.
        
    Returns:
        Standalone Keras model wrapping the sublayer
        
    Raises:
        ValueError: If layer cannot be found in the model
        AttributeError: If layer has no defined input and input_shape not provided
    """
    target_layer = None
    
    for layer in model.layers:
        if layer.name == layer_name:
            target_layer = layer
            break
    
    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    if input_shape is not None:
        new_input = tf.keras.Input(shape=input_shape)
        new_output = target_layer(new_input)
        return tf.keras.Model(inputs=new_input, outputs=new_output, name=layer_name)
    
    try:
        layer_input = target_layer.input
        layer_output = target_layer.output
        return tf.keras.Model(
            inputs=layer_input, outputs=layer_output, name=layer_name
        )
    except AttributeError as e:
        raise AttributeError(
            f"Layer '{layer_name}' has no defined input. "
            f"Provide 'input_shape' parameter to create a new model. "
            f"Original error: {e}"
        ) from e


def export_submodels_to_savedmodel(
    model: tf.keras.Model,
    layer_names: list[str],
    output_dir: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Path]:
    """Export multiple sublayers as separate SavedModel models (I/O side effect).
    
    Args:
        model: Parent Keras model containing sublayers
        layer_names: Names of layers to extract and export
        output_dir: Base directory for exports
        input_shapes: Optional dict mapping layer names to their input shapes.
            Required for layers that haven't been called independently.
        
    Returns:
        Dict mapping layer names to their export paths
    """
    output_dir = Path(output_dir)
    export_paths: dict[str, Path] = {}
    
    for layer_name in layer_names:
        input_shape = input_shapes.get(layer_name) if input_shapes else None
        submodel = extract_submodel(model, layer_name, input_shape)
        layer_path = output_dir / layer_name
        export_model_to_savedmodel(submodel, layer_path)
        export_paths[layer_name] = layer_path
    
    return export_paths


def export_submodels_to_tfjs(
    model: tf.keras.Model,
    layer_names: list[str],
    output_dir: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    quantization_dtype: str | None = None,
) -> dict[str, Path]:
    """Export multiple sublayers as separate TFJS models (I/O side effect).
    
    WARNING: tensorflowjs has dependency conflicts. Consider using
    export_submodels_to_savedmodel() instead.
    
    Args:
        model: Parent Keras model containing sublayers
        layer_names: Names of layers to extract and export
        output_dir: Base directory for exports
        input_shapes: Optional dict mapping layer names to their input shapes.
            Required for layers that haven't been called independently.
        quantization_dtype: Optional quantization ('uint8' or 'uint16')
        
    Returns:
        Dict mapping layer names to their export paths
    """
    output_dir = Path(output_dir)
    export_paths: dict[str, Path] = {}
    
    for layer_name in layer_names:
        input_shape = input_shapes.get(layer_name) if input_shapes else None
        submodel = extract_submodel(model, layer_name, input_shape)
        layer_path = output_dir / layer_name
        export_model_to_tfjs(submodel, layer_path, quantization_dtype)
        export_paths[layer_name] = layer_path
    
    return export_paths


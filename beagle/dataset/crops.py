from __future__ import annotations

from typing import Callable
import tensorflow as tf
import numpy as np


def create_overlapping_crops(
    data_dict: dict[str, tf.Tensor],
    crop_size: int = 256,
    stride: int = 192,
) -> dict[str, tf.Tensor]:
    """
    Create overlapping crops from image-like fields (pure function with TF graph ops).
    
    Automatically detects which field(s) to crop by looking for 3D tensors (H, W, C).
    Non-image fields (scalars, vectors) are replicated for each crop.
    
    Args:
        data_dict: Dictionary containing tensors (at least one 3D image-like tensor)
        crop_size: Size of each crop (e.g., 256)
        stride: Step size between crops (e.g., 192 gives 64px overlap)
    
    Returns:
        Dictionary with all fields batched by number of crops
    """
    # Find all 3D tensors (image-like fields) to crop
    image_fields = []
    reference_field = None
    reference_tensor = None
    
    for key, tensor in data_dict.items():
        # Use ndims which is known at graph construction time
        if tensor.shape.ndims == 3:  # (H, W, C) format
            image_fields.append(key)
            if reference_field is None:
                reference_field = key
                reference_tensor = tensor
    
    if reference_tensor is None:
        raise ValueError(
            "No 3D tensor found in data_dict for cropping. "
            "Need at least one (H, W, C) tensor. "
            f"Found fields: {list(data_dict.keys())}"
        )
    
    # Use reference tensor to compute crop dimensions
    image = tf.expand_dims(reference_tensor, 0)  # [1, H, W, C]
    
    # Extract patches from reference to get dimensions
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, crop_size, crop_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    patches_shape = tf.shape(patches)
    num_rows = patches_shape[1]
    num_cols = patches_shape[2]
    num_crops = num_rows * num_cols
    
    # Build result dict
    result = {}
    
    # Crop all 3D tensors (image-like fields)
    for field_name in image_fields:
        field_tensor = data_dict[field_name]
        field_expanded = tf.expand_dims(field_tensor, 0)
        
        field_patches = tf.image.extract_patches(
            images=field_expanded,
            sizes=[1, crop_size, crop_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        num_channels = tf.shape(field_tensor)[2]
        field_crops = tf.reshape(field_patches, [-1, crop_size, crop_size, num_channels])
        result[field_name] = field_crops
    
    # Add position metadata
    row_indices = tf.repeat(tf.range(num_rows), num_cols)
    col_indices = tf.tile(tf.range(num_cols), [num_rows])
    
    result['row_idx'] = row_indices
    result['col_idx'] = col_indices
    result['num_rows'] = tf.fill([num_crops], num_rows)
    result['num_cols'] = tf.fill([num_crops], num_cols)
    
    # Replicate non-image fields (labels, vectors, etc.) for each crop
    for key, tensor in data_dict.items():
        if key not in result:
            result[key] = tf.tile([tensor], [num_crops])
    
    return result


def reconstruct_from_crops(
    crops: dict[tuple[int, int], np.ndarray],
    num_rows: int,
    num_cols: int,
    crop_size: int,
    stride: int,
) -> np.ndarray:
    """
    Reconstruct full image from crops using averaging in overlap regions (pure function).
    
    Args:
        crops: Dict of {(row_idx, col_idx): crop_array}
        num_rows: Number of crop rows
        num_cols: Number of crop columns
        crop_size: Size of each crop
        stride: Stride between crops
    
    Returns:
        Reconstructed image array
    """
    # Calculate output dimensions
    height = (num_rows - 1) * stride + crop_size
    width = (num_cols - 1) * stride + crop_size
    channels = next(iter(crops.values())).shape[2]
    
    # Initialize output and weight arrays for averaging overlaps
    output = np.zeros((height, width, channels), dtype=np.float32)
    weights = np.zeros((height, width, channels), dtype=np.float32)
    
    # Place each crop
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if (row_idx, col_idx) not in crops:
                continue
            
            crop = crops[(row_idx, col_idx)]
            
            # Calculate position
            y_start = row_idx * stride
            x_start = col_idx * stride
            y_end = y_start + crop_size
            x_end = x_start + crop_size
            
            # Add crop to output with weights
            output[y_start:y_end, x_start:x_end, :] += crop
            weights[y_start:y_end, x_start:x_end, :] += 1.0
    
    # Average overlapping regions
    output = output / np.maximum(weights, 1.0)
    
    return output


def compute_crop_stats(
    crops: dict[tuple[int, int], np.ndarray]
) -> tuple[float, float]:
    """
    Compute mean and std from crop dictionary (pure function).
    
    Args:
        crops: Dict of {(row_idx, col_idx): crop_array}
    
    Returns:
        (mean, std) tuple
    """
    all_values = np.concatenate([crop.flatten() for crop in crops.values()])
    # Remove NaNs
    all_values = all_values[~np.isnan(all_values)]
    
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))
    
    return mean, std


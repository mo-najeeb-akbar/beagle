"""
Composable augmentation utilities.

Makes it easy to:
1. Write custom augmentations as simple functions
2. Compose multiple augmentations
3. Use library augmentations (albumentations, imgaug, etc.)
"""
from __future__ import annotations

from typing import Callable, Any
import tensorflow as tf


# Type alias for augmentation functions
AugmentFn = Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]]


def compose(*augment_fns: AugmentFn) -> AugmentFn:
    """
    Compose multiple augmentation functions (pure).
    
    Args:
        *augment_fns: Variable number of augmentation functions
    
    Returns:
        Composed augmentation function
    
    Example:
        >>> flip = random_flip_left_right()
        >>> rotate = random_rotate_90()
        >>> brightness = random_brightness(0.2)
        >>> augment = compose(flip, rotate, brightness)
        >>> data = augment({'image': img_tensor})
    """
    def composed(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        result = data_dict
        for fn in augment_fns:
            result = fn(result)
        return result
    
    return composed


def apply_to_field(
    field_name: str,
    transform_fn: Callable[[tf.Tensor], tf.Tensor],
) -> AugmentFn:
    """
    Create augmentation that applies transform to a specific field (pure).
    
    Args:
        field_name: Name of field to transform
        transform_fn: Function that transforms a single tensor
    
    Returns:
        Augmentation function that works on data dicts
    
    Example:
        >>> # Apply normalization to 'image' field only
        >>> normalize = apply_to_field('image', lambda x: x / 255.0)
        >>> data = normalize({'image': img, 'mask': mask})
        >>> # Only 'image' is transformed, 'mask' unchanged
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if field_name in data_dict:
            data_dict[field_name] = transform_fn(data_dict[field_name])
        return data_dict
    
    return augment


def apply_to_fields(
    field_names: list[str],
    transform_fn: Callable[[tf.Tensor], tf.Tensor],
) -> AugmentFn:
    """
    Apply same transform to multiple fields (e.g., image and mask) (pure).
    
    Args:
        field_names: List of field names to transform
        transform_fn: Function that transforms a single tensor
    
    Returns:
        Augmentation function that applies transform to all specified fields
    
    Example:
        >>> # Apply same geometric transform to image and mask
        >>> flip_both = apply_to_fields(
        ...     ['image', 'mask'],
        ...     lambda x: tf.image.random_flip_left_right(x)
        ... )
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        for field in field_names:
            if field in data_dict:
                data_dict[field] = transform_fn(data_dict[field])
        return data_dict
    
    return augment


def apply_geometric(
    transform_fn: Callable[[tf.Tensor, tf.Tensor | None], tuple[tf.Tensor, tf.Tensor | None]],
    image_field: str = 'image',
    mask_field: str | None = 'mask',
) -> AugmentFn:
    """
    Apply geometric transform to both image and mask (pure).
    
    Ensures image and mask receive the same geometric transformation
    (critical for segmentation tasks).
    
    Args:
        transform_fn: Function that takes (image, mask) and returns (image, mask)
        image_field: Name of image field (default: 'image')
        mask_field: Name of mask field (default: 'mask', None to skip)
    
    Returns:
        Augmentation function
    
    Example:
        >>> def random_flip(img, mask):
        ...     if tf.random.uniform([]) > 0.5:
        ...         img = tf.image.flip_left_right(img)
        ...         if mask is not None:
        ...             mask = tf.image.flip_left_right(mask)
        ...     return img, mask
        >>> 
        >>> flip = apply_geometric(random_flip)
        >>> data = flip({'image': img, 'mask': mask})
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if image_field not in data_dict:
            return data_dict
        
        img = data_dict[image_field]
        mask = data_dict.get(mask_field) if mask_field else None
        
        img_transformed, mask_transformed = transform_fn(img, mask)
        
        data_dict[image_field] = img_transformed
        if mask_field and mask_transformed is not None:
            data_dict[mask_field] = mask_transformed
        
        return data_dict
    
    return augment


# Common augmentation builders (returns TensorFlow augmentation functions)

def random_flip_left_right(
    fields: list[str] | None = None,
    image_field: str = 'image',
    mask_field: str | None = 'mask',
) -> AugmentFn:
    """
    Random horizontal flip (pure).
    
    Args:
        fields: If provided, apply to these specific fields
                If None, uses geometric mode (image + mask)
        image_field: Image field name for geometric mode
        mask_field: Mask field name for geometric mode
    
    Returns:
        Augmentation function
    """
    if fields is not None:
        return apply_to_fields(
            fields,
            lambda x: tf.image.random_flip_left_right(x),
        )
    
    def transform(img: tf.Tensor, mask: tf.Tensor | None) -> tuple[tf.Tensor, tf.Tensor | None]:
        # Use same random value for both
        do_flip = tf.random.uniform([]) > 0.5
        if do_flip:
            img = tf.image.flip_left_right(img)
            if mask is not None:
                mask = tf.image.flip_left_right(mask)
        return img, mask
    
    return apply_geometric(transform, image_field, mask_field)


def random_flip_up_down(
    fields: list[str] | None = None,
    image_field: str = 'image',
    mask_field: str | None = 'mask',
) -> AugmentFn:
    """Random vertical flip (pure)."""
    if fields is not None:
        return apply_to_fields(
            fields,
            lambda x: tf.image.random_flip_up_down(x),
        )
    
    def transform(img: tf.Tensor, mask: tf.Tensor | None) -> tuple[tf.Tensor, tf.Tensor | None]:
        do_flip = tf.random.uniform([]) > 0.5
        if do_flip:
            img = tf.image.flip_up_down(img)
            if mask is not None:
                mask = tf.image.flip_up_down(mask)
        return img, mask
    
    return apply_geometric(transform, image_field, mask_field)


def random_rotate_90(
    fields: list[str] | None = None,
    image_field: str = 'image',
    mask_field: str | None = 'mask',
) -> AugmentFn:
    """Random 90-degree rotation (pure)."""
    if fields is not None:
        def rotate(x: tf.Tensor) -> tf.Tensor:
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            return tf.image.rot90(x, k=k)
        return apply_to_fields(fields, rotate)
    
    def transform(img: tf.Tensor, mask: tf.Tensor | None) -> tuple[tf.Tensor, tf.Tensor | None]:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        if mask is not None:
            mask = tf.image.rot90(mask, k=k)
        return img, mask
    
    return apply_geometric(transform, image_field, mask_field)


def random_brightness(
    max_delta: float,
    field: str = 'image',
) -> AugmentFn:
    """Random brightness adjustment (pure)."""
    return apply_to_field(
        field,
        lambda x: tf.image.random_brightness(x, max_delta),
    )


def random_contrast(
    lower: float,
    upper: float,
    field: str = 'image',
) -> AugmentFn:
    """Random contrast adjustment (pure)."""
    return apply_to_field(
        field,
        lambda x: tf.image.random_contrast(x, lower, upper),
    )


def clip_values(
    min_val: float = 0.0,
    max_val: float = 1.0,
    field: str = 'image',
) -> AugmentFn:
    """Clip values to range (useful after augmentations) (pure)."""
    return apply_to_field(
        field,
        lambda x: tf.clip_by_value(x, min_val, max_val),
    )


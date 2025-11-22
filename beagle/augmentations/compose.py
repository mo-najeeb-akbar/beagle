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
            return {**data_dict, field_name: transform_fn(data_dict[field_name])}
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
        updates = {
            field: transform_fn(data_dict[field])
            for field in field_names
            if field in data_dict
        }
        return {**data_dict, **updates}
    
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
        
        updates = {image_field: img_transformed}
        if mask_field and mask_transformed is not None:
            updates[mask_field] = mask_transformed
        
        return {**data_dict, **updates}
    
    return augment


def apply_same_transform_to_all(
    fields: list[str],
    transform_fn: Callable[[list[tf.Tensor]], list[tf.Tensor]],
) -> AugmentFn:
    """
    Apply same random transform to multiple fields consistently (pure).
    
    Useful for geometric augmentations where you need the same random
    transformation applied to images, masks, depth maps, etc.
    
    Args:
        fields: List of field names to transform together
        transform_fn: Function that takes list of tensors and returns list
                      The same random parameters should be used for all tensors
    
    Returns:
        Augmentation function
    
    Example:
        >>> def consistent_flip(tensors):
        ...     # Same random decision for all tensors
        ...     do_flip = tf.random.uniform([]) > 0.5
        ...     if do_flip:
        ...         return [tf.image.flip_left_right(t) for t in tensors]
        ...     return tensors
        >>> 
        >>> flip = apply_same_transform_to_all(['image', 'mask', 'depth'], consistent_flip)
        >>> data = flip({'image': img, 'mask': mask, 'depth': depth_map})
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        present_fields = [f for f in fields if f in data_dict]
        if not present_fields:
            return data_dict
        
        tensors = [data_dict[f] for f in present_fields]
        transformed = transform_fn(tensors)
        
        updates = dict(zip(present_fields, transformed))
        return {**data_dict, **updates}
    
    return augment


# Common augmentation builders (returns TensorFlow augmentation functions)

def random_flip_left_right(
    fields: list[str] = ['image'],
    probability: float = 0.5,
) -> AugmentFn:
    """
    Random horizontal flip applied consistently to all specified fields (pure).
    
    Args:
        fields: List of field names to flip (e.g., ['image', 'mask', 'depth'])
        probability: Probability of applying flip (default: 0.5)
    
    Returns:
        Augmentation function
    
    Example:
        >>> # Flip image and mask together
        >>> flip = random_flip_left_right(fields=['image', 'mask'])
        >>> data = flip({'image': img, 'mask': mask})
    """
    def transform(tensors: list[tf.Tensor]) -> list[tf.Tensor]:
        do_flip = tf.random.uniform([]) < probability
        if do_flip:
            return [tf.image.flip_left_right(t) for t in tensors]
        return tensors
    
    return apply_same_transform_to_all(fields, transform)


def random_flip_up_down(
    fields: list[str] = ['image'],
    probability: float = 0.5,
) -> AugmentFn:
    """
    Random vertical flip applied consistently to all specified fields (pure).
    
    Args:
        fields: List of field names to flip (e.g., ['image', 'mask', 'depth'])
        probability: Probability of applying flip (default: 0.5)
    
    Returns:
        Augmentation function
    """
    def transform(tensors: list[tf.Tensor]) -> list[tf.Tensor]:
        do_flip = tf.random.uniform([]) < probability
        if do_flip:
            return [tf.image.flip_up_down(t) for t in tensors]
        return tensors
    
    return apply_same_transform_to_all(fields, transform)


def random_rotate_90(
    fields: list[str] = ['image'],
) -> AugmentFn:
    """
    Random 90-degree rotation applied consistently to all specified fields (pure).
    
    Args:
        fields: List of field names to rotate (e.g., ['image', 'mask', 'depth'])
    
    Returns:
        Augmentation function
    """
    def transform(tensors: list[tf.Tensor]) -> list[tf.Tensor]:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        return [tf.image.rot90(t, k=k) for t in tensors]
    
    return apply_same_transform_to_all(fields, transform)


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


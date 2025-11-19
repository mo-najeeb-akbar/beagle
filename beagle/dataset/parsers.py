from __future__ import annotations

from typing import Callable
import tensorflow as tf


def make_default_image_parser(
    grayscale: bool = True,
) -> Callable[[tf.Tensor], dict[str, tf.Tensor]]:
    """
    Create default parser for standard image TFRecords (pure).
    
    Args:
        grayscale: Whether to convert to grayscale
    
    Returns:
        Parser function that returns dict with 'image' field
    """
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(
            example_proto, {"image": tf.io.FixedLenFeature([], tf.string)}
        )
        img = tf.io.decode_image(parsed["image"], channels=3)
        
        if grayscale:
            img = tf.image.rgb_to_grayscale(img)
        
        img = tf.cast(img, tf.float32) / 255.0
        
        return {"image": img}
    
    return parse


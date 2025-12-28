from __future__ import annotations

from pathlib import Path
from functools import partial

import tensorflow as tf
import glob

from beagle.dataset import (
    build_dataset_pipeline,
    compute_fields_mean_std,
    save_field_stats,
    load_field_stats,
    load_tfr_dict,
)
from beagle.augmentations import (
    compose,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
)


def make_nema_vae_parser(
    feature_dict: dict,
    shape_dict: dict
) -> callable:
    """Create parser for polymer TFRecords."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:

        feature_spec = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }   
        parsed = tf.io.parse_single_example(example_proto, feature_spec)
        
        # Decode PNG images
        image = tf.io.decode_png(parsed['image'], channels=1)
        image = tf.cast(image, tf.float32)
        
        return {'image': image}
    return parse


def gaussian_noise(stddev: float, field: str = 'depth') -> callable:
    """
    Add Gaussian noise for adversarial robustness (pure).
   
    Args:
        stddev: Standard deviation of noise (for standardized data, ~0.05-0.1)
        field: Field name to apply noise to
       
    Returns:
        Augmentation function
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if field in data_dict:
            noise = tf.random.normal(
                shape=tf.shape(data_dict[field]),
                mean=0.0,
                stddev=stddev,
                dtype=data_dict[field].dtype
            )
            # KEY CHANGE: Return new dict instead of mutating
            return {**data_dict, field: data_dict[field] + noise}
        return data_dict
    return augment


def create_nema_vae_iterator(
    data_dir: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> tuple:

    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    files = sorted(glob.glob(tfrecord_pattern))
    json_path = str(data_dir / "nema_vae.json")
    
    # Load schema
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_nema_vae_parser(feature_dict, shape_dict)
    
    img_height, img_width, _ = shape_dict['image'] 
    
    def z_score_norm(tensor: tf.Tensor) -> tf.Tensor:
        # return (tensor - stats['depth'][0]) / stats['depth'][1]
        return (tensor  - 127.5) / 127.5

    field_configs = {
        'image': z_score_norm
    }
    
    # Optional augmentation
    aug_fn = None
    if augment:
        aug_fn = compose(
            random_flip_left_right(fields=['image']),
            random_flip_up_down(fields=['image']),
            random_rotate_90(fields=['image']),
        )
    
    # Create iterator with optional train/val split
    iterator, batches_per_epoch = build_dataset_pipeline(
        files=files,
        parser=parser,
        field_configs=field_configs,
        batch_size=batch_size,
        crop_size=None,
        stride=None,
        augment_fn=aug_fn,
        shuffle=shuffle,
        repeat=True,
        image_shape=(img_height, img_width),
    )
    
    return iterator, batches_per_epoch


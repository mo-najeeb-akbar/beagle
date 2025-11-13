# Polymer Dataset Example

This example demonstrates loading custom TFRecords with overlapping crops.

## Files

- `polymer_writer.py` - Converts polymer depth maps to TFRecords
- `polymer_loader.py` - Loads TFRecords with crops and augmentation

## Usage

### 1. Write TFRecords

```bash
make run CMD='python examples/polymer_writer.py /data/input /data/output'
```

### 2. Compute Statistics

```bash
make run CMD='python examples/polymer_loader.py /data/output --compute-stats'
```

Output example:
```
Mean: 123.456789
Std:  45.678901
Use: precomputed_stats=(123.456789, 45.678901)
```

### 3. Create Dataloader

```python
from beagle.dataset import load_tfr_dict, create_crop_iterator
from beagle.augmentations import create_transform, MODERATE_AUGMENT, apply_transform
import tensorflow as tf

# Load schema and create parser
feature_dict, shape_dict = load_tfr_dict("data/polymer.json")

def make_parser(feature_dict, shape_dict):
    def parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        surface = tf.io.parse_tensor(parsed['surface'], out_type=tf.float32)
        surface = tf.reshape(surface, shape_dict['surface'] + [1])
        surface = tf.where(tf.math.is_nan(surface), 0.0, surface)
        return {'image': surface}
    return parse

parser = make_parser(feature_dict, shape_dict)

# Create augmentation
transform = create_transform(MODERATE_AUGMENT)
aug_fn = lambda img: apply_transform(transform, img)['image']

# Create iterator
iterator, n_batches = create_crop_iterator(
    tfrecord_pattern="data/*.tfrecord",
    parser=parser,
    crop_size=256,
    stride=192,
    batch_size=32,
    augment_fn=aug_fn,
    precomputed_stats=(123.456, 45.678),  # From step 2
    shuffle=True,
)

# Use in training loop
for epoch in range(10):
    for _ in range(n_batches):
        batch = next(iterator)
        # train_step(batch['image'])
```

## Key Features

- **Overlapping crops**: Generate 256x256 crops with 192px stride (64px overlap)
- **Statistics**: Welford's algorithm for stable mean/std computation
- **Augmentation**: Random flips, rotations, brightness, contrast, noise
- **JAX arrays**: Automatic TensorFlow → JAX conversion
- **Efficient**: Caching, prefetching, parallel processing

## Library Functions Used

From `beagle.dataset`:
- `load_tfr_dict()` - Load TFRecord schema
- `compute_welford_stats()` - Compute global mean/std
- `create_crop_iterator()` - Create dataloader with crops
- `to_jax()` - Convert TensorFlow → JAX

From `beagle.augmentations`:
- `create_transform()` - Build augmentation pipeline
- `apply_transform()` - Apply augmentations
- `MODERATE_AUGMENT` - Preset config

## Customization

### Different crop sizes

```python
iterator, _ = create_crop_iterator(
    crop_size=512,
    stride=384,  # 128px overlap
    ...
)
```

### No augmentation

```python
iterator, _ = create_crop_iterator(
    augment_fn=None,
    ...
)
```

### Custom augmentation

```python
from beagle.augmentations import AugmentConfig, create_transform

config = AugmentConfig(
    flip_horizontal=True,
    flip_vertical=False,
    rotate_90=True,
    brightness_limit=0.2,
)
transform = create_transform(config)
aug_fn = lambda img: apply_transform(transform, img)['image']
```


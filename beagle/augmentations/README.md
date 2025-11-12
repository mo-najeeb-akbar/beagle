# Augmentations Module

Functional image augmentation pipeline built on albumentations.

## Quick Start

```python
from beagle.augmentations import create_transform, apply_transform, MODERATE_AUGMENT
import numpy as np

# Create transform from preset
transform = create_transform(MODERATE_AUGMENT)

# Apply to image (H, W, C) in [0, 1]
image = np.random.rand(256, 256, 3).astype(np.float32)
result = apply_transform(transform, image)
augmented = result['image']

# With mask for segmentation
mask = np.random.rand(256, 256).astype(np.float32)
result = apply_transform(transform, image, mask=mask)
augmented_image = result['image']
augmented_mask = result['mask']
```

## Presets

```python
from beagle.augmentations import MINIMAL_AUGMENT, MODERATE_AUGMENT, HEAVY_AUGMENT

# MINIMAL: Only flips and 90° rotations
# MODERATE: + rotation (±15°), shifts, brightness/contrast
# HEAVY: + aggressive rotation (±45°), hue/saturation, noise, blur
```

## Custom Configuration

```python
from beagle.augmentations import AugmentConfig, create_transform

config = AugmentConfig(
    flip_horizontal=True,
    flip_vertical=True,
    rotate_90=True,
    rotation_limit=15.0,           # degrees
    shift_limit=0.1,                # fraction of image
    scale_limit=(0.9, 1.1),         # (min, max)
    brightness_limit=0.1,
    contrast_limit=(0.9, 1.1),
    hue_shift_limit=20.0,
    saturation_limit=(0.8, 1.2),
    gaussian_noise_var=0.01,
    gaussian_blur_limit=(3, 7),
    geometric_prob=1.0,
    color_prob=0.8,
    noise_prob=0.5,
)

transform = create_transform(config)
```

## Integration with TFRecord Pipeline

```python
from beagle.dataset import create_tfrecord_iterator
from beagle.augmentations import create_transform, apply_transform, MODERATE_AUGMENT

transform = create_transform(MODERATE_AUGMENT)
augment_fn = lambda img: apply_transform(transform, img)['image']

iterator, n_batches = create_tfrecord_iterator(
    "data/*.tfrecord",
    batch_size=32,
    augment_fn=augment_fn,
)

for batch in iterator:
    # batch['image'] is augmented JAX array
    pass
```

## Available Transforms

**Geometric**: Horizontal/vertical flips, 90° rotations, arbitrary rotation, shift, scale  
**Color**: Brightness, contrast, hue, saturation  
**Noise**: Gaussian noise, Gaussian blur

## Design

- **Immutable**: Configs are frozen, transforms are stateless
- **Composable**: Build complex from simple
- **Type-safe**: Full type hints
- **Functional**: Separate creation from application

## Dependencies

Requires `albumentations` (included in examples Docker image):

```bash
pip install albumentations
# or
make examples
```

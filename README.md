# Beagle

Functional dataset utilities for TFRecord creation/loading with JAX, training loops, and augmentations.

## Quick Start

```python
from beagle.dataset import Datum, write_dataset, serialize_float_array
from beagle.dataset.types import identity
import numpy as np

# Create and write TFRecords
data = [[
    Datum(
        name="features",
        value=np.array([1.0, 2.0, 3.0]),
        serialize_fn=serialize_float_array,
        decompress_fn=identity
    )
]]
write_dataset(data, "output_dir", num_shards=1)

# Load as JAX iterator
from beagle.dataset import create_tfrecord_iterator
iterator, n_batches = create_tfrecord_iterator("output_dir/*.tfrecord", batch_size=32)
for batch in iterator:
    print(batch['features'])  # JAX array
```

## Features

- **Dataset**: TFRecord writing/reading with custom serialization, multi-shard parallel writing
- **Training**: Functional training loops with checkpointing, metrics tracking, validation
- **Augmentations**: Image augmentation pipeline (albumentations) with immutable configs
- **Network**: VAE, U-Net, HRNet, ViT, receptive field utilities
- **Functional**: Immutable data structures, pure functions, composable pipelines

## Installation

```bash
git clone <repository>
cd beagle
make build  # Build Docker images
make shell  # Development shell
```

## Docker Workflow

**All code runs in Docker for consistency.**

```bash
# Core commands
make build      # Build images (after changing requirements.txt)
make shell      # Interactive dev shell
make test       # Run tests (94% coverage)
make coverage   # Tests with coverage report
make examples   # Shell with example dependencies (opencv, albumentations, ultralytics)

# Run arbitrary commands
make run CMD='python examples/root_writer.py /data/input /data/output'
make run CMD='pytest tests/test_loader.py -v'

# Mount external data
MOUNT_DIR=~/datasets make shell  # Available at /data in container
MOUNT_DIR=~/datasets MOUNT_TARGET=/input make shell

# GPU support (requires nvidia-docker2)
make run CMD='python -c "import jax; print(jax.devices())"'  # Verify GPU
NVIDIA_VISIBLE_DEVICES=0 make shell  # Use specific GPU
NVIDIA_VISIBLE_DEVICES="" make shell  # CPU only
```

## Usage Examples

### Training Loop

```python
from beagle.training import train_loop
import jax

@jax.jit
def train_step(state, batch, rng_key):
    # Your training logic
    return new_state, {'loss': loss}

def data_iterator_fn():
    for batch in dataloader:
        yield batch

final_state, history = train_loop(
    state=initial_state,
    train_step_fn=train_step,
    data_iterator_fn=data_iterator_fn,
    num_epochs=100,
    num_batches=50,
    rng_key=key,
    checkpoint_dir='/checkpoints',
)
```

### Augmentations

```python
from beagle.augmentations import create_transform, apply_transform, MODERATE_AUGMENT

transform = create_transform(MODERATE_AUGMENT)
augmented = apply_transform(transform, image)['image']

# With mask for segmentation
result = apply_transform(transform, image, mask=mask)
augmented_image, augmented_mask = result['image'], result['mask']
```

## Module Documentation

- `beagle/training/README.md` - Training loop API
- `beagle/augmentations/README.md` - Augmentation pipeline
- `docs/receptive_field_guide.md` - Receptive field computation

## Examples

See `examples/` for complete working examples:
- `root_writer.py` - Image dataset with FastSAM preprocessing
- `polymer_writer.py` - Depth map dataset creation
- `tfrecord_to_jax_example.py` - TFRecord → JAX iterator
- `training_example.py` - Full training loop example

## Testing

```bash
make test                                          # All tests
make run CMD='pytest tests/test_loader.py -v'     # Specific test
make run CMD='pytest --hypothesis-show-statistics' # Property-based stats
```

## License

MIT

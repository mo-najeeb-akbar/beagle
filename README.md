# Beagle

Write and train custom JAX models on datasets with minimal friction.

## What It Does

- **Dataset I/O**: Write/read TFRecords with custom serialization, load as JAX iterators
- **Training**: Functional training loops with checkpointing, metrics, validation
- **Augmentations**: Image augmentation pipeline with immutable configs
- **Networks**: VAE, U-Net, HRNet, ViT, receptive field utilities

Future: WebGPU conversion for zero-cost inference with ultra-minimal latency.

## Quick Start

```bash
git clone <repository>
cd beagle
make build  # Build Docker images
make shell  # Development shell
```

### Write TFRecords

```python
from beagle.dataset import Datum, write_dataset, serialize_float_array
from beagle.dataset.types import identity
import numpy as np

data = [[
    Datum(
        name="features",
        value=np.array([1.0, 2.0, 3.0]),
        serialize_fn=serialize_float_array,
        decompress_fn=identity
    )
]]
write_dataset(data, "output_dir", num_shards=1)
```

### Load as JAX Iterator

```python
from beagle.dataset import create_tfrecord_iterator

iterator, n_batches = create_tfrecord_iterator(
    "output_dir/*.tfrecord",
    batch_size=32
)

for batch in iterator:
    print(batch['features'])  # JAX array
```

### Train Models

```python
from beagle.training import train_loop
import jax

@jax.jit
def train_step(state, batch, rng_key):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch['x'])
        return jnp.mean((preds - batch['y']) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), {'loss': loss}

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
```

## Docker Workflow

```bash
# Core commands
make build      # Build images
make shell      # Interactive dev shell
make test       # Run tests (94% coverage)
make examples   # Shell with example dependencies

# Run arbitrary commands
make run CMD='python examples/root_writer.py /data/input /data/output'

# Mount external data
MOUNT_DIR=~/datasets make shell  # Available at /data

# GPU support (requires nvidia-docker2)
make run CMD='python -c "import jax; print(jax.devices())"'
NVIDIA_VISIBLE_DEVICES=0 make shell
```

## Examples

See `examples/` for complete working examples:
- `root_writer.py` - Image dataset with FastSAM preprocessing
- `polymer_writer.py` - Depth map dataset creation
- `tfrecord_to_jax_example.py` - TFRecord → JAX iterator
- `training_example.py` - Full training loop

## Design Philosophy

- **Functional first**: Immutable data structures, pure functions, composable pipelines
- **Type-safe**: Full type hints throughout
- **Minimal**: Clean abstractions without magic
- **Docker**: Consistent development environment

## Testing

```bash
make test
make run CMD='pytest tests/test_loader.py -v'
```
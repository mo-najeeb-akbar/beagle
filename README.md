![Banner](./banner/dweegle.png)

A functional-first JAX/Flax library for training deep learning models with minimal friction.

## Core Features

### Dataset Pipeline
- **TFRecord I/O**: Write/read TFRecords with custom serialization and compression
- **JAX Iterators**: Zero-copy TFRecord → JAX array loading with batching and shuffling
- **Preprocessing**: Automatic cropping, standardization, field statistics computation
- **Augmentations**: Image augmentation pipeline with immutable configs (via Albumentations)

### Neural Networks
- **VAE Variants**: Categorical VAE, Compact VAE, Wavelet VAE for compression/generation
- **Vision Models**: U-Net (denoising), HRNet, Vision Transformer (ViT) with masking
- **Attention**: Multi-head attention blocks with layer scaling
- **Utilities**: Receptive field computation, patch embeddings, position encodings

### Training Infrastructure
- **Functional Loops**: Immutable training state, pure functions, composable pipelines
- **Checkpointing**: Automatic checkpoint save/load with metrics history
- **Mixed Precision**: FP16/BF16 support with automatic loss scaling
- **Callbacks**: Visualization callbacks for training progress monitoring
- **Inference**: Batch inference utilities with JIT compilation

### Visualization & Debugging
- **Dataset Inspection**: Visualize batches, preview augmentations, tensor statistics
- **Training Plots**: Image grids, reconstructions, comparison views
- **Headless Support**: Automatic backend detection for server environments

### Docker-First Development
- GPU support (CUDA 12 + JAX)
- Consistent environment with all dependencies
- Volume mounting for datasets
- 94% test coverage

## Future Roadmap

### Model Export & Deployment
- **JAX → TFJS/ONNX/TFLite → WebGPU**: Full conversion pipeline for web deployment
- **ONNX → Rust Burn → WebGPU**: Alternative Rust-based deployment path
- Automated conversion tests for all supported architectures

### Tooling & Integration
- **MCP Server**: ✅ Available - LLM integration for programmatic model training/inference (see `MCP_README.md`)
- **Training Dashboard**: Wandb integration + minimal built-in metrics visualization
- Improved VAE training schedules with KL annealing

## Quick Start

```bash
git clone git@github.com:mo-najeeb-akbar/beagle.git
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

## MCP Server

Beagle includes an MCP (Model Context Protocol) server for LLM integration. This allows AI assistants like Claude to help you:

- Create and manage TFRecord datasets
- Configure and train neural networks
- Generate complete training scripts
- Debug and visualize training results

See `MCP_README.md` for setup and usage instructions.

```bash
# Install MCP dependencies
pip install -r requirements-mcp.txt

# Configure in Claude Desktop
# Add to claude_desktop_config.json:
{
  "mcpServers": {
    "beagle": {
      "command": "python",
      "args": ["/path/to/beagle/mcp_server_main.py"]
    }
  }
}
```

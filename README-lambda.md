# Beagle Installation for Lambda Labs

This is a minimal installation guide for Lambda Labs cloud instances, which come pre-configured with CUDA-enabled ML frameworks.

## Lambda Labs Pre-installed Packages

Your Lambda Labs instance already has:
- `python3-torch-cuda 2.7.0`
- `python3-tensorflow-cuda 2.19.0`
- `python3-jax-cuda 0.6.0`
- `python3-keras 3.10.0`
- `nvidia-cuda-toolkit 12.8.93`
- NumPy (via the above packages)

## Minimal Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd beagle

# Upgrade pip (recommended)
pip install --upgrade pip

# Install with minimal dependencies (just flax + typing_extensions)
pip install -e .

# Or install from PyPI (when published)
pip install beagle
```

This installs only:
- `numpy>=1.24.0,<2.0` (pinned to 1.x for Lambda Labs compatibility)
- `flax>=0.8.0` (JAX neural network library)
- `typing_extensions>=4.0.0` (lightweight type hints)
- `tqdm>=4.60.0` (progress bars)

Total install size: **~10-15 MB** (vs ~2 GB with full TensorFlow/JAX)

**Important:** NumPy is pinned to 1.x because Lambda Labs' pre-compiled TensorFlow and JAX require it.

## Optional Features

Install only what you need:

```bash
# For image augmentations
pip install beagle[augmentations]

# For visualization/plotting
pip install beagle[visualization]

# For everything
pip install beagle[all]
```

## Verification

```python
import jax
import tensorflow as tf
import beagle

print(f"JAX devices: {jax.devices()}")  # Should show GPU
print(f"TF version: {tf.__version__}")
print(f"Beagle version: {beagle.__version__}")

# Test basic functionality
from beagle.training import TrainState
print("✓ Beagle core working!")
```

## Build and Install from Source

If you want to use the Lambda-optimized configuration:

```bash
# Use the Lambda Labs config
cp pyproject-lambda.toml pyproject.toml

# Build wheel
pip install build
python -m build

# Install
pip install dist/beagle-0.1.0-py3-none-any.whl
```

## What's Different from Full Install?

The minimal Lambda Labs install excludes:
- TensorFlow (use system version)
- JAX (use system version)
- NumPy (use system version)
- Pillow (not used by beagle)
- PyWavelets (not used by beagle)
- Rich (not used by beagle)
- Protobuf (comes with TensorFlow)

You get full beagle functionality with 99% less bloat!

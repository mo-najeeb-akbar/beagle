#!/bin/bash
# Minimal installation script for Lambda Labs instances

set -e

echo "========================================="
echo "Beagle Minimal Install for Lambda Labs"
echo "========================================="
echo ""

# Backup original pyproject.toml if it exists
if [ -f "pyproject.toml" ]; then
    echo "Backing up original pyproject.toml..."
    cp pyproject.toml pyproject-original.toml
fi

# Use Lambda Labs optimized config
echo "Using minimal Lambda Labs configuration..."
cp pyproject-lambda.toml pyproject.toml

# Upgrade pip first (optional but recommended)
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install in editable mode
echo "Installing beagle with minimal dependencies..."
pip install -e .

echo ""
echo "✓ Installation complete!"
echo ""
echo "Installed dependencies:"
echo "  - numpy<2.0 (pinned for Lambda Labs compatibility)"
echo "  - flax (JAX neural network library)"
echo "  - typing_extensions (type hints)"
echo "  - tqdm (progress bars)"
echo ""
echo "Using system packages:"
echo "  - tensorflow-cuda (system: 2.19.0)"
echo "  - jax-cuda (system: 0.6.0)"
echo "  - torch-cuda (system: 2.7.0)"
echo ""
echo "Optional extras available:"
echo "  pip install -e .[augmentations]  # Image augmentations"
echo "  pip install -e .[visualization]  # Matplotlib plotting"
echo "  pip install -e .[all]            # All extras"
echo ""
echo "Test your installation:"
echo "  python -c 'import beagle; print(beagle.__version__)'"
echo ""

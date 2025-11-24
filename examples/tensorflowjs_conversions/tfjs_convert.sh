#!/bin/bash
set -e

IMAGE_NAME="tfjs-converter"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/model.keras"
    echo ""
    echo "Example:"
    echo "  $0 ~/models/encoder.keras"
    echo "  # Creates ~/models/encoder_js/"
    exit 1
fi

MODEL_PATH="$1"

# Convert to absolute path
MODEL_PATH=$(realpath "$MODEL_PATH")

# Validate input exists
if [ ! -e "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Get parent directory to mount
PARENT_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_PATH")

echo "Building Docker image (if needed)..."
docker build -t "$IMAGE_NAME" "$(dirname "$0")" 2>&1 | grep -E "(Successfully|CACHED|Step)" || true

echo ""
echo "Converting model..."
echo "  Input:  $MODEL_PATH"
echo "  Output: ${MODEL_PATH%.keras}_js/"
echo "  Output: ${MODEL_PATH%.h5}_js/"
echo ""

# Run conversion
docker run --rm \
    -v "$PARENT_DIR:/data" \
    "$IMAGE_NAME" \
    "/data/$MODEL_NAME"


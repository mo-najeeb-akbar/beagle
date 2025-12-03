#!/bin/bash
set -e

IMAGE_NAME="mrsid-converter"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <input.sid> [output.tif]"
    echo ""
    echo "Examples:"
    echo "  $0 myfile.sid output.tif"
    echo "  $0 myfile.sid output.jpg"
    echo "  $0 myfile.sid  # outputs to myfile.tif"
    exit 1
fi

INPUT_FILE="$1"

# Convert to absolute path
INPUT_FILE=$(realpath "$INPUT_FILE")

# Validate input exists
if [ ! -e "$INPUT_FILE" ]; then
    echo "Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Get parent directory to mount
PARENT_DIR=$(dirname "$INPUT_FILE")
INPUT_NAME=$(basename "$INPUT_FILE")

# Determine output file
if [ "$#" -eq 2 ]; then
    OUTPUT_FILE="$2"
else
    # Default output: same name but .tif extension
    OUTPUT_FILE="${INPUT_NAME%.*}.tif"
fi

OUTPUT_NAME=$(basename "$OUTPUT_FILE")

echo "Building Docker image (if needed)..."
docker build -t "$IMAGE_NAME" "$(dirname "$0")" 2>&1 | grep -E "(Successfully|CACHED|Step)" || true

echo ""
echo "Converting MrSID file..."
echo "  Input:  $INPUT_FILE"
echo "  Output: $PARENT_DIR/$OUTPUT_NAME"
echo ""

# Run the conversion
docker run --rm \
    -v "$PARENT_DIR:/data" \
    "$IMAGE_NAME" \
    mrsiddecode -i "/data/$INPUT_NAME" -o "/data/$OUTPUT_NAME"

echo ""
echo "Conversion complete!"

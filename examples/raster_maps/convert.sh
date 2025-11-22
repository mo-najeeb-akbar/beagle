#!/bin/bash

# MrSID Docker Conversion Script
# Usage: ./convert.sh input.sid output.tif

set -e

IMAGE_NAME="mrsid-converter"

# Build the Docker image if it doesn't exist
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input.sid> [output.tif]"
    echo ""
    echo "Examples:"
    echo "  $0 myfile.sid output.tif"
    echo "  $0 myfile.sid output.jpg"
    echo "  $0 myfile.sid  # outputs to myfile.tif"
    exit 1
fi

INPUT_FILE=$1
INPUT_DIR=$(dirname "$(realpath "$INPUT_FILE")")
INPUT_NAME=$(basename "$INPUT_FILE")

# Determine output file
if [ $# -eq 2 ]; then
    OUTPUT_FILE=$2
else
    # Default output: same name but .tif extension
    OUTPUT_FILE="${INPUT_NAME%.*}.tif"
fi

OUTPUT_NAME=$(basename "$OUTPUT_FILE")

echo "Converting: $INPUT_NAME -> $OUTPUT_NAME"
echo "Input directory: $INPUT_DIR"

# Run the conversion
docker run --rm \
    -v "$INPUT_DIR:/data" \
    $IMAGE_NAME \
    mrsiddecode -i "/data/$INPUT_NAME" -o "/data/$OUTPUT_NAME"

echo ""
echo "Conversion complete! Output saved to: $INPUT_DIR/$OUTPUT_NAME"

# TensorFlow.js Model Converter

Simple Docker-based tool for converting TensorFlow SavedModel or Keras models to TensorFlow.js format.

## Usage

Just call the script with your model path:

```bash
./tfjs_convert.sh /path/to/your/model
```

That's it! The script will:
- Build the Docker image (if needed)
- Mount the parent directory automatically
- Auto-detect SavedModel (directory) vs Keras (file) format
- Convert the model
- Create output with `_js` suffix

## Examples

```bash
# Convert a SavedModel directory (most common)
./tfjs_convert.sh ~/models/wavelet_vae_tf
# Output: ~/models/wavelet_vae_tf_js/

# Convert a .keras file
./tfjs_convert.sh ~/models/encoder.keras
# Output: ~/models/encoder.keras_js/

# Convert a .h5 file
./tfjs_convert.sh /data/vae.h5
# Output: /data/vae.h5_js/

# Make script executable first (one time)
chmod +x tfjs_convert.sh
```

## Output

The converter automatically:
- Detects input format (SavedModel directory vs Keras file)
- Adds `_js` suffix to the input name
- Creates output directory with `model.json` and weight files

## Supported Input Formats

- TensorFlow SavedModel directories (auto-detected)
  - Converted to `tfjs_graph_model` format (optimized for WebGPU)
  - Uses `serving_default` signature and `serve` tags
- Keras models (`.keras`, `.h5` files - auto-detected)
  - Converted to `tfjs_layers_model` format

## Notes

- First run will build the Docker image (takes a minute)
- Subsequent runs are fast
- SavedModel conversions use graph model format for WebGPU compatibility
- Use the output in browser/Node.js with `@tensorflow/tfjs`
- Models are converted using the `tensorflowjs_converter` CLI tool


FROM python:3.11-slim

# Install Python packages
RUN pip install --no-cache-dir tensorflow tensorflowjs

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Entrypoint script for conversion
ENTRYPOINT ["/bin/bash", "-c", "\
    set -e && \
    INPUT_PATH=\"$1\" && \
    if [ ! -e \"$INPUT_PATH\" ]; then \
        echo \"Error: Input path does not exist: $INPUT_PATH\" && exit 1; \
    fi && \
    BASENAME=$(basename \"$INPUT_PATH\") && \
    DIRNAME=$(dirname \"$INPUT_PATH\") && \
    OUTPUT_PATH=\"${DIRNAME}/${BASENAME}_js\" && \
    echo \"Converting: $INPUT_PATH\" && \
    echo \"Output to: $OUTPUT_PATH\" && \
    echo \"\" && \
    if [ -d \"$INPUT_PATH\" ]; then \
        echo \"Detected SavedModel format (directory)\" && \
        echo \"Using tfjs_graph_model format for WebGPU compatibility\" && \
        tensorflowjs_converter \
            --input_format=tf_saved_model \
            --output_format=tfjs_graph_model \
            --signature_name=serving_default \
            --saved_model_tags=serve \
            \"$INPUT_PATH\" \
            \"$OUTPUT_PATH\"; \
    else \
        echo \"Detected Keras format (file)\" && \
        tensorflowjs_converter --input_format=keras \"$INPUT_PATH\" \"$OUTPUT_PATH\"; \
    fi && \
    echo \"\" && \
    echo \"Conversion complete!\" && \
    echo \"Output saved to: $OUTPUT_PATH\" \
", "bash"]

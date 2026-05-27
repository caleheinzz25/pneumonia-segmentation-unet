#!/bin/bash
# Inference script

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/predict.sh <path_to_image_or_directory>"
    exit 1
fi

echo "Running prediction on $1..."
python -m src.predict --config config.yaml --input "$1"
echo "Prediction complete!"

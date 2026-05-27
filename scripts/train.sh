#!/bin/bash
# Training script for Pneumonia Segmentation

set -e

echo "Starting training..."
python -m src.train --config config.yaml
echo "Training complete!"

#!/bin/bash
# Evaluation script

set -e

echo "Running evaluation..."
python -m src.evaluate --config config.yaml
echo "Evaluation complete!"

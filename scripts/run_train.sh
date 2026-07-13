#!/bin/bash
# Prevent CUDA memory fragmentation → avoid spurious OOM on RTX 3070 (7.6 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kill any zombie GPU processes before starting
pkill -f "python.*src.train" 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="out/training_${TIMESTAMP}.log"
mkdir -p out
uv run python -m src.train --config config.yaml 2>&1 | tee "$LOG"

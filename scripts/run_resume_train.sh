#!/bin/bash
# Resume training from latest checkpoint
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kill zombie GPU processes
pkill -f "python.*src.train" 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="out/training_${TIMESTAMP}.log"
mkdir -p out
echo "[TrainMaster] Resuming training from epoch 2..."
echo "[TrainMaster] Log: $LOG"
uv run python -m src.train --config config.yaml --resume 2>&1 | tee "$LOG"
echo "[TrainMaster] Training finished."

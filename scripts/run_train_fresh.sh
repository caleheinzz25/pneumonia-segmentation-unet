#!/bin/bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pkill -f "python.*src.train" 2>/dev/null || true
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="out/training_${TIMESTAMP}.log"
mkdir -p out
echo "[Train] Fresh start with dice_bce_iou (bce_weight=0.5)"
echo "[Train] Log: $LOG"
uv run python -m src.train --config config.yaml 2>&1 | tee "$LOG"
echo "[Train] Finished."

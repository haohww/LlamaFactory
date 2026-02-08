#!/bin/bash
# SFT training for Qwen3-VL-30B-A3B on failure_v1 dataset using all 8 GPUs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=8

# Ensure conda env's newer libstdc++ is found before the system one,
# and include CUDA runtime libs.
export LD_LIBRARY_PATH=/opt/conda/envs/llamafactory/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Use cached model files only — avoids HF Hub timeout when 8 ranks hit the API simultaneously
export HF_HUB_OFFLINE=1

LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen3vl_30b_a3b_sft_failurev1_$(date +%Y%m%d_%H%M%S).log"

echo "Training started at $(date)"
echo "Log file: $LOG_FILE"

nohup llamafactory-cli train examples/train_full/qwen3vl_30b_a3b_sft_failurev1.yaml \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Monitor with: tail -f $LOG_FILE"

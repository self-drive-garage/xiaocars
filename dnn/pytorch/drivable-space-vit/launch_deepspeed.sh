#!/bin/bash

# DeepSpeed training launch script for drivable space ViT
# This script launches distributed training using DeepSpeed on 16 GPUs

# Set script to exit on any error
set -e

# Configuration
# NUM_GPUS=16
NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_NAME=${CONFIG_NAME:-config_deepspeed}

# Environment setup
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# Don't set CUDA_VISIBLE_DEVICES here - let DeepSpeed handle GPU assignment
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# Logging
echo "=== DeepSpeed Training Launch ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "Config: $CONFIG_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================="

# Check if DeepSpeed is installed
if ! python -c "import deepspeed" 2>/dev/null; then
    echo "Error: DeepSpeed is not installed. Please install it first:"
    echo "pip install deepspeed"
    exit 1
fi

# Check CUDA availability
if ! python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null; then
    echo "Error: CUDA is not available or PyTorch CUDA support is not installed"
    exit 1
fi

export HYDRA_FULL_ERROR=0
# Launch training with DeepSpeed - explicitly specify which GPUs to use
deepspeed --include localhost:0,1,2,3 \
    --master_port=$MASTER_PORT \
    ds_train.py \
    --config-path=config \
    --config-name=$CONFIG_NAME \
    hydra.job.chdir=false

echo "Training completed successfully!" 
#!/bin/bash

# DeepSpeed training launch script for drivable space ViT
# This script launches distributed training using DeepSpeed on 16 GPUs

# Set script to exit on any error
set -e

# Configuration
NUM_GPUS=16
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_NAME=${CONFIG_NAME:-config_deepspeed}

# Create output directories
OUTPUT_DIR="outputs/deepspeed"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Environment setup
# Don't set CUDA_VISIBLE_DEVICES here - let DeepSpeed handle GPU assignment
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
echo "=== DeepSpeed Training Launch ===" | tee "${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Number of GPUs: $NUM_GPUS" | tee -a "${LOG_FILE}"
echo "Master Port: $MASTER_PORT" | tee -a "${LOG_FILE}"
echo "Config: $CONFIG_NAME" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "=================================" | tee -a "${LOG_FILE}"

# Check if DeepSpeed is installed
if ! python -c "import deepspeed" 2>/dev/null; then
    echo "Error: DeepSpeed is not installed. Please install it first:" | tee -a "${LOG_FILE}"
    echo "pip install deepspeed" | tee -a "${LOG_FILE}"
    exit 1
fi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>&1 | tee -a "${LOG_FILE}"

export HYDRA_FULL_ERROR=0

# Function to run training
run_training() {
    # Launch training with DeepSpeed - use all 16 GPUs
    deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --master_port=$MASTER_PORT \
        ds_train.py \
        --config-path=config \
        --config-name=$CONFIG_NAME \
        hydra.job.chdir=false \
        training.resume="${OUTPUT_DIR}/checkpoint_epoch_*" \
        2>&1 | tee -a "${LOG_FILE}"
}

# Check if we should run in background
if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    echo "Starting training in background..." | tee -a "${LOG_FILE}"
    echo "Logs will be written to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "To monitor progress, use: tail -f ${LOG_FILE}" | tee -a "${LOG_FILE}"
    
    # Run training in background using nohup
    nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --master_port=$MASTER_PORT \
        ds_train.py \
        --config-path=config \
        --config-name=$CONFIG_NAME \
        hydra.job.chdir=false \
        training.resume="${OUTPUT_DIR}/checkpoint_epoch_*" \
        >> "${LOG_FILE}" 2>&1 &
    
    # Get the PID
    PID=$!
    echo "Training started with PID: $PID" | tee -a "${LOG_FILE}"
    echo "To stop training: kill $PID" | tee -a "${LOG_FILE}"
    
    # Save PID to file for easy reference
    echo $PID > "${OUTPUT_DIR}/training.pid"
    echo "PID saved to: ${OUTPUT_DIR}/training.pid" | tee -a "${LOG_FILE}"
    
    # Wait a few seconds to check if process is still running
    sleep 5
    if ps -p $PID > /dev/null; then
        echo "Training is running successfully in background!" | tee -a "${LOG_FILE}"
    else
        echo "Warning: Training process may have exited. Check logs for errors." | tee -a "${LOG_FILE}"
    fi
else
    # Run in foreground
    echo "Starting training in foreground..." | tee -a "${LOG_FILE}"
    echo "Use './launch_deepspeed.sh --background' to run in background" | tee -a "${LOG_FILE}"
    run_training
    echo "Training completed!" | tee -a "${LOG_FILE}"
fi 
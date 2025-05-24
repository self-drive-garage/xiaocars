#!/bin/bash

# Launch distributed training on 8 GPUs with nohup
# Usage: ./launch_dist_train.sh [additional hydra args]

# Number of GPUs to use
NUM_GPUS=8

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Set PyTorch DataLoader memory settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Disable shared memory if needed (uses file system instead)
# export DATALOADER_USE_SHARED_MEMORY=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/dist_train_${TIMESTAMP}.log"

echo "Starting distributed training on $NUM_GPUS GPUs..."
echo "Additional arguments: $@"
echo "Log file: $LOG_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo ""

# Launch distributed training using torchrun with nohup
# torchrun handles the distributed setup automatically
nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    dist_train.py \
    $@ \
    > "$LOG_FILE" 2>&1 &

# Get the PID of the background process
PID=$!
echo "Training started with PID: $PID"
echo "PID stored in: logs/dist_train_${TIMESTAMP}.pid"
echo $PID > "logs/dist_train_${TIMESTAMP}.pid"

echo ""
echo "Training is running in the background!"
echo "Commands to manage the training:"
echo "  - Monitor progress:  tail -f $LOG_FILE"
echo "  - Check if running:  ps -p $PID"
echo "  - Stop training:     kill $PID"
echo "" 

#!/bin/bash

# Safer launch script for distributed training with reduced memory pressure
# Usage: ./launch_dist_train_safe.sh [additional hydra args]

# Number of GPUs to use
NUM_GPUS=8

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Set PyTorch memory settings to avoid shared memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optional: Completely disable shared memory (uses file system instead)
# This may reduce performance but avoids shared memory errors
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/dist_train_${TIMESTAMP}.log"

echo "Starting SAFE distributed training on $NUM_GPUS GPUs..."
echo "Using reduced workers to avoid shared memory issues"
echo "Additional arguments: $@"
echo "Log file: $LOG_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo ""

# Launch with reduced workers (2 workers per GPU = 16 total)
# You can adjust this by changing data.num_workers
nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    dist_train.py \
    data.num_workers=16 \
    $@ \
    > "$LOG_FILE" 2>&1 &

# Get the PID of the background process
PID=$!
echo "Training started with PID: $PID"
echo "PID stored in: logs/dist_train_${TIMESTAMP}.pid"
echo $PID > "logs/dist_train_${TIMESTAMP}.pid"

echo ""
echo "Training is running in the background with reduced memory pressure!"
echo "Commands to manage the training:"
echo "  - Monitor progress:  tail -f $LOG_FILE"
echo "  - Check if running:  ps -p $PID"
echo "  - Stop training:     kill $PID"
echo ""
echo "Note: Using 2 workers per GPU (16 total) to avoid shared memory issues"
echo "You can adjust this with data.num_workers parameter" 
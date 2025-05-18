#!/bin/bash

# Find CUDA installation
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/cuda" ]; then
    export CUDA_HOME=/usr/cuda
else
    # Try to detect from nvcc
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ $? -eq 0 ]; then
        export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
        echo "Auto-detected CUDA_HOME as $CUDA_HOME"
    else
        echo "ERROR: Could not find CUDA installation. Please set CUDA_HOME manually."
        exit 1
    fi
fi

echo "Using CUDA_HOME=$CUDA_HOME"

# Make sure CUDA libraries are in library path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run the training script with all arguments passed to this script
torchrun --nproc_per_node=16 train_dist.py --deepspeed "$@" 
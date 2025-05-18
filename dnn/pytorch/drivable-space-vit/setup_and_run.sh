#!/bin/bash

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo "CUDA development toolkit (nvcc) not found, attempting to install..."
    
    # Check the distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        echo "Cannot determine OS distribution. Please install CUDA toolkit manually."
        exit 1
    fi
    
    # Install CUDA toolkit based on distribution
    if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
        echo "Detected $OS $VERSION"
        echo "Installing CUDA development toolkit..."
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-0 || sudo apt-get install -y cuda-toolkit
    elif [[ "$OS" == "centos" || "$OS" == "rhel" || "$OS" == "fedora" ]]; then
        echo "Detected $OS $VERSION"
        echo "Installing CUDA development toolkit..."
        sudo yum install -y cuda-toolkit-12-0 || sudo yum install -y cuda-toolkit
    else
        echo "Unsupported OS: $OS. Please install CUDA toolkit manually."
        echo "Continuing with CPU-only operations..."
    fi
    
    # Check if installation succeeded
    if ! command -v nvcc &> /dev/null
    then
        echo "Failed to install CUDA toolkit. Continuing with CPU-only operations..."
        # Set environment variables to use CPU operations
        export DS_BUILD_CPU_ADAM=1
        export DS_BUILD_FUSED_ADAM=0
        export DS_BUILD_TRANSFORMER=0
        export DS_BUILD_TRANSFORMER_INFERENCE=0
        export DS_BUILD_STOCHASTIC_TRANSFORMER=0
        export DS_BUILD_UTILS=0
        export DS_BUILD_SPARSE_ATTN=0
        export DS_BUILD_RAGGED_DEVICE_OPS=0
        export DS_BUILD_OPS=0
        export CUDA_HOME="NOT_NEEDED"
    else
        echo "CUDA toolkit installation successful!"
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        echo "Set CUDA_HOME=$CUDA_HOME"
    fi
else
    echo "CUDA toolkit found!"
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    echo "Set CUDA_HOME=$CUDA_HOME"
fi

# Run the training script with the specified number of GPUs
NUM_GPUS=${1:-16}  # Default to 16 GPUs if not specified
echo "Running training with $NUM_GPUS GPUs..."

# Execute with torchrun
torchrun --nproc_per_node=$NUM_GPUS train_dist.py --deepspeed --pipeline_parallel_size=4 --tensor_parallel_size=4 --data_dir=datasets/xiaocars --output_dir=outputs/deepspeed

echo "Training complete!" 
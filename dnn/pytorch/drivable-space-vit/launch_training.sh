#!/bin/bash

# Create a timestamped log file
LOG_DIR="logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export HYDRA_FULL_ERROR=1

# Run training with nohup to keep it running after SSH session ends
echo "Starting training in background. Log will be written to: ${LOG_FILE}"
nohup torchrun --nproc_per_node=16 train.py > ${LOG_FILE} 2>&1 &

# Save the process ID to allow killing it later if needed
echo $! > ${LOG_DIR}/training_${TIMESTAMP}.pid
echo "Process ID: $! (saved to ${LOG_DIR}/training_${TIMESTAMP}.pid)"

# Print instructions for monitoring the training
echo ""
echo "To monitor training progress, use:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To stop training, use:"
echo "  kill \$(cat ${LOG_DIR}/training_${TIMESTAMP}.pid)"

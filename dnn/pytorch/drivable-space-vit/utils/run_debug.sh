#!/bin/bash

# Make sure script is executable
# chmod +x run_debug.sh

# Set environment variables
export HYDRA_FULL_ERROR=1

# Clean up any previous logs
echo "Cleaning up previous logs..."
rm -f debug.log rank*.log combined_log.log

# Run the distributed training with logging
echo "Starting distributed training with 3 GPUs..."
torchrun --nproc_per_node=3 train.py 2>&1 | tee -a combined_log.log

# Gather all logs
echo "Combining logs for analysis..."
cat *.log > all_logs.log

echo "Done! Check all_logs.log for complete logs or debug.log for diagnostic information." 
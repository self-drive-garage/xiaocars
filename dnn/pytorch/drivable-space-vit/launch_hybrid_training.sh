#!/bin/bash
# save as launch_training.sh

# Set CUDA visible devices to all available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Launch using DeepSpeed
deepspeed --num_gpus=16 train_dist.py \
     --dp_size=2 \
     --pp_size=8 \
     --tp_size=1 \
     --zero_stage=0 \
     --batch_size=6 \
     --output_dir=./output_parallel_16gpu
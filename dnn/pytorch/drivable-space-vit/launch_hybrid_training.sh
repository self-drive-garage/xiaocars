#!/bin/bash
# save as launch_training.sh

# Set CUDA visible devices to all available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Launch using DeepSpeed
deepspeed --num_gpus=16 train_dist.py \
    --dp_size=4 \
    --pp_size=2 \
    --tp_size=2 \
    --zero_stage=2 \
    --embed_dim=1024 \
    --num_heads=16 \
    --num_layers=24 \
    --img_size=256 \
    --batch_size=4 \
    --gradient_accumulation=4 \
    --mixed_precision
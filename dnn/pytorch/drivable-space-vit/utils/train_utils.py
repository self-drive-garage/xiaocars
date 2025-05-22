import yaml
import glob
from pathlib import Path
import random
import numpy as np
import torch
import json
import logging

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory"""
    checkpoint_files = glob.glob(str(checkpoint_dir / 'checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers from filenames
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    if not epochs:
        return None
    
    # Find the checkpoint with the highest epoch number
    latest_epoch = max(epochs)
    latest_checkpoint = checkpoint_dir / f'checkpoint_epoch_{latest_epoch}.pth'
    
    return str(latest_checkpoint)

def seed_everything(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_deepspeed_config(args, config, dataset_size):
    """Generate DeepSpeed configuration."""
    # Convert from DictConfig to plain dict if needed
    if not isinstance(config, dict):
        config = {k: v for k, v in config.items()}
    
    # Get batch size from config
    batch_size = config['dataset']['batch_size']
    
    # Calculate steps
    steps_per_epoch = dataset_size // (batch_size * args.dp_size)
    if config['training'].get('max_steps') is None:
        max_steps = steps_per_epoch * config['training']['epochs']
    else:
        max_steps = config['training']['max_steps']
    
    # Get warmup and LR
    warmup_steps = int(0.1 * max_steps)
    if config['training'].get('base_lr') is None:
        base_lr = 5e-5
    else:
        base_lr = config['training']['base_lr']
    
    # Configure optimizer parameters
    optimizer_params = {
        "type": "AdamW",
        "params": {
            "lr": base_lr,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    }
    
    # Configure scheduler
    scheduler_params = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": base_lr,
            "warmup_num_steps": warmup_steps,
            "total_num_steps": max_steps,
            "warmup_type": "linear"
        }
    }
    
    # Configure Zero optimization
    zero_params = {
        "stage": args.zero_stage,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8,
        "offload_optimizer": {
            "device": "cpu" if args.zero_stage >= 2 else "none",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu" if args.zero_stage >= 3 else "none",
            "pin_memory": True
        }
    }
    
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "optimizer": optimizer_params,
        "scheduler": scheduler_params,
        "zero_optimization": zero_params,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "communication_data_type": "fp16",
        "pipeline": {
            "activation_checkpoint_interval": 0 if args.pp_size <= 1 else 1,
            "tensorboard": {
                "enabled": True,
                "output_path": "logs/",
                "job_name": "deepspeed_job"
            }
        },
        # Add parameter to help with P2P tensor handling
        "comms_logger": {
            "enabled": True,
            "verbose": False,
            "prof_all": False,
            "debug": False
        },
        "copy_non_contiguous_tensors": True
    }
    
    return ds_config


def debug_attention_forward(attn_module, x, attn_mask=None, need_weights=False):
    """
    Debug wrapper for MultiheadAttention forward pass that collects detailed diagnostic information
    in case of errors, useful for understanding issues with attention operations during training.
    
    Args:
        attn_module: The MultiheadAttention module
        x: Input tensor to attention module
        attn_mask: Optional attention mask
        need_weights: Whether to return attention weights
        
    Returns:
        The result of the attention operation
        
    Raises:
        ValueError with detailed diagnostic information if an error occurs
    """
    # Collect debug information
    debug_info = []
    debug_info.append(f"Input x shape: {x.shape}")
    
    # Collect attention module weights information
    try:
        # Check if we can access the attention module weights
        debug_info.append(f"Self-attention in_proj_weight shape: {attn_module.in_proj_weight.shape if hasattr(attn_module, 'in_proj_weight') else 'None'}")
        debug_info.append(f"Self-attention out_proj.weight shape: {attn_module.out_proj.weight.shape}")
        debug_info.append(f"Self-attention out_proj.bias shape: {attn_module.out_proj.bias.shape if attn_module.out_proj.bias is not None else 'None'}")
        debug_info.append(f"Self-attention weight dtype: {attn_module.out_proj.weight.dtype}")
    except Exception as e:
        debug_info.append(f"Error accessing attention weights: {e}")
    
    try:
        # Perform the actual attention operation
        return attn_module(x, x, x, attn_mask=attn_mask, need_weights=need_weights)
    except Exception as e:
        # Add error information
        debug_info.append(f"Error in attention forward pass: {e}")
        debug_info.append(f"q, k, v shapes: {x.shape}")
        if hasattr(attn_module, 'out_proj'):
            debug_info.append(f"out_proj.weight: {attn_module.out_proj.weight.shape}, {attn_module.out_proj.weight.dtype}")
        
        # Raise all collected debug information as a single ValueError
        try:
            raise ValueError("Attention module diagnostic information:\n" + "\n".join(debug_info)) from e
        except ValueError as e:
            logger.error(str(e))


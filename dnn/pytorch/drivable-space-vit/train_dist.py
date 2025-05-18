import os
import sys
import time
import argparse
import logging
import numpy as np
import random
from pathlib import Path
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
import glob
import json
import subprocess

# Set CUDA_HOME properly before importing DeepSpeed
cuda_home = "/usr/local/cuda-12.9"  # Use your actual CUDA path
os.environ["CUDA_HOME"] = cuda_home
os.environ["CUDA_PATH"] = cuda_home

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Import DeepSpeed for model parallelism
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.utils import RepeatingLoader

from model.model import (
    create_model, 
    create_loss_function, 
    create_optimizer, 
    create_scheduler,
    save_model_checkpoint, 
    load_model_from_checkpoint
)
from model.driving_dataset import DrivingDataset, create_dataloader
from visualize import visualize_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Basic NCCL configuration
os.environ["NCCL_DEBUG"] = "INFO"  # Get more diagnostic information
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Use blocking sync to prevent CUDA errors

# Force TCP communication (more reliable than InfiniBand for testing)
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Specify network interface (adjust for your system)

# Extend timeouts for large models
os.environ["NCCL_TIMEOUT"] = "3600"  # 60 minute timeout (in seconds)

# Manage memory more carefully
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match device IDs to PCI bus order

# For maximum performance (faster but may have slight variations)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # Enables auto-tuner


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


def update_config_with_args(config, args):
    """Update config with command line arguments"""
    # Update model config
    if args.img_size is not None:
        config['model']['img_size'] = args.img_size
    else:
        # Reduce image size to save memory
        config['model'].setdefault('img_size', 160)  # Further reduced from 192 to 160
    if args.patch_size is not None:
        config['model']['patch_size'] = args.patch_size
    if args.embed_dim is not None:
        config['model']['embed_dim'] = args.embed_dim
    else:
        # Use smaller embedding dimension to save memory
        config['model'].setdefault('embed_dim', 512)  # Reduced from default 768
    if args.num_layers is not None:
        config['model']['num_layers'] = args.num_layers
    if args.num_heads is not None:
        config['model']['num_heads'] = args.num_heads
    if args.mlp_ratio is not None:
        config['model']['mlp_ratio'] = args.mlp_ratio
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
        config['model']['attn_dropout'] = args.dropout
    
    # Update dataset config
    if args.seq_len is not None:
        config['dataset']['seq_len'] = args.seq_len
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    else:
        # Calculate batch size based on GPU count
        gpu_count = torch.cuda.device_count()
        per_gpu_batch = max(1, min(2, int(32 / gpu_count)))  # Limit to at most 2 per GPU, at least 1
        config['dataset'].setdefault('batch_size', per_gpu_batch)
        logger.info(f"Auto batch size: {per_gpu_batch} per GPU, {per_gpu_batch * gpu_count} total across {gpu_count} GPUs")
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    else:
        # Auto-set num_workers to 2 per GPU (reduced from 4 to be safer)
        gpu_count = torch.cuda.device_count()
        config['dataset'].setdefault('num_workers', min(2, os.cpu_count() // gpu_count))
    
    # Update training config
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if args.warmup_epochs is not None:
        config['training']['warmup_epochs'] = args.warmup_epochs
    if args.min_lr is not None:
        config['training']['min_lr'] = args.min_lr
    if args.gradient_accumulation is not None:
        config['training']['gradient_accumulation'] = args.gradient_accumulation
    if args.mixed_precision is not None:
        config['training']['mixed_precision'] = args.mixed_precision
    if args.reconstruction_weight is not None:
        config['training']['reconstruction_weight'] = args.reconstruction_weight
    if args.consistency_weight is not None:
        config['training']['consistency_weight'] = args.consistency_weight
    if args.future_weight is not None:
        config['training']['future_weight'] = args.future_weight
    
    # Update logging config
    if args.log_interval is not None:
        config['logging']['log_interval'] = args.log_interval
    if args.save_interval is not None:
        config['logging']['save_interval'] = args.save_interval
    if args.eval_interval is not None:
        config['logging']['eval_interval'] = args.eval_interval
    if args.visualize_every is not None:
        config['logging']['visualize_every'] = args.visualize_every
    if args.num_viz_samples is not None:
        config['logging']['num_viz_samples'] = args.num_viz_samples
    
    return config


def seed_everything(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment(args):
    # Set random seed
    seed_everything(args.seed)
    return args


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, config, scaler=None, rank=0):
    """Train model for one epoch with DeepSpeed"""
    # Synchronize all processes before starting training
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            print(f"Rank {rank}: Synchronized before epoch {epoch+1}")
        except Exception as e:
            print(f"Rank {rank}: Barrier synchronization failed: {e}")

    model.train()
    running_loss = 0.0
    total_steps = len(loader)
    start_time = time.time()
    
    # Use tqdm for progress bar (only on main process)
    if rank == 0:
        pbar = tqdm(enumerate(loader), total=total_steps, desc=f"Epoch {epoch+1}")
    else:
        pbar = enumerate(loader)
    
    for step, batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with DeepSpeed handling
        outputs = model(batch, task='all')
        loss, loss_dict = loss_fn(outputs, batch)
        
        # Backward pass with DeepSpeed handling
        model.backward(loss)
        model.step()
        
        # Update metrics
        running_loss += loss.item()
        
        # Update progress bar (only on main process)
        if rank == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': model.get_lr()[0] if hasattr(model, 'get_lr') else 0.0
            })
            
            # Log progress at intervals
            if step % config['logging']['log_interval'] == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch: [{epoch+1}/{config['training']['epochs']}][{step}/{total_steps}] "
                    f"Loss: {loss.item():.4f} "
                    f"Time: {elapsed:.2f}s "
                    f"LR: {model.get_lr()[0] if hasattr(model, 'get_lr') else 0.0:.6f}"
                )
                
                # Log individual loss components
                for loss_name, loss_value in loss_dict.items():
                    logger.info(f"  {loss_name}: {loss_value:.4f}")
    
    # Return average loss
    return running_loss / total_steps


def parse_args():
    parser = argparse.ArgumentParser(description='Train Drivable Space ViT model using hybrid parallelism')
    
    # DeepSpeed parallel strategies
    parser.add_argument('--dp_size', type=int, default=4,
                        help='Data Parallelism size (number of data-parallel replicas)')
    parser.add_argument('--pp_size', type=int, default=2,
                        help='Pipeline Parallelism size (number of pipeline stages)')
    parser.add_argument('--tp_size', type=int, default=2,
                        help='Tensor Parallelism size (number of tensor-parallel slices)')
    parser.add_argument('--zero_stage', type=int, default=2,
                        help='ZeRO optimization stage (0, 1, 2, or 3)')
    
    # Config file argument
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='datasets/xiaocars',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to save checkpoints and logs')
    
    # Model arguments (can override config)
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for ViT')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=None,
                        help='MLP expansion ratio')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout probability')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length')
    
    # Training arguments (can override config)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU for training')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='Minimum learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=None,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', default=None,
                        help='Use mixed precision training')
    
    # Loss weights (can override config)
    parser.add_argument('--reconstruction_weight', type=float, default=None,
                        help='Weight for reconstruction loss')
    parser.add_argument('--consistency_weight', type=float, default=None,
                        help='Weight for consistency loss')
    parser.add_argument('--future_weight', type=float, default=None,
                        help='Weight for future prediction loss')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging and saving (can override config)
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Logging interval in steps')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Checkpoint saving interval in epochs')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Evaluation interval in epochs')
    
    # Visualization (can override config)
    parser.add_argument('--visualize_every', type=int, default=None,
                        help='Visualize predictions every N epochs')
    parser.add_argument('--num_viz_samples', type=int, default=None,
                        help='Number of samples to visualize')
    
    # # DeepSpeed config
    # parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
    #                     help='DeepSpeed configuration file')
    
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def validate(model, loader, loss_fn, device, epoch, config, rank=0):
    """Validate the model using DeepSpeed"""
    model.eval()
    val_loss = 0.0
    total_steps = len(loader)
    
    # Use tqdm for progress bar (only on main process)
    if rank == 0:
        pbar = tqdm(loader, total=total_steps, desc=f"Validation {epoch+1}")
    else:
        pbar = loader
    
    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with task='all' to generate all outputs needed for loss calculation
            outputs = model(batch, task='all')
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Update metrics
            val_loss += loss.item()
            
            # Update progress bar (only on main process)
            if rank == 0:
                pbar.set_postfix({'loss': loss.item()})
    
    # Average validation loss
    val_loss = val_loss / total_steps
    
    # Log validation results (only on main process)
    if rank == 0:
        logger.info(f"Validation Epoch: [{epoch+1}/{config['training']['epochs']}] Loss: {val_loss:.4f}")
        
        # Log individual loss components
        for loss_name, loss_value in loss_dict.items():
            logger.info(f"  Validation {loss_name}: {loss_value:.4f}")
    
    return val_loss


def setup_distributed():
    """Initialize distributed training using environment variables set by torchrun"""
    # Get rank and world_size from environment variables
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Process {rank}/{world_size} (local_rank={local_rank}) starting")
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"Rank {rank}: Using CUDA device {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    else:
        device = torch.device('cpu')
        print(f"Rank {rank}: Using CPU")
    
    # Print key environment variables
    print(f"Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"Rank {rank}: MASTER_PORT={os.environ.get('MASTER_PORT', 'not set')}")
    
    # Initialize process group
    print(f"Rank {rank}: About to initialize process group")
    try:
        # Set timeout to prevent hanging
        timeout = timedelta(seconds=60)
        
        # Initialize the process group using environment variables set by torchrun
        # Use NCCL backend for best GPU performance
        torch.distributed.init_process_group(
            "nccl",
            timeout=timeout
        )
        print(f"Rank {rank}: Process group initialized successfully with NCCL")
    except Exception as e:
        print(f"Rank {rank}: Process group initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Rank {rank}: Process group state: is_initialized={dist.is_initialized()}")
    
    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        print("Cleaning up distributed training")
        dist.destroy_process_group()
        print("Distributed training cleanup complete")


def create_pipeline_model(model, loss_fn, num_stages):
    """Create a pipeline-parallel model from a base model"""
    from deepspeed.pipe import PipelineModule, LayerSpec
    
    # For our custom multi-view transformer model, we need to create a custom partitioning
    layers_spec = []
    
    # Stage 1: Input processing and embeddings
    # Includes patch embedding, positional embedding, etc.
    input_layer = nn.ModuleList([
        model.patch_embed,
        model.pos_drop
    ])
    layers_spec.append(LayerSpec(nn.Sequential, *input_layer))
    
    # Calculate layers per stage
    total_layers = (
        len(model.spatial_transformer_layers) + 
        len(model.cross_view_transformer_layers) +
        len(model.temporal_transformer_layers)
    )
    
    # Need at least 2 stages for pipeline (input and output)
    num_transformer_stages = max(1, num_stages - 2)
    
    # Distribute transformer layers across stages
    layers_per_stage = max(1, total_layers // num_transformer_stages)
    
    # Add spatial transformer layers
    for i in range(0, len(model.spatial_transformer_layers), layers_per_stage):
        end_idx = min(i + layers_per_stage, len(model.spatial_transformer_layers))
        stage_layers = model.spatial_transformer_layers[i:end_idx]
        if stage_layers:
            layers_spec.append(LayerSpec(nn.Sequential, *stage_layers))
    
    # Add cross-view transformer layers
    for i in range(0, len(model.cross_view_transformer_layers), layers_per_stage):
        end_idx = min(i + layers_per_stage, len(model.cross_view_transformer_layers))
        stage_layers = model.cross_view_transformer_layers[i:end_idx]
        if stage_layers:
            layers_spec.append(LayerSpec(nn.Sequential, *stage_layers))
    
    # Add temporal transformer layers
    for i in range(0, len(model.temporal_transformer_layers), layers_per_stage):
        end_idx = min(i + layers_per_stage, len(model.temporal_transformer_layers))
        stage_layers = model.temporal_transformer_layers[i:end_idx]
        if stage_layers:
            layers_spec.append(LayerSpec(nn.Sequential, *stage_layers))
    
    # Final stage: Output heads
    output_layer = nn.ModuleList([
        model.norm,
        model.drivable_space_decoder,
        model.image_reconstruction_decoder,
        model.future_prediction_head
    ])
    layers_spec.append(LayerSpec(nn.Sequential, *output_layer))
    
    # If we have too many stages, consolidate
    if len(layers_spec) < num_stages:
        logger.warning(f"Requested {num_stages} pipeline stages but model only has {len(layers_spec)} natural divisions")
        logger.warning(f"Will use {len(layers_spec)} pipeline stages instead")
    elif len(layers_spec) > num_stages:
        # Need to consolidate layers
        logger.warning(f"Model has {len(layers_spec)} layers but only {num_stages} stages requested")
        logger.warning("Consolidating layers to fit into requested number of stages")
        
        # Simple consolidation: merge layers until we have the right number
        while len(layers_spec) > num_stages:
            # Find the smallest layer
            smallest_idx = 0
            smallest_size = float('inf')
            
            for i, layer_spec in enumerate(layers_spec[:-1]):  # Don't merge the final layer
                # Estimate size by number of parameters
                layer_size = sum(p.numel() for p in layer_spec.kwargs.parameters())
                if layer_size < smallest_size:
                    smallest_size = layer_size
                    smallest_idx = i
            
            # Merge the smallest layer with the next one
            if smallest_idx < len(layers_spec) - 1:
                combined_modules = list(layers_spec[smallest_idx].kwargs) + list(layers_spec[smallest_idx + 1].kwargs)
                layers_spec[smallest_idx] = LayerSpec(nn.Sequential, *combined_modules)
                layers_spec.pop(smallest_idx + 1)
    
    # Create the pipeline model
    pipeline_model = PipelineModule(
        layers=layers_spec,
        loss_fn=loss_fn,
        num_stages=min(num_stages, len(layers_spec)),
        activation_checkpoint_interval=0  # Disable activation checkpointing for now
    )
    
    return pipeline_model


def generate_deepspeed_config(args, config, train_dataset_size):
    """Generate DeepSpeed configuration based on arguments and config"""
    
    # Calculate total elements across all parallelism dimensions
    total_process_count = args.dp_size * args.pp_size * args.tp_size
    
    # Verify the number matches available GPUs
    available_gpus = torch.cuda.device_count()
    if total_process_count > available_gpus:
        logger.warning(f"Requested {total_process_count} processes but only {available_gpus} GPUs available!")
        # Adjust parameters to fit within available GPUs
        args.dp_size = max(1, available_gpus // (args.pp_size * args.tp_size))
    
    # Calculate steps for scheduler
    steps_per_epoch = train_dataset_size // (config['dataset']['batch_size'] * args.dp_size)
    warmup_steps = config['training']['warmup_epochs'] * steps_per_epoch
    total_steps = config['training']['epochs'] * steps_per_epoch
    
    # Create the DeepSpeed config
    ds_config = {
        # Communication parameters
        "communication_data_type": "fp16",  # More efficient communication
        
        # Training batch size = micro_batch_size_per_gpu * gradient_accumulation_steps * dp_size
        "train_batch_size": config['dataset']['batch_size'] * args.dp_size * config['training'].get('gradient_accumulation', 1),
        "train_micro_batch_size_per_gpu": config['dataset']['batch_size'],
        "gradient_accumulation_steps": config['training'].get('gradient_accumulation', 1),
        "steps_per_print": config['logging'].get('log_interval', 10),
        
        # Optimizer settings
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config['training']['lr'],
                "betas": [
                    config['training'].get('beta1', 0.9),
                    config['training'].get('beta2', 0.999)
                ],
                "eps": config['training'].get('eps', 1e-8),
                "weight_decay": config['training']['weight_decay']
            }
        },
        
        # Learning rate schedule
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config['training']['lr'],
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps
            }
        },
        
        # Mixed precision settings
        "fp16": {
            "enabled": config['training'].get('mixed_precision', False),
            "auto_cast": True,
            "loss_scale": 0,  # Auto scaling
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
        },
        
        # Gradient clipping
        "gradient_clipping": config['training'].get('gradient_clipping', 1.0),
        
        # ZeRO optimization
        "zero_optimization": {
            "stage": args.zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            # Memory offloading (for larger models)
            "offload_optimizer": {
                "device": "cpu" if args.zero_stage >= 2 else "none",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu" if args.zero_stage >= 3 else "none",
                "pin_memory": True
            },
        },
        
        # Pipeline parallelism configuration
        "pipeline": {
            "enabled": args.pp_size > 1,
            "num_stages": args.pp_size,
            "pipe_chunk_size": 2,  # Smaller for better memory usage
            "activation_checkpoint_interval": 1,
            "checkpoint_stages": True,
        },
        
        # Activation checkpointing (crucial for large models)
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": min(4, args.pp_size),
            "synchronize_checkpoint_boundary": True,
            "profile": False
        },
        
        # Logging and monitoring
        "wall_clock_breakdown": False,
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 3
        }
    }
    
    # Save the configuration to a file
    with open(args.deepspeed_config, 'w') as f:
        json.dump(ds_config, f, indent=4)
    
    return ds_config


def main():
    # Add immediate logging to see if script starts
    print("Starting distributed training script...")
    
    # Parse arguments
    args = parse_args()
    
    # Calculate total processes needed based on parallelism parameters
    total_process_count = args.dp_size * args.pp_size * args.tp_size
    available_gpus = torch.cuda.device_count()
    
    if total_process_count > available_gpus:
        print(f"Warning: Requested {total_process_count} processes but only {available_gpus} GPUs available")
        print(f"Adjusting parallelism parameters to fit available hardware")
        # Adjust to make it fit
        args.dp_size = max(1, available_gpus // (args.pp_size * args.tp_size))
        total_process_count = args.dp_size * args.pp_size * args.tp_size
        print(f"Adjusted configuration: DP={args.dp_size}, PP={args.pp_size}, TP={args.tp_size}, Total={total_process_count}")
    
    # Log the parallelism strategy
    print(f"Hybrid Parallelism Strategy:")
    print(f"  Data Parallel (DP) size: {args.dp_size}")
    print(f"  Pipeline Parallel (PP) size: {args.pp_size}")
    print(f"  Tensor Parallel (TP) size: {args.tp_size}")
    print(f"  ZeRO stage: {args.zero_stage}")
    print(f"  Total GPUs used: {total_process_count} of {available_gpus} available")
    
    # Setup distributed environment with DeepSpeed
    # When using DeepSpeed, these will be set automatically by DeepSpeed
    deepspeed.init_distributed(dist_backend='nccl')
    
    # Get rank and local_rank from DeepSpeed/distributed setup
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Process {rank}/{world_size} (local_rank={local_rank}) ready with device {device}")
    
    # Using default config path
    config_path = "config"
    config_name = "config"
    
    # Initialize Hydra without using
    # Initialize Hydra without using the decorator
    with hydra.initialize(version_base=None, config_path=config_path):
        hydra_config = hydra.compose(config_name=config_name)
        config = OmegaConf.to_container(hydra_config, resolve=True)
    
    print(f"Configuration loaded for rank {rank}")
    
    # Get CLI arguments but don't use them to override Hydra config directly
    # Instead, parse them manually and update the config
    seed = args.seed
    resume = args.resume
    auto_resume = not args.resume # New parameter for automatic resuming from latest checkpoint
    data_dir = args.data_dir
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Update configuration with command-line parameters
    if epochs is not None:
        config['training']['epochs'] = epochs
    
    if batch_size is not None:
        config['dataset']['batch_size'] = batch_size
    else:
        # For model parallelism, we can use larger batches per GPU
        per_gpu_batch = max(2, min(4, int(64 / args.dp_size)))  # Increase based on model size
        config['dataset'].setdefault('batch_size', per_gpu_batch)
        if rank == 0:
            logger.info(f"Auto batch size: {per_gpu_batch} per GPU, {per_gpu_batch * args.dp_size} total across {args.dp_size} data parallel workers")
    
    # Set appropriate image size - for model parallelism, we can use larger images
    if args.img_size is not None:
        config['model']['img_size'] = args.img_size
    else:
        # With model parallelism, we can use larger images
        config['model']['img_size'] = min(256, 64 * args.pp_size)  # Scale with pipeline parallelism
        if rank == 0:
            logger.info(f"Setting larger image size for model parallelism: {config['model']['img_size']}")
    
    # Set appropriate embedding dimension - scale with tensor parallelism
    if args.embed_dim is not None:
        config['model']['embed_dim'] = args.embed_dim
    else:
        # With tensor parallelism, we can use larger embedding dimensions
        config['model']['embed_dim'] = min(1024, 128 * args.tp_size * 2)  # Scale with tensor parallelism
        if rank == 0:
            logger.info(f"Setting larger embedding dimension for tensor parallelism: {config['model']['embed_dim']}")
    
    # Ensure num_heads is divisible by tensor parallelism size and embed_dim
    if args.num_heads is not None:
        config['model']['num_heads'] = args.num_heads
    else:
        # Ensure num_heads is divisible by tp_size for tensor parallelism
        base_heads = config['model']['embed_dim'] // 64  # 64 dim per head is standard
        config['model']['num_heads'] = max(1, (base_heads // args.tp_size) * args.tp_size)
        if rank == 0:
            logger.info(f"Setting num_heads to {config['model']['num_heads']} for compatibility with TP={args.tp_size}")
    
    # Increase depth with pipeline parallelism
    if args.num_layers is not None:
        config['model']['num_layers'] = args.num_layers
    else:
        base_layers = 12
        config['model']['num_layers'] = min(36, base_layers * args.pp_size)
        if rank == 0:
            logger.info(f"Setting num_layers to {config['model']['num_layers']} for PP={args.pp_size}")
    
    # Auto-set num_workers to 2 per GPU (reduced from 4 to be safer)
    if config['dataset'].get('num_workers') is None:
        config['dataset']['num_workers'] = min(2, os.cpu_count() // world_size)
    
    print(f"Configuration updated for rank {rank}")
    
    # Create output directory
    output_path = Path(output_dir)
    if rank == 0:  # Only create directories on main process
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = output_path / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create visualization directory
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set up logging to file
        log_file = output_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Checkpoint directory for auto-resume
    checkpoint_dir = output_path / 'checkpoints'
    
    # Auto-resume from latest checkpoint if enabled and no specific checkpoint provided
    if auto_resume and not resume and rank == 0:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume = latest_checkpoint
            logger.info(f"Auto-resuming from latest checkpoint: {resume}")
    
    # Log hardware info
    if rank == 0:  # Only log on main process
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Using hybrid parallelism with:")
            logger.info(f"  - Data Parallel (DP) size: {args.dp_size}")
            logger.info(f"  - Pipeline Parallel (PP) size: {args.pp_size}")
            logger.info(f"  - Tensor Parallel (TP) size: {args.tp_size}")
            logger.info(f"  - ZeRO optimization stage: {args.zero_stage}")
    
    # Save merged config to output directory
    if rank == 0:  # Only save on main process
        with open(output_path / 'used_config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    # Setup tensorboard (only on main process)
    writer = SummaryWriter(log_dir=output_path / 'logs') if rank == 0 else None
    
    # Create datasets and dataloaders
    if rank == 0:
        logger.info("Creating datasets...")
    
    train_dataset = DrivingDataset(
        data_dir=data_dir,
        split='train',
        seq_len=config['dataset']['seq_len'],
        img_size=config['model']['img_size'],
        random_sequence=config['dataset']['random_sequence'],
        cache_images=config['dataset']['cache_images'],
        config=config,
    )
    
    val_dataset = DrivingDataset(
        data_dir=data_dir,
        split='val',
        seq_len=config['dataset']['seq_len'],
        img_size=config['model']['img_size'],
        random_sequence=False,  # Always deterministic for validation
        cache_images=config['dataset']['cache_images'],
        config=config,
    )
    
    if rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Generate DeepSpeed config
    if rank == 0:
        logger.info("Generating DeepSpeed configuration...")
    ds_config = generate_deepspeed_config(args, config, len(train_dataset))
    
    # Create samplers and dataloaders based on the parallelism method
    if args.pp_size > 1:
        # For pipeline parallelism, we need different dataloader strategies
        train_sampler = None  # DeepSpeed pipeline will handle this
        val_sampler = None
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            shuffle=True,  # For pipeline, we can use shuffle
            sampler=None,
            config=config,
            debug=False
        )
        
        # Wrap with RepeatingLoader for pipeline
        train_loader = RepeatingLoader(train_loader)
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            shuffle=False,
            sampler=None,
            config=config,
            debug=False
        )
    else:
        # Standard distributed training with samplers
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=args.dp_size,
            rank=rank % args.dp_size,
            shuffle=True,
            drop_last=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=args.dp_size,
            rank=rank % args.dp_size,
            shuffle=False,
            drop_last=True
        )
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            shuffle=False,  # Don't shuffle here, the sampler will do it
            sampler=train_sampler,
            config=config,
            debug=False
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            shuffle=False,
            sampler=val_sampler,
            config=config,
            debug=False
        )
    
    # Create model
    if rank == 0:
        logger.info("Creating model...")
    
    model_config = {
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'in_chans': config['model']['num_channels'],
        'embed_dim': config['model']['embed_dim'],
        'depth': config['model']['num_layers'],
        'num_heads': config['model']['num_heads'],
        'mlp_ratio': config['model']['mlp_ratio'],
        'dropout': config['model']['dropout'],
        'attn_dropout': config['model']['attn_dropout'],
        'ego_motion_dim': config['model']['ego_motion_dim'],
    }
    
    # Initialize model, optimizer, scheduler, and loss function
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Create base model
    model = create_model(**model_config, config=config)
    
    # Create loss function
    loss_fn = create_loss_function(
        reconstruction_weight=config['training']['reconstruction_weight'],
        consistency_weight=config['training']['consistency_weight'],
        future_weight=config['training']['future_weight'],
        config=config
    )
    
    # Create pipeline model if using pipeline parallelism
    if args.pp_size > 1:
        if rank == 0:
            logger.info(f"Creating pipeline-parallel model with {args.pp_size} stages")
        model = create_pipeline_model(model, loss_fn, args.pp_size)
    
    # Resume from checkpoint if specified
    if resume:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {resume}")
        
        # Initialize DeepSpeed with model and resume from checkpoint
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        
        # Load checkpoint
        tag = None  # Automatic tag
        if os.path.isdir(resume):
            # Directory path - DeepSpeed will look for checkpoint files inside
            load_path = resume
        else:
            # File path - extract directory path
            load_path = os.path.dirname(resume)
            tag = os.path.basename(resume).split('.')[-2]  # Remove .pt extension
        
        # Load DeepSpeed checkpoint
        _, client_state = model_engine.load_checkpoint(load_path, tag=tag)
        
        # Extract epoch info if available
        if client_state and "epoch" in client_state:
            start_epoch = client_state["epoch"] + 1
            best_val_loss = client_state.get("best_val_loss", float('inf'))
            if rank == 0:
                logger.info(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    else:
        # Initialize DeepSpeed with model (no resume)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
    
    # Print model summary (parameter count)
    if rank == 0:
        param_count = sum(p.numel() for p in model_engine.parameters())
        trainable_param_count = sum(p.numel() for p in model_engine.parameters() if p.requires_grad)
        logger.info(f"Model parameter count: {param_count/1e6:.2f}M parameters")
        logger.info(f"Trainable parameter count: {trainable_param_count/1e6:.2f}M parameters")
        logger.info(f"Starting training from epoch {start_epoch} to {config['training']['epochs']}")
    
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            if rank == 0:
                logger.info(f"Starting epoch {epoch+1}/{config['training']['epochs']}")
            
            # Set epoch for sampler (for data parallel)
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Train one epoch with DeepSpeed
            if args.pp_size > 1:
                # Pipeline parallelism training
                model_engine.train()
                train_loss = 0.0
                total_steps = len(train_dataset) // (config['dataset']['batch_size'] * args.dp_size)
                
                if rank == 0:
                    pbar = tqdm(range(total_steps), total=total_steps, desc=f"Epoch {epoch+1}")
                else:
                    pbar = range(total_steps)
                
                for step in pbar:
                    # DeepSpeed pipeline handles batching internally
                    loss = model_engine.train_batch(data_iter=train_loader)
                    train_loss += loss.item()
                    
                    if rank == 0 and step % config['logging']['log_interval'] == 0:
                        logger.info(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")
                
                train_loss /= total_steps
            else:
                # ZeRO data parallelism
                model_engine.train()
                train_loss = 0.0
                total_steps = len(train_loader)
                
                if rank == 0:
                    pbar = tqdm(enumerate(train_loader), total=total_steps, desc=f"Epoch {epoch+1}")
                else:
                    pbar = enumerate(train_loader)
                
                for step, batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # Forward pass with task='all' for loss calculation
                    outputs = model_engine(batch, task='all')
                    loss, loss_dict = loss_fn(outputs, batch)
                    
                    # Backward and optimize with DeepSpeed
                    model_engine.backward(loss)
                    model_engine.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    
                    if rank == 0:
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'lr': model_engine.get_lr()[0]
                        })
                        
                        # Log progress at intervals
                        if step % config['logging']['log_interval'] == 0:
                            logger.info(
                                f"Epoch: [{epoch+1}/{config['training']['epochs']}][{step}/{total_steps}] "
                                f"Loss: {loss.item():.4f} "
                                f"LR: {model_engine.get_lr()[0]:.6f}"
                            )
                            
                            # Log individual loss components
                            for loss_name, loss_value in loss_dict.items():
                                logger.info(f"  {loss_name}: {loss_value:.4f}")
                
                train_loss /= total_steps
            
            # Save checkpoint at intervals
            if (epoch + 1) % config['logging']['save_interval'] == 0 or epoch == config['training']['epochs'] - 1:
                # Prepare client state to save with checkpoint
                client_state = {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_config": model_config
                }
                
                # Save checkpoint with DeepSpeed
                checkpoint_path = os.path.join(str(checkpoint_dir), f"checkpoint_epoch_{epoch}")
                model_engine.save_checkpoint(checkpoint_path, client_state=client_state)
                
                if rank == 0:
                    logger.info(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")
            
            # Evaluate model at intervals
            if (epoch + 1) % config['logging']['eval_interval'] == 0:
                # Validation loop
                model_engine.eval()
                val_loss = 0.0
                total_steps = len(val_loader)
                
                with torch.no_grad():
                    if rank == 0:
                        pbar = tqdm(val_loader, total=total_steps, desc=f"Validation {epoch+1}")
                    else:
                        pbar = val_loader
                        
                    for batch in pbar:
                        # Move batch to device
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = model_engine(batch, task='all')
                        loss, loss_dict = loss_fn(outputs, batch)
                        
                        # Update metrics
                        val_loss += loss.item()
                
                val_loss /= total_steps
                
                # Average validation loss across processes
                if world_size > 1:
                    val_loss_tensor = torch.tensor([val_loss], device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_tensor.item() / world_size
                
                # Log metrics on main process
                if rank == 0:
                    logger.info(f"Validation Epoch {epoch+1}: Loss: {val_loss:.4f}")
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('LR', model_engine.get_lr()[0], epoch)
                    
                    # Save best model
                    is_best = val_loss < best_val_loss
                    best_val_loss = min(val_loss, best_val_loss)
                    
                    if is_best:
                        logger.info(f"New best validation loss: {best_val_loss:.4f}")
                        
                        # Save best checkpoint
                        client_state = {
                            "epoch": epoch,
                            "best_val_loss": best_val_loss,
                            "model_config": model_config
                        }
                        
                        best_checkpoint_path = os.path.join(str(checkpoint_dir), "best_checkpoint")
                        model_engine.save_checkpoint(best_checkpoint_path, client_state=client_state)
                        logger.info(f"Saved best checkpoint at epoch {epoch+1}")
            
            # Visualize predictions if configured
            if rank == 0 and epoch % config['logging']['visualize_every'] == 0:
                logger.info(f"Visualizing predictions for epoch {epoch+1}")
                viz_dir_epoch = viz_dir / f"epoch_{epoch+1}"
                viz_dir_epoch.mkdir(exist_ok=True)
                
                # Set model to eval mode
                model_engine.eval()
                
                # Run visualization
                try:
                    with torch.no_grad():
                        # For visualization, we often need to unwrap the model
                        # This is tricky with DeepSpeed, especially with pipeline parallelism
                        # For ZeRO, we can use the unwrapped module
                        if args.pp_size <= 1:
                            # Standard ZeRO model
                            unwrapped_model = model_engine.module
                            visualize_predictions(
                                model=unwrapped_model,
                                data_loader=val_loader,
                                device=device,
                                output_dir=viz_dir_epoch,
                                num_samples=config['logging']['num_viz_samples'],
                                rank=rank
                            )
                        else:
                            logger.info("Visualization not supported with pipeline parallelism")
                except Exception as e:
                    logger.error(f"Error during visualization: {e}")
        
        # Save final model
        if rank == 0:
            logger.info("Training complete! Saving final model.")
            client_state = {
                "epoch": config['training']['epochs'] - 1,
                "best_val_loss": best_val_loss,
                "model_config": model_config
            }
            
            final_checkpoint_path = os.path.join(str(checkpoint_dir), "final_checkpoint")
            model_engine.save_checkpoint(final_checkpoint_path, client_state=client_state)
            logger.info(f"Saved final checkpoint at {final_checkpoint_path}")
            
            # Close tensorboard writer
            if writer is not None:
                writer.close()
    
    except Exception as e:
        if rank == 0:
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try to save emergency checkpoint
            try:
                client_state = {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_config": model_config,
                    "error": str(e)
                }
                
                emergency_path = os.path.join(str(checkpoint_dir), f"emergency_checkpoint_epoch_{epoch}")
                model_engine.save_checkpoint(emergency_path, client_state=client_state)
                logger.info(f"Saved emergency checkpoint at {emergency_path}")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
    
    finally:
        # Clean up (DeepSpeed handles most of this automatically)
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

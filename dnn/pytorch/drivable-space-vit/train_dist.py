import os
import sys
import time
import argparse
import logging
import numpy as np
import random
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Tuple, Optional
import glob
import json
import subprocess

# Set CUDA_HOME properly before importing DeepSpeed
cuda_home = "/usr/local/cuda-12.6"  # Use your actual CUDA path
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

from model.model import (
    create_model, 
    create_loss_function, 
)

from model.pipeline_model import create_pipeline_model
from model.pipeline_processors import ComprehensiveInputProcessor, OutputProcessor

from utils.train_utils import (
    find_latest_checkpoint,
    generate_deepspeed_config,
    seed_everything,
)

from utils.dist_args import parse_args

from model.driving_dataset import DrivingDataset, create_dataloader
from visualize import visualize_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Basic NCCL configuration for single-node multi-GPU
os.environ["NCCL_DEBUG"] = "INFO"  # Get more diagnostic information
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Use blocking sync to prevent CUDA errors
os.environ["NCCL_P2P_LEVEL"] = "NVL"  # Use NVLINK for P2P transfers if available
os.environ["NCCL_SHM_DISABLE"] = "0"  # Ensure shared memory transport is used

# For maximum performance (faster but may have slight variations)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # Enables auto-tuner


class RepeatingLoader:
    """A data loader that can be used with DeepSpeed pipeline parallelism."""
    def __init__(self, loader):
        self.loader = loader
        self.data_iter = iter(self.loader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch_dict = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch_dict = next(self.data_iter)
        
        # Convert dictionary batch to list format for DeepSpeed pipeline
        # We need to maintain a consistent order of items
        batch_list = []
        for key in sorted(batch_dict.keys()):
            batch_list.append(batch_dict[key])
        
        # Store original batch for reference if needed
        self.original_batch = batch_dict
        
        return batch_list


def setup_distributed_environment(args):
    """Set up the distributed training environment."""
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
    deepspeed.init_distributed(dist_backend='nccl')
    
    # Get rank and local_rank from DeepSpeed/distributed setup
    rank = int(os.environ.get('RANK', 0))
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Process {rank}/{world_size} (local_rank={local_rank}) ready with device {device}")
    
    return rank, local_rank, world_size, device


def load_and_update_config(args, rank, world_size):
    """Load and update configuration from Hydra and command line args."""
    # Using default config path
    config_path = "config"
    config_name = "config"
    
    # Initialize Hydra without using the decorator
    with hydra.initialize(version_base=None, config_path=config_path):
        hydra_config = hydra.compose(config_name=config_name)
        config = OmegaConf.to_container(hydra_config, resolve=True)
    
    print(f"Configuration loaded for rank {rank}")
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Update configuration with command-line parameters
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
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
    
    return config


def setup_directories(config, output_dir, rank):
    """Create output directories for checkpoints, logs, and visualizations."""
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
        
        # Save config
        with open(output_path / 'used_config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    checkpoint_dir = output_path / 'checkpoints'
    viz_dir = output_path / 'visualizations'
    
    return output_path, checkpoint_dir, viz_dir


def handle_checkpoint_resume(args, checkpoint_dir, rank):
    """Handle checkpoint resumption logic."""
    resume = args.resume
    auto_resume = not args.resume
    
    # Auto-resume from latest checkpoint if enabled and no specific checkpoint provided
    if auto_resume and not resume and rank == 0:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume = latest_checkpoint
            logger.info(f"Auto-resuming from latest checkpoint: {resume}")
    
    return resume


def create_datasets_and_loaders(config, args, data_dir, rank, world_size, device):
    """Create datasets and dataloaders for training and validation."""
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
    
    return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler, ds_config


def initialize_model(config, args, rank):
    """Initialize the model, loss function, and if needed, pipeline model."""
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
    
    return model, loss_fn, model_config


def initialize_deepspeed(model, ds_config, args, resume, checkpoint_dir, rank):
    """Initialize DeepSpeed engine and handle checkpoint resumption."""
    start_epoch = 0
    best_val_loss = float('inf')
    
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
        # logger.info(f"Starting training from epoch {start_epoch} to {config['training']['epochs']}")
    
    return model_engine, optimizer, start_epoch, best_val_loss


def train_epoch_pipeline(model_engine, train_loader, train_dataset, config, epoch, rank, args):
    """Train for one epoch using pipeline parallelism."""
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
    return train_loss


def train_epoch_data_parallel(model_engine, train_loader, device, config, epoch, rank):
    """Train for one epoch using data parallelism."""
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
    return train_loss


def save_checkpoint(model_engine, checkpoint_dir, epoch, best_val_loss, model_config, rank):
    """Save a checkpoint of the model."""
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
    
    return checkpoint_path


def validate(model_engine, val_loader, device, loss_fn, rank, world_size):
    """Validate the model and return validation loss."""
    model_engine.eval()
    val_loss = 0.0
    total_steps = len(val_loader)
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, total=total_steps, desc="Validation")
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
    
    return val_loss


def visualize_epoch(model_engine, val_loader, device, viz_dir, config, epoch, rank, args):
    """Visualize model predictions for the current epoch."""
    if rank == 0:
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


def save_final_model(model_engine, checkpoint_dir, config, best_val_loss, model_config, rank):
    """Save the final model after training is complete."""
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


def handle_training_error(model_engine, checkpoint_dir, epoch, best_val_loss, model_config, rank, error):
    """Handle errors during training and try to save an emergency checkpoint."""
    if rank == 0:
        logger.error(f"Training failed with error: {error}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to save emergency checkpoint
        try:
            client_state = {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_config": model_config,
                "error": str(error)
            }
            
            emergency_path = os.path.join(str(checkpoint_dir), f"emergency_checkpoint_epoch_{epoch}")
            model_engine.save_checkpoint(emergency_path, client_state=client_state)
            logger.info(f"Saved emergency checkpoint at {emergency_path}")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")


def cleanup(writer, rank):
    """Clean up resources before exiting."""
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Add immediate logging to see if script starts
    print("Starting distributed training script...")
    
    # Parse arguments
    args = parse_args()
    
    # Set up distributed environment
    rank, local_rank, world_size, device = setup_distributed_environment(args)
    
    # Load and update configuration
    config = load_and_update_config(args, rank, world_size)
    
    # Set up directories
    output_path, checkpoint_dir, viz_dir = setup_directories(config, args.output_dir, rank)
    
    # Handle checkpoint resumption
    resume = handle_checkpoint_resume(args, checkpoint_dir, rank)
    
    # Log hardware info
    if rank == 0:
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
    
    # Setup tensorboard (only on main process)
    writer = SummaryWriter(log_dir=output_path / 'logs') if rank == 0 else None
    
    # Create datasets and dataloaders
    train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler, ds_config = create_datasets_and_loaders(
        config, args, args.data_dir, rank, world_size, device
    )
    
    # Initialize model
    model, loss_fn, model_config = initialize_model(config, args, rank)
    
    # Initialize DeepSpeed and load checkpoint if resuming
    model_engine, optimizer, start_epoch, best_val_loss = initialize_deepspeed(
        model, ds_config, args, resume, checkpoint_dir, rank
    )
    
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
                train_loss = train_epoch_pipeline(model_engine, train_loader, train_dataset, config, epoch, rank, args)
            else:
                # ZeRO data parallelism
                train_loss = train_epoch_data_parallel(model_engine, train_loader, device, config, epoch, rank)
            
            # Save checkpoint at intervals
            if (epoch + 1) % config['logging']['save_interval'] == 0 or epoch == config['training']['epochs'] - 1:
                save_checkpoint(model_engine, checkpoint_dir, epoch, best_val_loss, model_config, rank)
            
            # Evaluate model at intervals
            if (epoch + 1) % config['logging']['eval_interval'] == 0:
                val_loss = validate(model_engine, val_loader, device, loss_fn, rank, world_size)
                
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
                visualize_epoch(model_engine, val_loader, device, viz_dir, config, epoch, rank, args)
        
        # Save final model
        save_final_model(model_engine, checkpoint_dir, config, best_val_loss, model_config, rank)
    
    except Exception as e:
        handle_training_error(model_engine, checkpoint_dir, epoch, best_val_loss, model_config, rank, e)
    
    finally:
        # Clean up
        cleanup(writer, rank)


if __name__ == "__main__":
    main()

import os
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

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model.model import (
    create_model, 
    create_loss_function, 
    create_optimizer, 
    create_scheduler,
    save_model_checkpoint, 
    load_model_from_checkpoint
)
from model.driving_dataset import DrivingDataset, create_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Set PyTorch memory optimization environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
        # Calculate batch size based on GPU count - aim for total batch size of 32 across all GPUs
        gpu_count = torch.cuda.device_count() if args.distributed else 1
        per_gpu_batch = max(1, min(2, int(32 / gpu_count)))  # Limit to at most 2 per GPU, at least 1
        config['dataset'].setdefault('batch_size', per_gpu_batch)
        logger.info(f"Auto batch size: {per_gpu_batch} per GPU, {per_gpu_batch * gpu_count} total across {gpu_count} GPUs")
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    else:
        # Auto-set num_workers to 2 per GPU (reduced from 4 to be safer)
        gpu_count = torch.cuda.device_count() if args.distributed else 1
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
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    return args


def visualize_predictions(model, data_loader, device, output_dir, num_samples=10, rank=0):
    """Visualize model predictions"""
    # Importing visualization libraries here to avoid dependencies if not used
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    model.eval()
    
    # Only visualize on main process
    if rank != 0:
        return
        
    samples_visualized = 0
    with torch.no_grad():
        for batch in data_loader:
            if samples_visualized >= num_samples:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Visualize each sample in the batch
            for i in range(min(batch['left_images'].size(0), num_samples - samples_visualized)):
                # Get current sample
                left_img = batch['left_images'][i, -1]  # Last frame in sequence
                right_img = batch['right_images'][i, -1]
                
                # Get reconstructions if available
                left_recon = outputs.get('left_reconstructed', None)
                right_recon = outputs.get('right_reconstructed', None)
                
                # Get drivable space prediction if available
                drivable_space = outputs.get('drivable_space', None)
                
                # Create figure
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Plot original left and right images
                axes[0, 0].imshow(left_img.permute(1, 2, 0).cpu().numpy())
                axes[0, 0].set_title('Left Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(right_img.permute(1, 2, 0).cpu().numpy())
                axes[0, 1].set_title('Right Image')
                axes[0, 1].axis('off')
                
                # Plot reconstructions if available
                if left_recon is not None:
                    axes[1, 0].imshow(left_recon[i].permute(1, 2, 0).cpu().numpy())
                    axes[1, 0].set_title('Left Reconstruction')
                    axes[1, 0].axis('off')
                
                if right_recon is not None:
                    axes[1, 1].imshow(right_recon[i].permute(1, 2, 0).cpu().numpy())
                    axes[1, 1].set_title('Right Reconstruction')
                    axes[1, 1].axis('off')
                
                # Plot drivable space prediction if available
                if drivable_space is not None:
                    drivable_map = drivable_space[i].squeeze().cpu().numpy()
                    axes[0, 2].imshow(drivable_map, cmap='viridis')
                    axes[0, 2].set_title('Drivable Space Prediction')
                    axes[0, 2].axis('off')
                
                # Save figure
                fig.tight_layout()
                plt.savefig(output_dir / f'sample_{samples_visualized}.png')
                plt.close(fig)
                
                samples_visualized += 1
                if samples_visualized >= num_samples:
                    break


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, config, scaler=None, rank=0):
    # Synchronize all processes before starting training
    if True and torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            print(f"Rank {rank}: Successfully synchronized at start of epoch {epoch+1}")
        except Exception as e:
            print(f"Rank {rank}: Error during barrier synchronization: {e}")

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
        
        # Forward pass with mixed precision if enabled
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                loss, loss_dict = loss_fn(outputs, batch)
                
                # Scale loss for gradient accumulation if needed
                if config['training']['gradient_accumulation'] > 1:
                    loss = loss / config['training']['gradient_accumulation']
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Update weights if gradient accumulation steps reached
            if (step + 1) % config['training']['gradient_accumulation'] == 0:
                # Apply gradient clipping to prevent exploding gradients
                if config.get('training', {}).get('gradient_clipping', 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('training', {}).get('gradient_clipping', 1.0))
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard forward pass
            outputs = model(batch)
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Scale loss for gradient accumulation if needed
            if config['training']['gradient_accumulation'] > 1:
                loss = loss / config['training']['gradient_accumulation']
                
            # Backward pass and optimize
            loss.backward()
            
            # Update weights if gradient accumulation steps reached
            if (step + 1) % config['training']['gradient_accumulation'] == 0:
                # Apply gradient clipping to prevent exploding gradients
                if config.get('training', {}).get('gradient_clipping', 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('training', {}).get('gradient_clipping', 1.0))
                    
                optimizer.step()
                optimizer.zero_grad()
        
        # Update metrics
        running_loss += loss.item() * (config['training']['gradient_accumulation'] if config['training']['gradient_accumulation'] > 1 else 1)
        
        # Update progress bar (only on main process)
        if rank == 0:
            pbar.set_postfix({
                'loss': loss.item() * (config['training']['gradient_accumulation'] if config['training']['gradient_accumulation'] > 1 else 1),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Log progress at intervals
            if step % config['logging']['log_interval'] == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch: [{epoch+1}/{config['training']['epochs']}][{step}/{total_steps}] "
                    f"Loss: {loss.item() * (config['training']['gradient_accumulation'] if config['training']['gradient_accumulation'] > 1 else 1):.4f} "
                    f"Time: {elapsed:.2f}s "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Log individual loss components
                for loss_name, loss_value in loss_dict.items():
                    logger.info(f"  {loss_name}: {loss_value:.4f}")
    
    # Return average loss
    return running_loss / total_steps


def parse_args():
    parser = argparse.ArgumentParser(description='Train Drivable Space ViT model')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Use distributed training')
    parser.add_argument('--world_size', type=int, default=None,
                        help='Number of GPUs to use for distributed training')
    parser.add_argument('--rank', type=int, default=None,
                        help='Rank of the current process')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='Local rank for distributed training')
    
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
                        help='Batch size for training')
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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
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
    
    return parser.parse_args()


def validate(model, loader, loss_fn, device, epoch, config, rank=0):
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
            
            # Forward pass
            outputs = model(batch)
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


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    if world_size > 1:
        print(f"Initializing distributed training with rank {rank} and world_size {world_size}")
        
        # Use environment variables for multi-node training if provided
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        print(f"Using MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        
        # Set NCCL environment variables to help with debugging and stability
        os.environ['NCCL_DEBUG'] = 'INFO'
        # Remove the socket interface limitation to let NCCL auto-detect
        # os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface for local training
        
        # Add more debugging settings
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Make NCCL more robust to slow nodes
        os.environ['NCCL_IB_TIMEOUT'] = '30'    # Increase timeout for InfiniBand operations
        
        # Set timeout to 120 seconds for large setups (increased from 60)
        timeout = timedelta(seconds=120)
        
        # Set device for this process BEFORE initializing process group
        local_device = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_device)
        print(f"Set CUDA device to {local_device} for rank {rank}")
        
        # Initialize CUDA context on the assigned device
        dummy = torch.zeros(1).cuda()
        print(f"Initialized CUDA context on device {local_device} for rank {rank}")
        
        try:
            # Try NCCL for GPU training (which we know is available from check_backends.py)
            backend = "nccl"
            print(f"Initializing process group with backend {backend} for rank {rank}")
            dist.init_process_group(
                backend,
                init_method=f"env://",
                rank=rank,
                world_size=world_size,
                timeout=timeout
            )
            print(f"Successfully initialized process group with backend {backend} for rank {rank}")
            
        except Exception as e:
            print(f"NCCL initialization failed for rank {rank}: {e}")
            print(f"Rank {rank} falling back to Gloo backend")
            try:
                # Fall back to Gloo if NCCL fails
                backend = "gloo"
                dist.init_process_group(
                    backend,
                    init_method=f"env://",
                    rank=rank,
                    world_size=world_size,
                    timeout=timeout
                )
                print(f"Successfully initialized process group with backend {backend} for rank {rank}")
            except Exception as e:
                print(f"Failed to initialize process group with Gloo for rank {rank}: {e}")
                raise
    else:
        print("Running in single GPU mode")


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        print("Cleaning up distributed training")
        dist.destroy_process_group()
        print("Distributed training cleanup complete")


def create_efficient_sampler(dataset, is_distributed, world_size=None, rank=None, shuffle=True):
    """Create an efficient sampler for distributed training that doesn't load all indices into memory"""
    if is_distributed:
        # Create a distributed sampler that only loads indices for this rank
        return DistributedSampler(
            dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True  # Drop last to ensure same batch size on all GPUs
        )
    elif shuffle:
        # For non-distributed training with shuffling
        return torch.utils.data.RandomSampler(dataset)
    else:
        # For non-distributed training without shuffling
        return torch.utils.data.SequentialSampler(dataset)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(hydra_config: DictConfig):
    
    # Add immediate logging to see if script starts
    print("Starting training script...")
    
    # Handle command-line arguments (for backward compatibility)
    args = parse_args()
    args = setup_environment(args)
    
    print(f"Arguments parsed. Distributed training: {args.distributed}")
    
    # Setup distributed training
    if args.distributed:
        if args.world_size is None:
            # Start with a more conservative number of GPUs - just 4 to verify functionality
            suggested_gpus = min(4, torch.cuda.device_count())
            logger.warning(f"Starting with {suggested_gpus} GPUs (out of {torch.cuda.device_count()} available) for initial testing.")
            logger.warning(f"Once this works, you can increase to more GPUs with --nproc_per_node=N")
            args.world_size = suggested_gpus
            print(f"Using {args.world_size} GPUs for training")
        
        # Check if we have enough GPUs
        if args.world_size > torch.cuda.device_count():
            logger.warning(f"Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available. Using {torch.cuda.device_count()} GPUs.")
            args.world_size = torch.cuda.device_count()
            
        if args.rank is None:
            args.rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', args.rank % torch.cuda.device_count()))
            
        print(f"Setting up distributed training with world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}")
        
        try:
            setup_distributed(args.rank, args.world_size)
            device = torch.device(f'cuda:{args.local_rank}')
            print(f"Successfully set up distributed training on device {device}")
        except Exception as e:
            print(f"Failed to set up distributed training: {e}")
            print("Falling back to single GPU mode")
            args.distributed = False
            device = torch.device(args.device)
    else:
        device = torch.device(args.device)
        print(f"Using single GPU mode on device {device}")
    
    # Create a dict config from Hydra's DictConfig
    # This allows easier manipulation and saving
    config = OmegaConf.to_container(hydra_config, resolve=True)
    
    # Update config with command-line arguments (if provided)
    config = update_config_with_args(config, args)
    
    print("Configuration loaded and updated")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if args.rank == 0:  # Only create directories on main process
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create visualization directory
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set up logging to file
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Log hardware info
    if args.rank == 0:  # Only log on main process
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            if args.distributed:
                logger.info(f"Using distributed training with {args.world_size} GPUs")
    
    # Save merged config to output directory
    if args.rank == 0:  # Only save on main process
        with open(output_dir / 'used_config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    # Setup tensorboard (only on main process)
    if args.rank == 0:
        writer = SummaryWriter(log_dir=output_dir / 'logs')
    else:
        writer = None
    
    # Create datasets and dataloaders
    if args.rank == 0:
        logger.info("Creating datasets...")
    
    train_dataset = DrivingDataset(
        data_dir=args.data_dir,
        split='train',
        seq_len=config['dataset']['seq_len'],
        img_size=config['model']['img_size'],
        random_sequence=config['dataset']['random_sequence'],
        cache_images=config['dataset']['cache_images'],
        config=config,
    )
    
    val_dataset = DrivingDataset(
        data_dir=args.data_dir,
        split='val',
        seq_len=config['dataset']['seq_len'],
        img_size=config['model']['img_size'],
        random_sequence=False,  # Always deterministic for validation
        cache_images=config['dataset']['cache_images'],
        config=config,
    )
    
    if args.rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create distributed samplers if using distributed training
    train_sampler = create_efficient_sampler(train_dataset, args.distributed, args.world_size, args.rank)
    val_sampler = create_efficient_sampler(val_dataset, args.distributed, args.world_size, args.rank, False)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        config=config,
        debug=True  # Ensure we're not in debug mode for actual training
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle=False,
        sampler=val_sampler,
        config=config,
        debug=False  # Ensure we're not in debug mode for validation
    )
    
    # Create model
    if args.rank == 0:
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
    
    if args.resume:
        if args.rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        model, checkpoint = load_model_from_checkpoint(args.resume, device=device, config=config)
        optimizer = create_optimizer(model, config['training']['lr'], config['training']['weight_decay'], config=config)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        scheduler = create_scheduler(
            optimizer, 
            warmup_epochs=config['training']['warmup_epochs'], 
            max_epochs=config['training']['epochs'],
            min_lr=config['training']['min_lr'],
            config=config
        )
        
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            
        if args.rank == 0:
            logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.6f}")
    else:
        model = create_model(**model_config, config=config)
        optimizer = create_optimizer(model, config['training']['lr'], config['training']['weight_decay'], config=config)
        scheduler = create_scheduler(
            optimizer, 
            warmup_epochs=config['training']['warmup_epochs'], 
            max_epochs=config['training']['epochs'],
            min_lr=config['training']['min_lr'],
            config=config
        )
    
    # Print model summary on main process
    if args.rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameter count: {param_count / 1e6:.2f}M parameters")
        
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameter count: {trainable_param_count / 1e6:.2f}M parameters")
    
    # Create loss function
    loss_fn = create_loss_function(
        reconstruction_weight=config['training']['reconstruction_weight'],
        consistency_weight=config['training']['consistency_weight'],
        future_weight=config['training']['future_weight'],
        config=config
    )
    
    # Move model to device and wrap with DDP if using distributed training
    model = model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # Training loop
    if args.rank == 0:
        logger.info(f"Starting training from epoch {start_epoch} to {config['training']['epochs']}")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        if args.rank == 0:
            logger.info(f"Starting epoch {epoch+1}/{config['training']['epochs']}")
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            rank=args.rank,
        )
        
        # Update learning rate
        scheduler.step(epoch)
        
        # Validate
        if epoch % config['logging']['eval_interval'] == 0:
            val_loss = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                config=config,
                rank=args.rank,
            )
            
            # Log metrics on main process
            if args.rank == 0:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
                
                # Save best model
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                
                if is_best:
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                    
                    # Save checkpoint
                    save_model_checkpoint(
                        model=model.module if args.distributed else model,  # Unwrap DDP if needed
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=val_loss,
                        save_path=checkpoint_dir / 'best_checkpoint.pth',
                        model_config=model_config,
                        additional_data={'best_val_loss': best_val_loss},
                        config=config
                    )
            
            # Save regular checkpoint on main process
            if args.rank == 0 and epoch % config['logging']['save_interval'] == 0:
                save_model_checkpoint(
                    model=model.module if args.distributed else model,  # Unwrap DDP if needed
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=train_loss,
                    save_path=checkpoint_dir / f'checkpoint_epoch_{epoch}.pth',
                    model_config=model_config,
                    additional_data={'best_val_loss': best_val_loss},
                    config=config
                )
                
            # Visualize predictions on main process
            if args.rank == 0 and (epoch + 1) % config['logging']['visualize_every'] == 0:
                logger.info("Visualizing predictions...")
                epoch_viz_dir = viz_dir / f'epoch_{epoch+1}'
                epoch_viz_dir.mkdir(exist_ok=True)
                
                visualize_predictions(
                    model=model.module if args.distributed else model,  # Unwrap DDP if needed
                    data_loader=val_loader,
                    device=device,
                    output_dir=epoch_viz_dir,
                    num_samples=config['logging']['num_viz_samples'],
                    rank=args.rank,
                )
    
    # Save final model on main process
    if args.rank == 0:
        save_model_checkpoint(
            model=model.module if args.distributed else model,  # Unwrap DDP if needed
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config['training']['epochs'] - 1,
            loss=train_loss,
            save_path=checkpoint_dir / 'final_checkpoint.pth',
            model_config=model_config,
            additional_data={'best_val_loss': best_val_loss},
            config=config
        )
        
        logger.info("Training complete!")
        if writer is not None:
            writer.close()
    
    # Cleanup distributed training
    if args.distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()

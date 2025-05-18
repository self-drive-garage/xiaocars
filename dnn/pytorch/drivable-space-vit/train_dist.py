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
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minute timeout (in seconds)

# Manage memory more carefully
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match device IDs to PCI bus order
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly select just two GPUs for testing


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
        
        # Forward pass with mixed precision if enabled
        if scaler is not None:
            amp_device = config['training'].get('amp_device', 'cuda')
            scaler = torch.amp.GradScaler(amp_device) if config['training']['mixed_precision'] else None
            with torch.amp.autocast(amp_device):
                # Set task to 'all' to generate all outputs needed for loss calculation
                outputs = model(batch, task='all')
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
            # Set task to 'all' to generate all outputs needed for loss calculation
            outputs = model(batch, task='all')
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
    parser = argparse.ArgumentParser(description='Train Drivable Space ViT model using distributed training')
    
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
        # Use gloo backend by default for better compatibility (like in dist_test.py)
        torch.distributed.init_process_group(
            "gloo",
            timeout=timeout
        )
        print(f"Rank {rank}: Process group initialized successfully with Gloo")
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


def main():
    # Add immediate logging to see if script starts
    print("Starting distributed training script...")
    
    # Setup distributed training first
    rank, local_rank, world_size, device = setup_distributed()
    
    # Using default config path
    config_path = "config"
    config_name = "config"
    
    # Initialize Hydra without using the decorator
    with hydra.initialize(version_base=None, config_path=config_path):
        hydra_config = hydra.compose(config_name=config_name)
        config = OmegaConf.to_container(hydra_config, resolve=True)
    
    print(f"Configuration loaded for rank {rank}")
    
    # Get CLI arguments but don't use them to override Hydra config directly
    # Instead, parse them manually and update the config
    seed = 42
    resume = None
    auto_resume = True  # New parameter for automatic resuming from latest checkpoint
    data_dir = "datasets/xiaocars"
    output_dir = "outputs"
    epochs = None
    batch_size = None
    
    # Extract params from environment (passed by torchrun command)
    import sys
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--seed" and i+1 < len(args):
            seed = int(args[i+1])
        elif arg == "--resume" and i+1 < len(args):
            resume = args[i+1]
            auto_resume = False  # If resume path is explicitly specified, don't auto-resume
        elif arg == "--data_dir" and i+1 < len(args):
            data_dir = args[i+1]
        elif arg == "--output_dir" and i+1 < len(args):
            output_dir = args[i+1]
        elif arg == "--epochs" and i+1 < len(args):
            epochs = int(args[i+1])
        elif arg == "--batch_size" and i+1 < len(args):
            batch_size = int(args[i+1])
        elif arg == "--no_auto_resume" and i < len(args):
            auto_resume = False
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Update configuration with command-line parameters
    if epochs is not None:
        config['training']['epochs'] = epochs
    
    if batch_size is not None:
        config['dataset']['batch_size'] = batch_size
    else:
        # Calculate batch size based on GPU count
        gpu_count = torch.cuda.device_count()
        per_gpu_batch = max(1, min(2, int(32 / gpu_count)))  # Limit to at most 2 per GPU, at least 1
        config['dataset'].setdefault('batch_size', per_gpu_batch)
        if rank == 0:
            logger.info(f"Auto batch size: {per_gpu_batch} per GPU, {per_gpu_batch * gpu_count} total across {gpu_count} GPUs")
    
    # Auto-set num_workers to 2 per GPU (reduced from 4 to be safer)
    if config['dataset'].get('num_workers') is None:
        gpu_count = torch.cuda.device_count()
        config['dataset']['num_workers'] = min(2, os.cpu_count() // gpu_count)
    
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
    
    # Broadcast resume path from rank 0 to all processes
    if world_size > 1:
        if rank == 0:
            resume_tensor = torch.tensor([len(resume) if resume else 0], dtype=torch.long, device=device)
        else:
            resume_tensor = torch.tensor([0], dtype=torch.long, device=device)
        
        dist.broadcast(resume_tensor, 0)
        
        if rank != 0 and resume_tensor.item() > 0:
            # All other ranks receive the path from rank 0
            resume_path_tensor = torch.zeros(resume_tensor.item(), dtype=torch.uint8, device=device)
            dist.broadcast(resume_path_tensor, 0)
            resume = ''.join([chr(x) for x in resume_path_tensor.tolist()])
        elif rank == 0 and resume:
            # Rank 0 sends the path to all other ranks
            resume_path_tensor = torch.tensor([ord(c) for c in resume], dtype=torch.uint8, device=device)
            dist.broadcast(resume_path_tensor, 0)
    
    # Log hardware info
    if rank == 0:  # Only log on main process
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Using distributed training with {world_size} GPUs")
    
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
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Drop last to ensure same batch size on all GPUs
    )
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True  # Drop last to ensure same batch size on all GPUs
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle=False,  # Don't shuffle here, the sampler will do it
        sampler=train_sampler,
        config=config,
        debug=False  # Ensure we're not in debug mode for actual training
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
    
    if resume:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {resume}")
        try:
            model, checkpoint = load_model_from_checkpoint(resume, device=device, config=config)
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
                
            if rank == 0:
                logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.6f}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            if rank == 0:
                logger.warning("Starting training from scratch.")
            model = create_model(**model_config, config=config)
            optimizer = create_optimizer(model, config['training']['lr'], config['training']['weight_decay'], config=config)
            scheduler = create_scheduler(
                optimizer, 
                warmup_epochs=config['training']['warmup_epochs'], 
                max_epochs=config['training']['epochs'],
                min_lr=config['training']['min_lr'],
                config=config
            )
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
    if rank == 0:
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
    
    # Move model to device and wrap with DDP
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # Training loop
    if rank == 0:
        logger.info(f"Starting training from epoch {start_epoch} to {config['training']['epochs']}")
    
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            train_sampler.set_epoch(epoch)
                
            if rank == 0:
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
                rank=rank,
            )
            
            # Update learning rate
            scheduler.step(epoch)
            
            # Save checkpoint based on save_interval (only on main process)
            if rank == 0 and (epoch % config['logging']['save_interval'] == 0 or epoch == config['training']['epochs'] - 1):
                save_model_checkpoint(
                    model=model.module,  # Unwrap DDP
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=train_loss,
                    save_path=checkpoint_dir / f'checkpoint_epoch_{epoch}.pth',
                    model_config=model_config,
                    additional_data={'best_val_loss': best_val_loss},
                    config=config
                )
                logger.info(f"Saved checkpoint for epoch {epoch+1}")
            
            # Visualize predictions based on visualize_every parameter (only on main process)
            if rank == 0 and (epoch % config['logging']['visualize_every'] == 0 or epoch == config['training']['epochs'] - 1):
                logger.info(f"Visualizing predictions for epoch {epoch+1}...")
                epoch_viz_dir = viz_dir / f'epoch_{epoch+1}'
                epoch_viz_dir.mkdir(exist_ok=True)
                
                visualize_predictions(
                    model=model.module,  # Unwrap DDP
                    data_loader=val_loader,
                    device=device,
                    output_dir=epoch_viz_dir,
                    num_samples=config['logging']['num_viz_samples'],
                    rank=rank,
                )
            
            # Validate at regular intervals
            if epoch % config['logging']['eval_interval'] == 0:
                val_loss = validate(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    epoch=epoch,
                    config=config,
                    rank=rank,
                )
                
                # Log metrics on main process
                if rank == 0:
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
                            model=model.module,  # Unwrap DDP
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            loss=val_loss,
                            save_path=checkpoint_dir / 'best_checkpoint.pth',
                            model_config=model_config,
                            additional_data={'best_val_loss': best_val_loss},
                            config=config
                        )
                        
                        # Visualize predictions after saving best model checkpoint
                        logger.info("Visualizing predictions after best checkpoint save...")
                        best_viz_dir = viz_dir / f'best_epoch_{epoch+1}'
                        best_viz_dir.mkdir(exist_ok=True)
                        
                        visualize_predictions(
                            model=model.module,  # Unwrap DDP
                            data_loader=val_loader,
                            device=device,
                            output_dir=best_viz_dir,
                            num_samples=config['logging']['num_viz_samples'],
                            rank=rank,
                        )
        
        # Save final model on main process
        if rank == 0:
            save_model_checkpoint(
                model=model.module,  # Unwrap DDP
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=config['training']['epochs'] - 1,
                loss=train_loss,
                save_path=checkpoint_dir / 'final_checkpoint.pth',
                model_config=model_config,
                additional_data={'best_val_loss': best_val_loss},
                config=config
            )
            
            # Visualize predictions after final checkpoint
            logger.info("Visualizing predictions after final checkpoint...")
            final_viz_dir = viz_dir / 'final'
            final_viz_dir.mkdir(exist_ok=True)
            
            visualize_predictions(
                model=model.module,  # Unwrap DDP
                data_loader=val_loader,
                device=device,
                output_dir=final_viz_dir,
                num_samples=config['logging']['num_viz_samples'],
                rank=rank,
            )
            
            logger.info("Training complete!")
            if writer is not None:
                writer.close()
    
    except Exception as e:
        if rank == 0:
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Save emergency checkpoint on error
            try:
                save_model_checkpoint(
                    model=model.module,  # Unwrap DDP
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=float('inf') if 'train_loss' not in locals() else train_loss,
                    save_path=checkpoint_dir / f'emergency_checkpoint_epoch_{epoch}.pth',
                    model_config=model_config,
                    additional_data={'best_val_loss': best_val_loss, 'error': str(e)},
                    config=config
                )
                logger.info(f"Saved emergency checkpoint for epoch {epoch+1}")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
    
    finally:
        # Cleanup distributed training
        cleanup_distributed()

if __name__ == "__main__":
    main()

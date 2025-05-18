import os
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
from typing import Dict, Any

import torch
import torch.nn as nn
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
    if args.patch_size is not None:
        config['model']['patch_size'] = args.patch_size
    if args.embed_dim is not None:
        config['model']['embed_dim'] = args.embed_dim
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
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    
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


def visualize_predictions(model, data_loader, device, output_dir, num_samples=10):
    """Visualize model predictions"""
    # Importing visualization libraries here to avoid dependencies if not used
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    model.eval()
    
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


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, config, scaler=None):
    model.train()
    running_loss = 0.0
    total_steps = len(loader)
    start_time = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(enumerate(loader), total=total_steps, desc=f"Epoch {epoch+1}")
    
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
                optimizer.step()
                optimizer.zero_grad()
        
        # Update metrics
        running_loss += loss.item() * (config['training']['gradient_accumulation'] if config['training']['gradient_accumulation'] > 1 else 1)
        
        # Update progress bar
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
    
    # Config file argument
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default= '/localhome/local-samehm/workspace/xiaocars/dnn/pytorch/drivable-space-vit/datasets/xiaocars',
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


def validate(model, loader, loss_fn, device, epoch, config):
    model.eval()
    val_loss = 0.0
    total_steps = len(loader)
    
    # Use tqdm for progress bar
    pbar = tqdm(loader, total=total_steps, desc=f"Validation {epoch+1}")
    
    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Update metrics
            val_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Average validation loss
    val_loss = val_loss / total_steps
    
    logger.info(f"Validation Epoch: [{epoch+1}/{config['training']['epochs']}] Loss: {val_loss:.4f}")
    
    # Log individual loss components
    for loss_name, loss_value in loss_dict.items():
        logger.info(f"  Validation {loss_name}: {loss_value:.4f}")
    
    return val_loss

@hydra.main(config_path="config", config_name="config")
def main(hydra_config: DictConfig):
    # Handle command-line arguments (for backward compatibility)
    args = parse_args()
    args = setup_environment(args)
    
    # Create a dict config from Hydra's DictConfig
    # This allows easier manipulation and saving
    config = OmegaConf.to_container(hydra_config, resolve=True)
    
    # Update config with command-line arguments (if provided)
    config = update_config_with_args(config, args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
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
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Save merged config to output directory
    with open(output_dir / 'used_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Create datasets and dataloaders
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
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle=True,
        config=config,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle=False,
        config=config,
    )
    
    # Create model
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
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model, checkpoint = load_model_from_checkpoint(args.resume, device=args.device, config=config)
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
    
    # Print model summary
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
    
    # Move model to GPU if available
    model = model.to(device)
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {config['training']['epochs']}")
    
    for epoch in range(start_epoch, config['training']['epochs']):
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
            )
            
            # Log metrics
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
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=val_loss,
                    save_path=checkpoint_dir / 'best_checkpoint.pth',
                    model_config=model_config,
                    additional_data={'best_val_loss': best_val_loss},
                    config=config
                )
            
            # Visualize predictions right after saving checkpoint
            logger.info("Visualizing predictions after checkpoint save...")
            epoch_viz_dir = viz_dir / f'epoch_{epoch+1}'
            epoch_viz_dir.mkdir(exist_ok=True)
            
            visualize_predictions(
                model=model,
                data_loader=val_loader,
                device=device,
                output_dir=epoch_viz_dir,
                num_samples=config['logging']['num_viz_samples'],
            )
        
        # Save regular checkpoint
        if epoch % config['logging']['save_interval'] == 0:
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=train_loss,
                save_path=checkpoint_dir / f'checkpoint_epoch_{epoch}.pth',
                model_config=model_config,
                additional_data={'best_val_loss': best_val_loss},
                config=config
            )
            
            # Visualize predictions right after saving checkpoint
            logger.info("Visualizing predictions after checkpoint save...")
            epoch_viz_dir = viz_dir / f'epoch_{epoch+1}'
            epoch_viz_dir.mkdir(exist_ok=True)
            
            visualize_predictions(
                model=model,
                data_loader=val_loader,
                device=device,
                output_dir=epoch_viz_dir,
                num_samples=config['logging']['num_viz_samples'],
            )
    
    # Save final model
    save_model_checkpoint(
        model=model,
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
        model=model,
        data_loader=val_loader,
        device=device,
        output_dir=final_viz_dir,
        num_samples=config['logging']['num_viz_samples'],
    )
    
    logger.info("Training complete!")
    writer.close()

if __name__ == "__main__":
    main()

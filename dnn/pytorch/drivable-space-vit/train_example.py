import os
import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model.model import (
    create_model, create_loss_function, create_optimizer, 
    create_scheduler, save_model_checkpoint, load_model_from_checkpoint
)
from model.driving_dataset import DrivingDataset, create_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Drivable Space Vision Transformer')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save model checkpoints and logs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--consistency_weight', type=float, default=1.0,
                        help='View consistency loss weight')
    parser.add_argument('--future_weight', type=float, default=0.5,
                        help='Future prediction loss weight')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint frequency (epochs)')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Forward pass
        outputs = model(batch, task='all')
        
        # Calculate loss
        loss, loss_dict = loss_fn(outputs, batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Accumulate loss components
        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average losses
    avg_loss = total_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def validate(model, val_loader, loss_fn, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(batch, task='all')
            
            # Calculate loss
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Update metrics
            total_loss += loss.item()
            
            # Accumulate loss components
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v
    
    # Average losses
    avg_loss = total_loss / len(val_loader)
    avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets and data loaders
    train_dataset = DrivingDataset(
        data_dir=args.data_dir,
        split='train',
        img_size=args.img_size,
    )
    
    val_dataset = DrivingDataset(
        data_dir=args.data_dir,
        split='val',
        img_size=args.img_size,
    )
    
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    
    # Create model config
    model_config = {
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
    }
    
    # Create model, optimizer, loss function, and scheduler
    start_epoch = 0
    if args.resume:
        # Resume from checkpoint
        logger.info(f'Resuming from checkpoint: {args.resume}')
        model, checkpoint = load_model_from_checkpoint(args.resume, device)
        optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resuming from epoch {start_epoch}')
    else:
        # Create new model
        model = create_model(**model_config)
        model = model.to(device)
        optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    
    # Create loss function
    loss_fn = create_loss_function(
        reconstruction_weight=args.recon_weight,
        consistency_weight=args.consistency_weight,
        future_weight=args.future_weight,
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        lr = scheduler.step(epoch)
        logger.info(f'Epoch {epoch}, learning rate: {lr[0]:.6f}')
        
        # Train for one epoch
        train_loss, train_components = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_components = validate(model, val_loader, loss_fn, device)
        
        # Log metrics
        logger.info(f'Epoch {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for k, v in train_components.items():
            writer.add_scalar(f'Components/train_{k}', v, epoch)
        
        for k, v in val_components.items():
            writer.add_scalar(f'Components/val_{k}', v, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_model_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                save_path, model_config,
                additional_data={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_components': train_components,
                    'val_components': val_components,
                }
            )
            logger.info(f'Saved checkpoint to {save_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_model.pth'
            save_model_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                save_path, model_config,
                additional_data={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_components': train_components,
                    'val_components': val_components,
                }
            )
            logger.info(f'New best model saved with val_loss: {val_loss:.4f}')
    
    # Save final model
    save_path = output_dir / 'final_model.pth'
    save_model_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, val_loss,
        save_path, model_config,
        additional_data={
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_components': train_components,
            'val_components': val_components,
        }
    )
    logger.info(f'Training completed. Final model saved to {save_path}')

if __name__ == '__main__':
    main() 
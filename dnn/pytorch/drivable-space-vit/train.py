#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Training Script for Transformer-Based Self-Supervised Drivable Space Detection

This script provides a command-line interface for training the model on
stereo driving data with ego motion information.
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import model and utilities
from model import (
    StereoTransformer, 
    DrivingDataset, 
    create_dataloader,
    SelfSupervisedLoss,
    CosineSchedulerWithWarmup,
    save_checkpoint,
    load_checkpoint,
    train_one_epoch,
    validate,
    visualize_predictions
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train self-supervised drivable space detection model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    parser.add_argument('--seq_len', type=int, default=5,
                        help='Sequence length (default: 5)')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension (default: 768)')
    
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers (default: 12)')
    
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads (default: 12)')
    
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size (default: 16)')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs (default: 10)')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading (default: 8)')
    
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval in batches (default: 10)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory (default: output)')
    
    # Hardware options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Number of gradient accumulation steps (default: 1)')
    
    # Visualization
    parser.add_argument('--visualize_every', type=int, default=5,
                        help='Visualize predictions every N epochs (default: 5)')
    
    parser.add_argument('--num_viz_samples', type=int, default=10,
                        help='Number of samples to visualize (default: 10)')
    
    return parser.parse_args()


def seed_everything(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create visualization directory
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Log hardware info
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Log args
    logger.info(f"Args: {args}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = DrivingDataset(
        data_dir=args.data_dir,
        split='train',
        seq_len=args.seq_len,
        img_size=args.img_size,
    )
    
    val_dataset = DrivingDataset(
        data_dir=args.data_dir,
        split='val',
        seq_len=args.seq_len,
        img_size=args.img_size,
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
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
    
    # Create model
    logger.info("Creating model...")
    
    model = StereoTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,  # RGB images
        embed_dim=args.embed_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=4,  # Fixed MLP ratio
        dropout=args.dropout,
        attn_dropout=args.dropout,
        ego_motion_dim=6,  # Fixed ego motion dimension
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameter count: {param_count / 1e6:.2f}M")
    
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameter count: {trainable_param_count / 1e6:.2f}M")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create learning rate scheduler
    scheduler = CosineSchedulerWithWarmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )
    
    # Create loss function
    loss_fn = SelfSupervisedLoss(
        reconstruction_weight=1.0,
        consistency_weight=1.0,
        future_weight=0.5,
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            model,
            optimizer,
            scheduler,
            path=args.resume,
        )
        
        logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.6f}")
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Train model
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device,
            loss_fn=loss_fn,
            log_interval=args.log_interval,
            scaler=scaler,
            gradient_accumulation=args.gradient_accumulation,
        )
        
        logger.info(f"Epoch {epoch+1} train loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(
            model=model,
            data_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )
        
        logger.info(f"Epoch {epoch+1} validation loss: {val_loss:.6f}")
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=val_loss,
            path=checkpoint_dir / 'latest.pt',
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                path=checkpoint_dir / 'best.pt',
            )
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
        
        # Visualize predictions
        if (epoch + 1) % args.visualize_every == 0:
            logger.info("Visualizing predictions...")
            epoch_viz_dir = viz_dir / f'epoch_{epoch+1}'
            epoch_viz_dir.mkdir(exist_ok=True)
            
            visualize_predictions(
                model=model,
                data_loader=val_loader,
                device=device,
                output_dir=epoch_viz_dir,
                num_samples=args.num_viz_samples,
            )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
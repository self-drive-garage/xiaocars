"""
Training script for Drivable Space Transformer with Hydra configuration.

Features:
- Hydra configuration management
- Automatic checkpointing every 5 epochs
- Visualization of predictions every 5 epochs
- Resume training capability
- Mixed precision training
- Weights & Biases logging (optional)
- Multi-GPU support
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Optional, Dict, Any, Tuple

# Import our modules
from model import create_model, DrivableSpaceTransformer
from dataset import create_dataloaders, CityscapesParquetDataset
from dataset import get_training_augmentation, get_validation_augmentation

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    

logger = logging.getLogger(__name__)


class DrivableSpaceTrainer:
    """
    Trainer class for Drivable Space Transformer.
    
    Handles training loop, validation, checkpointing, and visualization.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        
        # Create directories
        self.output_dir = Path(cfg.training.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.vis_dir = self.output_dir / "visualizations"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        # Initialize loss and metrics
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if cfg.training.mixed_precision else None
        
        # Training state
        self.start_epoch = 0
        self.best_val_iou = 0.0
        
        # Resume if specified
        if cfg.training.resume:
            self._resume_training()
            
        # Initialize wandb if available and enabled
        if cfg.logging.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
            
    def _init_model(self) -> DrivableSpaceTransformer:
        """Initialize the model."""
        model = create_model(OmegaConf.to_container(self.cfg.model))
        
        # Multi-GPU support
        if self.cfg.training.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            
        model = model.to(self.device)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {num_params:,} parameters")
        
        return model
    
    def _init_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize data loaders."""
        train_loader, val_loader = create_dataloaders(
            parquet_dir=self.cfg.data.parquet_dir,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.data.num_workers,
            img_size=tuple(self.cfg.model.img_size),
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        if self.cfg.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
                betas=(self.cfg.optimizer.beta1, self.cfg.optimizer.beta2)
            )
        elif self.cfg.optimizer.name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")
            
        return optimizer
    
    def _init_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler."""
        if self.cfg.scheduler.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.num_epochs,
                eta_min=self.cfg.scheduler.min_lr
            )
        elif self.cfg.scheduler.name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.scheduler.step_size,
                gamma=self.cfg.scheduler.gamma
            )
        elif self.cfg.scheduler.name == "none":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler.name}")
            
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.cfg.logging.project_name,
            name=self.cfg.logging.run_name,
            config=OmegaConf.to_container(self.cfg)
        )
        wandb.watch(self.model, log_freq=100)
        
    def _resume_training(self):
        """Resume training from checkpoint."""
        checkpoint_path = self.checkpoint_dir / self.cfg.training.resume_checkpoint
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return
            
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        
        # Load scaler state for mixed precision
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') 
                               else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou,
            'config': OmegaConf.to_container(self.cfg)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with IoU: {self.best_val_iou:.4f}")
            
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
        
    def _cleanup_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
                
    def visualize_predictions(self, epoch: int, num_samples: int = 5):
        """Visualize model predictions on validation set."""
        self.model.eval()
        
        # Get random samples
        indices = np.random.choice(len(self.val_loader.dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        # Define colors for visualization
        colors = ['black', 'green', 'yellow']  # non-drivable, drivable, uncertain
        labels = ['Non-drivable', 'Drivable', 'Uncertain']
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                sample = self.val_loader.dataset[idx]
                
                # Prepare input
                image = sample['image'].unsqueeze(0).to(self.device)
                mask_gt = sample['mask'].numpy()
                
                # Get prediction
                if self.cfg.training.mixed_precision and self.scaler:
                    with autocast(device_type='cuda'):
                        output = self.model(image)
                else:
                    output = self.model(image)
                    
                pred = output.argmax(dim=1).squeeze().cpu().numpy()
                
                # Denormalize image for visualization
                img_np = image.squeeze().cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std[:, None, None]) + mean[:, None, None]
                img_np = np.clip(img_np, 0, 1)
                img_np = np.transpose(img_np, (1, 2, 0))
                
                # Plot
                axes[i, 0].imshow(img_np)
                axes[i, 0].set_title(f'Input Image\n{sample["city"]}')
                axes[i, 0].axis('off')
                
                # Ground truth
                gt_colored = np.zeros((*mask_gt.shape, 3))
                for cls in range(3):
                    gt_colored[mask_gt == cls] = plt.cm.colors.to_rgb(colors[cls])
                axes[i, 1].imshow(gt_colored)
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                # Prediction
                pred_colored = np.zeros((*pred.shape, 3))
                for cls in range(3):
                    pred_colored[pred == cls] = plt.cm.colors.to_rgb(colors[cls])
                axes[i, 2].imshow(pred_colored)
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
                
        # Add legend
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
        fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        save_path = self.vis_dir / f"predictions_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {save_path}")
        
        # Log to wandb if available
        if self.cfg.logging.use_wandb and WANDB_AVAILABLE:
            wandb.log({f"predictions": wandb.Image(str(save_path))}, step=epoch)
            
    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
        """Compute IoU metrics."""
        ious = []
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
            else:
                ious.append(float('nan'))
                
        # Compute mean IoU (excluding NaN values)
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        
        return {
            'mIoU': mean_iou,
            'IoU_non_drivable': ious[0],
            'IoU_drivable': ious[1],
            'IoU_uncertain': ious[2] if num_classes > 2 else float('nan')
        }
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.training.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            # Forward pass
            if self.cfg.training.mixed_precision and self.scaler:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.cfg.training.mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            # Update metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb
            if self.cfg.logging.use_wandb and WANDB_AVAILABLE and batch_idx % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }, step=epoch * len(self.train_loader) + batch_idx)
                
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_ious = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['images'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                            # Forward pass
            if self.cfg.training.mixed_precision and self.scaler:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
            # Get predictions
            preds = outputs.argmax(dim=1)
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Compute IoU for each sample in batch
            for pred, mask in zip(preds, masks):
                iou_metrics = self.compute_iou(pred, mask)
                all_ious.append(iou_metrics)
                
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        # Aggregate metrics
        avg_loss = total_loss / total_samples
        
        # Compute mean IoU across all samples
        mean_metrics = {
            'val_loss': avg_loss,
            'val_mIoU': np.mean([m['mIoU'] for m in all_ious]),
            'val_IoU_non_drivable': np.nanmean([m['IoU_non_drivable'] for m in all_ious]),
            'val_IoU_drivable': np.nanmean([m['IoU_drivable'] for m in all_ious]),
            'val_IoU_uncertain': np.nanmean([m['IoU_uncertain'] for m in all_ious])
        }
        
        return mean_metrics
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Config:\n{OmegaConf.to_yaml(self.cfg)}")
        
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val mIoU: {val_metrics['val_mIoU']:.4f}"
            )
            
            # Log to wandb
            if self.cfg.logging.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch
                }, step=epoch)
                
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                
            # Visualize predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.visualize_predictions(epoch)
                
            # Save best model
            if val_metrics['val_mIoU'] > self.best_val_iou:
                self.best_val_iou = val_metrics['val_mIoU']
                self.save_checkpoint(epoch, is_best=True)
                
        logger.info("Training completed!")
        
        # Final visualization
        self.visualize_predictions(self.cfg.training.num_epochs - 1)
        
        # Close wandb
        if self.cfg.logging.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Create trainer and start training
    trainer = DrivableSpaceTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
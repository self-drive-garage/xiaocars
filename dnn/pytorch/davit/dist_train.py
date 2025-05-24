"""
Distributed Training script for Drivable Space Transformer with DDP.

Features:
- DistributedDataParallel (DDP) for multi-GPU training
- Hydra configuration management
- Automatic checkpointing every 5 epochs
- Visualization of predictions every 5 epochs
- Resume training capability
- Mixed precision training
- Weights & Biases logging (optional)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


def setup_distributed():
    """Initialize distributed training environment."""
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return dist.get_rank() == 0


class DistributedDrivableSpaceTrainer:
    """
    Distributed Trainer class for Drivable Space Transformer.
    
    Handles distributed training loop, validation, checkpointing, and visualization.
    """
    
    def __init__(self, cfg: DictConfig, local_rank: int, rank: int, world_size: int):
        self.cfg = cfg
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        
        # Only create directories on main process
        if is_main_process():
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
            
        # Initialize wandb if available and enabled (only on main process)
        if cfg.logging.use_wandb and WANDB_AVAILABLE and is_main_process():
            self._init_wandb()
            
    def _init_model(self) -> DDP:
        """Initialize the model with DDP."""
        model = create_model(OmegaConf.to_container(self.cfg.model))
        model = model.to(self.device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        
        # Log model info (only on main process)
        if is_main_process():
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model initialized with {num_params:,} parameters")
            logger.info(f"Using {self.world_size} GPUs for distributed training")
        
        return model
    
    def _init_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize distributed data loaders."""
        # Create datasets
        train_dataset = CityscapesParquetDataset(
            parquet_dir=self.cfg.data.parquet_dir,
            split='train',
            transform=get_training_augmentation(tuple(self.cfg.model.img_size))
        )
        
        val_dataset = CityscapesParquetDataset(
            parquet_dir=self.cfg.data.parquet_dir,
            split='val',
            transform=get_validation_augmentation(tuple(self.cfg.model.img_size))
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )
        
        # Import collate_fn
        from dataset import collate_fn
        
        # Create data loaders
        # Note: Effective batch size = batch_size * world_size
        # Reduce workers per GPU in distributed setting to avoid memory issues
        num_workers_per_gpu = max(0, self.cfg.data.num_workers // self.world_size)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers_per_gpu > 0 else False,
            collate_fn=collate_fn,
            prefetch_factor=2 if num_workers_per_gpu > 0 else None,
            multiprocessing_context='spawn' if num_workers_per_gpu > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=val_sampler,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if num_workers_per_gpu > 0 else False,
            collate_fn=collate_fn,
            prefetch_factor=2 if num_workers_per_gpu > 0 else None,
            multiprocessing_context='spawn' if num_workers_per_gpu > 0 else None
        )
        
        if is_main_process():
            logger.info(f"Train samples per GPU: {len(train_loader.dataset) // self.world_size}")
            logger.info(f"Val samples per GPU: {len(val_loader.dataset) // self.world_size}")
            logger.info(f"Effective batch size: {self.cfg.training.batch_size * self.world_size}")
            logger.info(f"Workers per GPU: {num_workers_per_gpu} (total: {num_workers_per_gpu * self.world_size})")
        
        return train_loader, val_loader
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer with adjusted learning rate for distributed training."""
        # Scale learning rate by world size
        lr = self.cfg.optimizer.lr * self.world_size
        
        if self.cfg.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.cfg.optimizer.weight_decay,
                betas=(self.cfg.optimizer.beta1, self.cfg.optimizer.beta2)
            )
        elif self.cfg.optimizer.name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")
        
        if is_main_process():
            logger.info(f"Scaled learning rate: {lr} (base_lr * {self.world_size})")
            
        return optimizer
    
    def _init_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler."""
        if self.cfg.scheduler.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.num_epochs,
                eta_min=self.cfg.scheduler.min_lr * self.world_size
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
        """Initialize Weights & Biases logging (only on main process)."""
        wandb.init(
            project=self.cfg.logging.project_name,
            name=f"{self.cfg.logging.run_name}_ddp_{self.world_size}gpus",
            config=OmegaConf.to_container(self.cfg),
            group="ddp_training"
        )
        wandb.watch(self.model, log_freq=100)
        
    def _resume_training(self):
        """Resume training from checkpoint."""
        checkpoint_path = Path(self.cfg.training.output_dir) / "checkpoints" / self.cfg.training.resume_checkpoint
        
        if not checkpoint_path.exists():
            if is_main_process():
                logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return
            
        if is_main_process():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint with map_location to current device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
            
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
        """Save training checkpoint (only on main process)."""
        if not is_main_process():
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
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
        if not is_main_process():
            return
            
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
                
    def visualize_predictions(self, epoch: int, num_samples: int = 5):
        """Visualize model predictions on validation set (only on main process)."""
        if not is_main_process():
            return
            
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
        
        # Set epoch for distributed sampler
        self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_samples = 0
        
        # Only show progress bar on main process
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.training.num_epochs}")
        else:
            pbar = self.train_loader
        
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
            
            # Update progress bar (only on main process)
            if is_main_process() and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
            
            # Log to wandb (only on main process)
            if is_main_process() and self.cfg.logging.use_wandb and WANDB_AVAILABLE and batch_idx % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }, step=epoch * len(self.train_loader) + batch_idx)
                
        # Gather metrics across all processes
        avg_loss = total_loss / total_samples
        
        # All-reduce the average loss
        avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_ious = []
        
        # Only show progress bar on main process
        if is_main_process():
            pbar = tqdm(self.val_loader, desc="Validation")
        else:
            pbar = self.val_loader
        
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
                    
                # Update progress bar (only on main process)
                if is_main_process() and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    
        # Aggregate metrics
        avg_loss = total_loss / total_samples
        
        # Compute mean IoU across all samples
        local_metrics = {
            'val_loss': avg_loss,
            'val_mIoU': np.mean([m['mIoU'] for m in all_ious]),
            'val_IoU_non_drivable': np.nanmean([m['IoU_non_drivable'] for m in all_ious]),
            'val_IoU_drivable': np.nanmean([m['IoU_drivable'] for m in all_ious]),
            'val_IoU_uncertain': np.nanmean([m['IoU_uncertain'] for m in all_ious])
        }
        
        # All-reduce metrics across all processes
        metrics_tensor = torch.tensor([v for v in local_metrics.values()]).to(self.device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        
        # Convert back to dict
        mean_metrics = {
            k: metrics_tensor[i].item() 
            for i, k in enumerate(local_metrics.keys())
        }
        
        return mean_metrics
    
    def train(self):
        """Main training loop."""
        if is_main_process():
            logger.info("Starting distributed training...")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Config:\n{OmegaConf.to_yaml(self.cfg)}")
        
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log metrics (only on main process)
            if is_main_process():
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
                    
            # Save checkpoint every 5 epochs (only on main process)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                
            # Visualize predictions every 5 epochs (only on main process)
            if (epoch + 1) % 5 == 0:
                self.visualize_predictions(epoch)
                
            # Save best model (only on main process)
            if is_main_process() and val_metrics['val_mIoU'] > self.best_val_iou:
                self.best_val_iou = val_metrics['val_mIoU']
                self.save_checkpoint(epoch, is_best=True)
                
        if is_main_process():
            logger.info("Training completed!")
            
            # Final visualization
            self.visualize_predictions(self.cfg.training.num_epochs - 1)
            
            # Close wandb
            if self.cfg.logging.use_wandb and WANDB_AVAILABLE:
                wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main distributed training function."""
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    
    # Set random seeds
    torch.manual_seed(cfg.training.seed + rank)
    np.random.seed(cfg.training.seed + rank)
    torch.cuda.manual_seed(cfg.training.seed + rank)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        # Create trainer and start training
        trainer = DistributedDrivableSpaceTrainer(cfg, local_rank, rank, world_size)
        trainer.train()
    finally:
        # Clean up
        cleanup_distributed()


if __name__ == "__main__":
    main() 
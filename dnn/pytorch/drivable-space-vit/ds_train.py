import os
import torch
import deepspeed
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter
import glob

from ds_model.ds_modular_model import (
    create_deepspeed_model_and_engine,
    create_loss_function,
    save_deepspeed_checkpoint,
    load_deepspeed_checkpoint,
    create_deepspeed_config
)

from utils.train_utils import (
    seed_everything,
)

from utils.validate import validate

from model.driving_dataset import DrivingDataset, create_dataloader
from visualize import visualize_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Set DEBUG level for our custom modules to capture detailed tensor shapes
logging.getLogger('model').setLevel(logging.INFO)
logging.getLogger('utils').setLevel(logging.INFO)

# Environment variable optimizations for DeepSpeed
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"  # Avoid loopback and docker interfaces
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand transport
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P transfers that might cause issues

# NCCL settings for improved stability
os.environ["NCCL_TIMEOUT"] = "300"  # 5 minute timeout (in seconds)
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Use blocking wait which can be more stable

# DeepSpeed specific optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"


@hydra.main(version_base=None, config_path="config", config_name="config_deepspeed")
def main(cfg: DictConfig):
    """Main training function using DeepSpeed with Hydra configuration"""
    
    # Set seed for reproducibility
    seed_everything(42)
    
    # Initialize distributed training (DeepSpeed handles this automatically)
    deepspeed.init_distributed()
    
    # Get rank and world size
    rank = deepspeed.comm.get_rank()
    world_size = deepspeed.comm.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set CUDA device to match local rank to avoid GPU conflicts
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Initialized DeepSpeed: rank {rank} of {world_size}, local_rank {local_rank}")
    
    # Setup output directories and logging
    output_dir = Path(cfg.logging.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard logging if rank is 0
    if rank == 0:
        writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    else:
        writer = None
    
    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_dataset = DrivingDataset(
        data_dir=cfg.dataset.get('data_dir', 'datasets/argoversev2'),
        split='train',
        seq_len=cfg.dataset.seq_len,
        img_size=cfg.model.img_size,
        random_sequence=cfg.dataset.get('random_sequence', False),
        cache_images=cfg.dataset.get('cache_images', False),
        rank=rank,
        world_size=world_size
    )
    val_dataset = DrivingDataset(
        data_dir=cfg.dataset.get('data_dir', 'datasets/argoversev2'),
        split='val',
        seq_len=cfg.dataset.seq_len,
        img_size=cfg.model.img_size,
        random_sequence=cfg.dataset.get('random_sequence', False),
        cache_images=cfg.dataset.get('cache_images', False),
        rank=rank,
        world_size=world_size
    )
    
    # Get distributed training info
    is_distributed = world_size > 1

    # Create dataloaders with distributed sampling
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        is_train=True
    )

    val_loader = create_dataloader(
        val_dataset, 
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        is_train=False
    )
    
    # Create model and initialize with DeepSpeed
    model_engine, optimizer, lr_scheduler, deepspeed_config, total_params = create_deepspeed_model_and_engine(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_chans,
        embed_dim=cfg.model.embed_dim,
        spatial_layers=cfg.model.spatial_layers,
        cross_view_layers=cfg.model.cross_view_layers,
        temporal_layers=cfg.model.temporal_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout,
        ego_motion_dim=cfg.model.ego_motion_dim,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        config=cfg  # Pass the original config object instead of converted dict
    )
    
    # Create loss function
    loss_fn = create_loss_function(
        reconstruction_weight=cfg.training.reconstruction_weight,
        consistency_weight=cfg.training.consistency_weight,
        future_weight=cfg.training.future_weight,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if hasattr(cfg.training, 'resume') and cfg.training.resume:
        checkpoint_pattern = cfg.training.resume
        
        # If the pattern contains wildcards, find matching directories
        if '*' in checkpoint_pattern:
            matching_dirs = glob.glob(checkpoint_pattern)
            if matching_dirs:
                # Sort by epoch number to get the latest
                matching_dirs.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                checkpoint_dir = matching_dirs[-1]
            else:
                checkpoint_dir = None
                logger.info(f"No checkpoints found matching pattern: {checkpoint_pattern}")
        else:
            checkpoint_dir = checkpoint_pattern if os.path.exists(checkpoint_pattern) else None
            
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            logger.info(f"Loading checkpoint from {checkpoint_dir}")
            client_state = load_deepspeed_checkpoint(
                model_engine,
                checkpoint_dir,
                load_optimizer_states=True,
                load_lr_scheduler_states=True
            )
            if client_state:
                start_epoch = client_state.get('epoch', 0) + 1
                logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Print model and training info
    if rank == 0:
        logger.info(f"Total model parameters: {total_params:,}")
        logger.info(f"DeepSpeed config: {deepspeed_config}")
        logger.info(f"Effective batch size: {deepspeed_config['train_batch_size']}")
        logger.info(f"Gradient accumulation steps: {deepspeed_config['gradient_accumulation_steps']}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, cfg.training.epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(
            model_engine=model_engine,
            train_loader=train_loader,
            loss_fn=loss_fn,
            epoch=epoch,
            cfg=cfg,
            writer=writer,
            rank=rank,
            world_size=world_size
        )
        
        # Ensure all processes are synchronized before validation
        if world_size > 1:
            deepspeed.comm.barrier()
        
        # Validate
        val_loss = validate_deepspeed(
            model_engine=model_engine,
            loader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch,
            config=OmegaConf.to_container(cfg, resolve=True),
            rank=rank,
            world_size=world_size
        )
        
        # Ensure all processes are synchronized after validation
        if world_size > 1:
            deepspeed.comm.barrier()
        
        # Step learning rate scheduler if using custom scheduler
        if lr_scheduler and hasattr(lr_scheduler, 'step'):
            lr_scheduler.step()
        
        # Save checkpoint (all ranks must participate in DeepSpeed checkpoint saving)
        if (epoch + 1) % cfg.logging.save_interval == 0:
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            if rank == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # All ranks must call save_checkpoint for DeepSpeed
            save_deepspeed_checkpoint(
                model_engine=model_engine,
                epoch=epoch,
                loss=train_loss,
                save_dir=str(checkpoint_dir),
                model_config=cfg.model,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            
            if rank == 0:
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save best model (all ranks must participate in DeepSpeed checkpoint saving)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_dir = output_dir / "best_model"
            if rank == 0:
                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure directory exists on all ranks before saving
            if world_size > 1:
                deepspeed.comm.barrier()
            
            # All ranks must call save_checkpoint for DeepSpeed
            save_deepspeed_checkpoint(
                model_engine=model_engine,
                epoch=epoch,
                loss=val_loss,
                save_dir=str(best_checkpoint_dir),
                model_config=cfg.model,
                additional_data={'best_val_loss': best_val_loss},
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            
            # if rank == 0:
            #     logger.info(f"Saved best model with validation loss {best_val_loss:.6f}")
        
        # Visualize predictions if configured
        # if (epoch + 1) % cfg.logging.visualize_every == 0:
        #     logger.info(f"Rank {rank}: Visualization triggered for epoch {epoch}")
        #     if rank == 0:
        #         visualize_dir = output_dir / 'visualizations' / f"epoch_{epoch}"
        #         visualize_dir.mkdir(parents=True, exist_ok=True)
        #         logger.info(f"Rank 0: Starting visualization, saving to {visualize_dir}")
        #         visualize_predictions_deepspeed(
        #             model_engine,
        #             val_loader,
        #             epoch,
        #             cfg.logging.num_viz_samples,
        #             str(visualize_dir)
        #         )
        #         logger.info(f"Rank 0: Completed visualization for epoch {epoch}")
        #     else:
        #         logger.info(f"Rank {rank}: Waiting for rank 0 to complete visualization")
            
        #     # All processes wait for rank 0 to finish visualization
        #     if world_size > 1:
        #         logger.info(f"Rank {rank}: Entering visualization barrier")
        #         deepspeed.comm.barrier()
        #         logger.info(f"Rank {rank}: Passed visualization barrier")
        
        # Add end-of-epoch logging
        # logger.info(f"Rank {rank}: Completed epoch {epoch}")
        
        # Add barrier at end of epoch to ensure all ranks are synchronized
        if world_size > 1:
            # logger.info(f"Rank {rank}: Entering end-of-epoch barrier")
            deepspeed.comm.barrier()
            # logger.info(f"Rank {rank}: Passed end-of-epoch barrier")
    
    # Final cleanup
    if writer is not None:
        writer.close()
    
    logger.info("Training completed!")
    return model_engine

def train_epoch(model_engine, train_loader, loss_fn, epoch, cfg, writer, rank, world_size):
    """Train for one epoch using DeepSpeed"""
    model_engine.train()
    
    # Track metrics
    total_loss = 0.0
    num_batches = 0
    
    # Set up progress logging
    log_interval = cfg.logging.log_interval
    total_batches = len(train_loader)
    
    # Log the number of batches this rank will process
    if rank == 0:
        logger.info(f"Training epoch {epoch}: Processing {total_batches} batches")
    
    # Synchronize the number of batches across all ranks
    if world_size > 1:
        # Get the minimum number of batches across all ranks
        local_batches = torch.tensor([total_batches], dtype=torch.int32).cuda()
        torch.distributed.all_reduce(local_batches, op=torch.distributed.ReduceOp.MIN)
        min_batches = local_batches.item()
        
        if total_batches != min_batches:
            logger.warning(f"Rank {rank}: Batch count mismatch! Local: {total_batches}, Global min: {min_batches}")
            logger.warning(f"Rank {rank}: Will only process {min_batches} batches to maintain synchronization")
        
        total_batches = min_batches
    
    # Main training loop
    for batch_idx, batch in enumerate(train_loader):
        # Stop if we've reached the synchronized batch count
        if batch_idx >= total_batches:
        #     logger.info(f"Rank {rank}: Stopping at batch {batch_idx} to maintain synchronization")
            break
            
        # Move data to current device (DeepSpeed handles device placement)
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model_engine(batch)
        
        # Calculate loss - SelfSupervisedLoss returns (loss_tensor, loss_dict)
        loss_output = loss_fn(outputs, batch)
        
        # Handle both tuple output (loss, loss_dict) and direct tensor output
        if isinstance(loss_output, tuple):
            loss, loss_dict = loss_output
            loss_value = loss_dict.get('total_loss', 0.0)
        else:
            loss = loss_output
            loss_value = loss.item()
        
        # Debug: Check tensor dtypes to diagnose the backward pass issue
        model_dtype = next(model_engine.parameters()).dtype
        if rank == 0 and batch_idx == 0:  # Log once per epoch
            logger.info(f"Model parameter dtype: {model_dtype}")
            logger.info(f"Loss dtype: {loss.dtype}")
            
            # Check some output dtypes
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"Output '{key}' dtype: {value.dtype}")
        
        # Ensure loss is in the same dtype as model parameters
        if loss.dtype != model_dtype:
            logger.warning(f"Converting loss from {loss.dtype} to {model_dtype}")
            loss = loss.to(model_dtype)
        
        # DeepSpeed backward pass (handles gradient scaling and accumulation)
        # raise Exception(f"loss: {loss.dtype}")
        model_engine.backward(loss)
        
        # DeepSpeed step (handles gradient accumulation and optimizer step internally)
        model_engine.step()
        
        # Update metrics
        total_loss += loss_value
        num_batches += 1
        
        # Log progress
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            current_lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else cfg.training.lr
            logger.info(f"Epoch {epoch} | Batch {batch_idx+1}/{total_batches} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
            
            # Log to tensorboard
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', avg_loss, global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                # Log memory usage if available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    writer.add_scalar('train/memory_allocated_gb', memory_allocated, global_step)
                    writer.add_scalar('train/memory_reserved_gb', memory_reserved, global_step)
        
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Synchronize loss across processes
    if world_size > 1:
        # Use DeepSpeed's communication backend
        avg_loss_tensor = torch.tensor([avg_loss]).cuda()
        deepspeed.comm.all_reduce(avg_loss_tensor)
        avg_loss = (avg_loss_tensor / world_size).item()
    
    # Log epoch summary
    if rank == 0:
        logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    # Log that training epoch is complete for debugging
    if rank == 0:
        logger.info(f"Completed training epoch {epoch}")
    
    # Ensure all processes complete the epoch before returning
    if world_size > 1:
        deepspeed.comm.barrier()
    
    return avg_loss

def validate_deepspeed(model_engine, loader, loss_fn, epoch, config, rank, world_size):
    """Validate model using DeepSpeed engine"""
    if rank == 0:
        logger.info(f"Starting validation for epoch {epoch}")
    model_engine.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Get total batches and synchronize across ranks
    total_batches = len(loader)
    
    # Synchronize the number of batches across all ranks (same as in training)
    if world_size > 1:
        # Get the minimum number of batches across all ranks
        local_batches = torch.tensor([total_batches], dtype=torch.int32).cuda()
        torch.distributed.all_reduce(local_batches, op=torch.distributed.ReduceOp.MIN)
        min_batches = local_batches.item()
        
        if total_batches != min_batches:
            logger.warning(f"Rank {rank}: Validation batch count mismatch! Local: {total_batches}, Global min: {min_batches}")
            logger.warning(f"Rank {rank}: Will only process {min_batches} batches to maintain synchronization")
        
        total_batches = min_batches
    
    if rank == 0:
        logger.info(f"Validation epoch {epoch}: Processing {total_batches} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Stop if we've reached the synchronized batch count
            if batch_idx >= total_batches:
                # logger.info(f"Rank {rank}: Stopping validation at batch {batch_idx} to maintain synchronization")
                break
                
            # Move data to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(batch)
            
            # Calculate loss
            loss_output = loss_fn(outputs, batch)
            
            # Handle both tuple output (loss, loss_dict) and direct tensor output
            if isinstance(loss_output, tuple):
                loss, loss_dict = loss_output
                loss_value = loss_dict.get('total_loss', 0.0)
            else:
                loss = loss_output
                loss_value = loss.item()
            
            total_loss += loss_value
            num_batches += 1
            
            # Log progress every 50 batches
            if rank == 0 and (batch_idx + 1) % 50 == 0:
                logger.info(f"Validation progress: {batch_idx + 1}/{total_batches} batches")
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # logger.info(f"Rank {rank}: Completed validation loop, syncing loss...")
    
    # Synchronize validation loss across processes
    if world_size > 1:
        avg_loss_tensor = torch.tensor([avg_loss]).cuda()
        deepspeed.comm.all_reduce(avg_loss_tensor)
        avg_loss = (avg_loss_tensor / world_size).item()
    
    if rank == 0:
        logger.info(f"Epoch {epoch} | Validation Loss: {avg_loss:.6f}")
    
    # logger.info(f"Rank {rank}: Validation complete for epoch {epoch}")
    
    return avg_loss

def visualize_predictions_deepspeed(model_engine, val_loader, epoch, num_samples, output_dir):
    """Visualize predictions using DeepSpeed model engine"""
    logger.info(f"visualize_predictions_deepspeed: Starting visualization for {num_samples} samples")
    model_engine.eval()
    
    samples_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
            
            logger.info(f"visualize_predictions_deepspeed: Processing batch {batch_idx}")
            
            # Move data to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(batch)
            
            # Save visualizations (implement based on your visualization needs)
            # This is a placeholder - you'll need to implement the actual visualization logic
            logger.info(f"Generated visualization {batch_idx} for epoch {epoch}")
            samples_processed += 1
    
    logger.info(f"visualize_predictions_deepspeed: Completed {samples_processed} visualizations")

def parse_deepspeed_args():
    """Parse and remove DeepSpeed-specific arguments before Hydra processes them"""
    import sys
    
    # Remove --local_rank argument that DeepSpeed adds
    filtered_args = []
    skip_next = False
    
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
            
        if arg.startswith('--local_rank'):
            if '=' in arg:
                # --local_rank=0 format
                continue
            else:
                # --local_rank 0 format
                skip_next = True
                continue
        
        filtered_args.append(arg)
    
    sys.argv = filtered_args

if __name__ == "__main__":
    # Parse DeepSpeed arguments before Hydra initialization
    parse_deepspeed_args()
    main() 
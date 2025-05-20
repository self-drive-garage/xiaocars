import os
import functools
import torch
import torch.distributed as dist
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from model.model import (
    create_model, 
    create_loss_function, 
    create_optimizer,
    create_scheduler,
    save_model_checkpoint
)

from model.pipeline_model import create_pipeline_model
from model.pipeline_processors import ComprehensiveInputProcessor, OutputProcessor

from utils.train_utils import (
    seed_everything,
)


from utils.validate import validate

from model.driving_dataset import DrivingDataset, create_dataloader
from visualize import visualize_predictions

# Import transformer layer classes for wrapping policy
from model.transformer_encoder_layer import TransformerEncoderLayer
from model.cross_view_transformer_layer import CrossViewTransformerLayer
from model.temporal_transfomer_layer import TemporalTransformerLayer
# Add imports for additional components
from model.patch_embed import PatchEmbed
from model.ego_motion_encoder import EgoMotionEncoder, MotionGuidedAttention
from model.drivable_space_decoder import DrivableSpaceDecoder
from model.future_predictor import MotionGuidedFuturePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "8388608"  # 8MB for debug traces
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"  # Avoid loopback and docker interfaces
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand transport
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P transfers that might cause issues


@hydra.main(config_path="config", config_name="config_pytorch_fsdp")
def main(cfg: DictConfig):
    """Main training function using PyTorch FSDP with Hydra configuration"""
    # Parse distributed training arguments
    
    # Set seed for reproducibility
    seed_everything(42)
    
    # Initialize distributed training
    rank, local_rank, world_size = init_distributed_training(cfg)
    
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
    
     # Create standard model and wrap with FSDP
    model = create_fsdp_model(cfg)
    
    # Create loss function, optimizer and scheduler
    loss_fn = create_loss_function(
        reconstruction_weight=cfg.training.reconstruction_weight,
        consistency_weight=cfg.training.consistency_weight,
        future_weight=cfg.training.future_weight,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    optimizer = create_optimizer(
        model,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    scheduler = create_scheduler(
        optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.epochs,
        min_lr=cfg.training.min_lr,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    # if cfg.resume:
    #     checkpoint_path = cfg.resume
    #     if os.path.exists(checkpoint_path):
    #         map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            
    #         # Determine the state dict type based on config
    #         fsdp_state_dict_type = "SHARDED_STATE_DICT"  # Default for FSDP
    #         if hasattr(cfg.training, 'fsdp_state_dict_type'):
    #             fsdp_state_dict_type = cfg.training.fsdp_state_dict_type
            
    #         # Load checkpoint - different handling for FSDP sharded state dict
    #         if isinstance(model, FSDP) and fsdp_state_dict_type == "SHARDED_STATE_DICT":
    #             logger.info(f"Loading FSDP sharded checkpoint from {checkpoint_path}")
    #             from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    #             from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                
    #             # Load metadata
    #             metadata = torch.load(os.path.join(os.path.dirname(checkpoint_path), "metadata.pt"), 
    #                                  map_location=map_location)
    #             start_epoch = metadata.get('epoch', 0) + 1
                
    #             # Set up FSDP state dict configuration
    #             with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    #                 checkpoint = torch.load(checkpoint_path, map_location=map_location)
    #                 model.load_state_dict(checkpoint['model_state_dict'])
                
    #             # Load optimizer and scheduler states if required
    #             if 'optimizer_checkpoint' in metadata and os.path.exists(metadata['optimizer_checkpoint']):
    #                 optimizer_state = torch.load(metadata['optimizer_checkpoint'], map_location=map_location)
    #                 optimizer.load_state_dict(optimizer_state)
                
    #             if scheduler and 'scheduler_checkpoint' in metadata and os.path.exists(metadata['scheduler_checkpoint']):
    #                 scheduler_state = torch.load(metadata['scheduler_checkpoint'], map_location=map_location)
    #                 scheduler.load_state_dict(scheduler_state)
    #         else:
    #             # Regular checkpoint loading
    #             checkpoint = torch.load(checkpoint_path, map_location=map_location)
    #             start_epoch = checkpoint.get('epoch', 0) + 1
                
    #             # Load model weights
    #             if isinstance(model, FSDP):
    #                 # Use FSDP load_state_dict with strict=False to handle sharded state dict
    #                 model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #             else:
    #                 model.load_state_dict(checkpoint['model_state_dict'])
                
    #             # Load optimizer and scheduler states
    #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #             if scheduler and 'scheduler_state_dict' in checkpoint:
    #                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
    #         logger.info(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, cfg.training.epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            cfg=cfg,
            writer=writer,
            rank=rank,
            world_size=world_size
        )
        
        # Validate
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch,
            cfg=cfg,
            writer=writer,
            rank=rank,
            world_size=world_size
        )
        
        # Step learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % cfg.logging.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            
            # Determine the state dict type based on config
            fsdp_state_dict_type = "SHARDED_STATE_DICT"  # Default for FSDP
            if hasattr(cfg.training, 'fsdp_state_dict_type'):
                fsdp_state_dict_type = cfg.training.fsdp_state_dict_type
            
            # Save with appropriate FSDP handling
            if isinstance(model, FSDP) and fsdp_state_dict_type == "SHARDED_STATE_DICT":
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                
                # Set up FSDP state dict configuration
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    model_state = {"model_state_dict": model.state_dict()}
                    torch.save(model_state, str(checkpoint_path))
                
                # Save optimizer and scheduler separately
                if optimizer:
                    opt_path = output_dir / f"optimizer_epoch_{epoch}.pt"
                    torch.save(optimizer.state_dict(), str(opt_path))
                    
                if scheduler:
                    sch_path = output_dir / f"scheduler_epoch_{epoch}.pt"
                    torch.save(scheduler.state_dict(), str(sch_path))
                
                # Save metadata
                metadata = {
                    'epoch': epoch,
                    'loss': train_loss,
                    'model_config': cfg.model,
                    'optimizer_checkpoint': str(opt_path) if optimizer else None,
                    'scheduler_checkpoint': str(sch_path) if scheduler else None,
                    'config': OmegaConf.to_container(cfg, resolve=True)
                }
                torch.save(metadata, str(output_dir / f"metadata.pt"))
                logger.info(f"Saved FSDP sharded checkpoint to {checkpoint_path}")
            else:
                # Regular checkpoint saving
                save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=train_loss,
                    save_path=str(checkpoint_path),
                    model_config=cfg.model,
                    config=OmegaConf.to_container(cfg, resolve=True)
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_dir / "best_model.pt"
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                save_path=str(best_checkpoint_path),
                model_config=cfg.model,
                additional_data={'best_val_loss': best_val_loss},
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            logger.info(f"Saved best model with validation loss {best_val_loss:.6f}")
        
        # Visualize predictions if configured
        if rank == 0 and (epoch + 1) % cfg.logging.visualize_every == 0:
            visualize_dir = output_dir / 'visualizations' / f"epoch_{epoch}"
            visualize_dir.mkdir(parents=True, exist_ok=True)
            visualize_predictions(
                model=model,
                val_loader=val_loader,
                epoch=epoch,
                num_samples=cfg.logging.num_viz_samples,
                output_dir=str(visualize_dir),
                device=f"cuda:{local_rank}",
                cfg=OmegaConf.to_container(cfg, resolve=True)
            )
    
    # Final cleanup
    if writer is not None:
        writer.close()
    
    logger.info("Training completed!")
    return model

def init_distributed_training(cfg):
    """Initialize distributed training and return rank information"""
    # Set CUDA device
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) 
    torch.cuda.set_device(local_rank)
    
    # Initialize process group if not already initialized
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    # Get rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    logger.info(f"Initialized process group: rank {rank} of {world_size}")
    return rank, local_rank, world_size

def create_fsdp_model(cfg):
    """Create model and wrap with FSDP based on configuration"""
    # Create base model
    base_model = create_model(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.num_channels,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout,
        ego_motion_dim=cfg.model.ego_motion_dim,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Configure FSDP options
    
    # 1. Determine transformer layers for auto wrapping policy
    transformer_layer_classes = {
         # Add important model components for proper sharding
        PatchEmbed,  # Image patch embedding (convolutional)
        EgoMotionEncoder,  # Ego motion processing
        TransformerEncoderLayer,
        CrossViewTransformerLayer,
        TemporalTransformerLayer,
        MotionGuidedAttention,  # Motion-guided attention
        DrivableSpaceDecoder,  # Decoder for drivable space
        MotionGuidedFuturePredictor  # Future prediction component
    }
    
    # 2. Create auto wrapping policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_classes
    )
    
    # 3. Add a size-based policy as a fallback for any large modules not explicitly listed
    # Modules with more than 100K parameters will also be wrapped
    MIN_PARAMS_SIZE = 100000  # Wrap modules with >100K parameters
    
    # Create a combined policy that first checks for specific layer types, then falls back to size-based
    def combined_auto_wrap_policy(module, recurse, unwrapped_params=None, nonwrapped_numel=None, 
                                 min_params=MIN_PARAMS_SIZE, **kwargs):
        """Custom policy combining transformer layers and size-based policies"""
        # Use whichever parameter is provided
        param_size = unwrapped_params if unwrapped_params is not None else nonwrapped_numel
        
        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes
        )
        
        size_policy = functools.partial(
            size_based_auto_wrap_policy, 
            min_num_params=min_params
        )
        
        # First try the transformer-based policy
        if transformer_policy(module, recurse, param_size):
            return True
        
        # If that doesn't match, try the size-based policy
        return size_policy(module, recurse, param_size)
    
    # Use the combined policy
    auto_wrap_policy = combined_auto_wrap_policy
    
    # 4. Configure mixed precision
    mixed_precision_config = None
    if cfg.training.mixed_precision:
        if torch.cuda.is_bf16_supported():
            # Use BFloat16 if supported (preferred for stability with transformers)
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
        else:
            # Fall back to FP16
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
    
    # 5. Configure sharding strategy based on zero_stage
    sharding_strategy = ShardingStrategy.FULL_SHARD  # Default to ZeRO-3 equivalent
    # if hasattr(cfg.training, 'zero_stage'):
    #     if cfg.training.zero_stage == 1:
    #         sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    #     elif cfg.training.zero_stage == 2:
    #         sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    # Use native FSDP config parameters if available
    # elif hasattr(cfg.training, 'fsdp_sharding_strategy'):
        # if cfg.training.fsdp_sharding_strategy == "SHARD_GRAD_OP":
        #     sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        # elif cfg.training.fsdp_sharding_strategy == "FULL_SHARD":
        #     sharding_strategy = ShardingStrategy.FULL_SHARD
        # elif cfg.training.fsdp_sharding_strategy == "NO_SHARD":
        #     sharding_strategy = ShardingStrategy.NO_SHARD
    
    # 6. Configure CPU offloading
    cpu_offload = None
    # if hasattr(cfg.training, 'zero_cpu_offload') and cfg.training.zero_cpu_offload:
    #     cpu_offload = CPUOffload(offload_params=True)
    # # Use native FSDP config parameters if available
    # elif hasattr(cfg.training, 'fsdp_cpu_offload') and cfg.training.fsdp_cpu_offload:
    #     cpu_offload = CPUOffload(offload_params=True)
    
    # Configure backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE  # Default
    if hasattr(cfg.training, 'fsdp_backward_prefetch'):
        if cfg.training.fsdp_backward_prefetch == "BACKWARD_POST":
            backward_prefetch = BackwardPrefetch.BACKWARD_POST
        elif cfg.training.fsdp_backward_prefetch == "BACKWARD_PRE":
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    
    # Configure use_orig_params
    use_orig_params = False
    if hasattr(cfg.training, 'fsdp_use_orig_params'):
        use_orig_params = cfg.training.fsdp_use_orig_params
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        mixed_precision=mixed_precision_config,
        device_id=torch.cuda.current_device(),
        use_orig_params=use_orig_params,
    )
    
    # Apply activation checkpointing if needed
    use_activation_checkpointing = cfg.training.get('activation_checkpointing', False)
    if use_activation_checkpointing:
        apply_activation_checkpointing(fsdp_model, transformer_layer_classes)
    
    return fsdp_model

def apply_activation_checkpointing(model, transformer_layer_classes):
    """Apply activation checkpointing to transformer layers"""
    def check_fn(submodule):
        return any(isinstance(submodule, cls) for cls in transformer_layer_classes)
    
    # Apply checkpoint wrapper to eligible modules
    for m in model.modules():
        if check_fn(m):
            checkpoint_wrapper(m)
    
    logger.info("Applied activation checkpointing to transformer layers")

def train_epoch(model, train_loader, optimizer, loss_fn, epoch, cfg, writer, rank, world_size):
    """Train for one epoch"""
    model.train()
    
    # Configure gradient accumulation
    grad_accum_steps = cfg.training.gradient_accumulation
    
    # Configure gradient clipping
    grad_clip_value = cfg.training.gradient_clipping
    
    # Track metrics
    total_loss = 0.0
    num_batches = 0
    
    # Set up progress logging
    log_interval = cfg.logging.log_interval
    total_batches = len(train_loader)
    
    # Main training loop
    for batch_idx, batch in enumerate(train_loader):
        # Move data to current device
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate loss
        loss = loss_fn(outputs, batch)
        loss_value = loss.item()
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if at accumulation step or last batch
        if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx + 1 == len(train_loader):
            # Clip gradients if configured
            if grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss_value
        num_batches += 1
        
        # Log progress
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch} | Batch {batch_idx+1}/{total_batches} | Loss: {avg_loss:.6f}")
            
            # Log to tensorboard
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', avg_loss, global_step)
                
                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/learning_rate', current_lr, global_step)
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Synchronize loss across processes
    if world_size > 1:
        torch.distributed.all_reduce(torch.tensor([avg_loss]).cuda(), op=torch.distributed.ReduceOp.SUM)
        avg_loss = avg_loss / world_size
    
    # Log epoch summary
    if rank == 0:
        logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    return avg_loss

if __name__ == "__main__":
    main()

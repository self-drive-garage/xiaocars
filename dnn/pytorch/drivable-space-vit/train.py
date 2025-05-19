import os
import functools
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
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
from torch.utils.checkpoint import checkpoint_wrapper, activate_checkpoint_wrapper

# Import your model factory functions
from model.model import create_model, create_loss_function, create_optimizer, create_scheduler
from model.multi_view_transformer import MultiViewTransformer, TransformerBlock  # Assuming TransformerBlock is the correct layer class

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Print configuration for debugging
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize distributed process group
    init_distributed_training(cfg)
    
    # Create model using configuration from hydra
    model = create_model_with_fsdp(cfg)
    
    # Create optimizer, scheduler and loss function
    optimizer = create_optimizer(model, 
                               lr=cfg.training.lr, 
                               weight_decay=cfg.training.weight_decay, 
                               config=OmegaConf.to_container(cfg, resolve=True))
    
    scheduler = create_scheduler(optimizer, 
                               warmup_epochs=cfg.training.warmup_epochs, 
                               max_epochs=cfg.training.epochs, 
                               min_lr=cfg.training.min_lr, 
                               config=OmegaConf.to_container(cfg, resolve=True))
    
    loss_fn = create_loss_function(
        reconstruction_weight=cfg.training.reconstruction_weight,
        consistency_weight=cfg.training.consistency_weight,
        future_weight=cfg.training.future_weight,
        config=OmegaConf.to_container(cfg, resolve=True))
    
    # Training loop would go here...
    
    return model, optimizer, scheduler, loss_fn

def init_distributed_training(cfg):
    """Initialize distributed training environment"""
    # Get local rank from environment variable
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize process group
    if torch.distributed.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        print(f"Initialized process group: rank {torch.distributed.get_rank()} of {torch.distributed.get_world_size()}")
    else:
        print("Distributed training not available, running in single-GPU mode")

def create_model_with_fsdp(cfg):
    """Create model and wrap with FSDP based on configuration"""
    # First create the base model using create_model from model.py
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
    
    # Get local rank for device id
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Define the transformer layer wrapping policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}  # Use your actual transformer layer class
    )
    
    # Determine sharding strategy from config
    sharding_strategy = ShardingStrategy.FULL_SHARD  # Default ZeRO-3
    if hasattr(cfg.training, 'zero_stage'):
        if cfg.training.zero_stage == 1:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP  # ZeRO-1
        elif cfg.training.zero_stage == 2:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD  # ZeRO-2
    
    # Configure CPU offloading if specified
    cpu_offload = None
    if hasattr(cfg.training, 'zero_cpu_offload') and cfg.training.zero_cpu_offload:
        cpu_offload = CPUOffload(offload_params=True)
    
    # Configure mixed precision
    mixed_precision_config = None
    if cfg.training.mixed_precision:
        mixed_precision_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # Wrap the model with FSDP
    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mixed_precision_config,
        device_id=torch.cuda.current_device(),
    )
    
    # Apply activation checkpointing if configured
    if cfg.deepspeed.activation_checkpointing:
        apply_activation_checkpointing(model, cfg)
    
    return model

def apply_activation_checkpointing(model, cfg):
    """Apply activation checkpointing to the model"""
    # Use non_reentrant to save memory
    check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
    
    # Apply checkpointing using the helper function
    for module in model.modules():
        if check_fn(module):
            module = checkpoint_wrapper(module)
    
    # Activate checkpoint wrapper
    activate_checkpoint_wrapper(model)
    
    print("Activation checkpointing applied to transformer blocks")

if __name__ == "__main__":
    main()

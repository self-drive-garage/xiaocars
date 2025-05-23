import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam

from .ds_modular_vision_transformer import DeepSpeedModularVisionTransformer
from model.self_supervised_loss import SelfSupervisedLoss
from model.cosine_scheduler_with_warmup import CosineSchedulerWithWarmup

def get_default_model_config():
    """Return default model configuration"""
    return {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'embed_dim': 768,
        'spatial_layers': 4,
        'cross_view_layers': 4,
        'temporal_layers': 4,
        'num_heads': 12,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'ego_motion_dim': 9,
    }

def get_default_training_config():
    """Return default training configuration"""
    return {
        'reconstruction_weight': 1.0,
        'consistency_weight': 1.0,
        'future_weight': 0.5,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'epochs': 100,
        'min_lr': 1e-6,
    }

def create_deepspeed_config():
    """Create DeepSpeed configuration dictionary from Hydra config"""
    
    # Use fixed batch size and gradient accumulation for now
    batch_size = 2  # Per GPU batch size
    grad_accum = 8  # Gradient accumulation steps
    train_batch_size = batch_size * grad_accum * 4  # Total effective batch size per GPU
    
    # Simple, reliable DeepSpeed configuration
    deepspeed_config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "wall_clock_breakdown": False,
        
        # ZeRO-3 configuration with safe defaults
        "zero_optimization": {
            "stage": 3,
            "cpu_offload": False,
            "overlap_comm": True,
            "contiguous_gradients": True,
            # "bucket_size": 200000000,  # 200M (safe value)
            # "stage3_max_live_parameters": 1000000000,  # 1B parameters
            # "stage3_max_reuse_distance": 1000000000,  # 1B parameters  
            # "stage3_prefetch_bucket_size": 50000000,  # 50M parameters
            # "stage3_param_persistence_threshold": 1000000,  # 1M parameters
            "gather_16bit_weights_on_model_save": True,
        },
        
        # FP16 mixed precision
        "fp16": {
            "enabled": True,
            "loss_scale": 0,  # Dynamic loss scaling
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        
        # Gradient clipping
        "gradient_clipping": 1.0,
    }
    
    return deepspeed_config

def create_modular_model(
    img_size=None,
    patch_size=None,
    in_chans=None,
    embed_dim=None,
    spatial_layers=None,
    cross_view_layers=None,
    temporal_layers=None,
    num_heads=None,
    mlp_ratio=None,
    dropout=None,
    attn_dropout=None,
    ego_motion_dim=None,
    config=None,
    use_deepspeed_attention=True,
    **kwargs
):
    """
    Create and initialize the ModularVisionTransformer model optimized for DeepSpeed
    """
    # Get model configuration
    model_config = get_default_model_config()
    if config is not None and 'model' in config:
        # Update with provided config
        for key, value in config['model'].items():
            if key in model_config:
                model_config[key] = value
    
    # Use provided parameters if given, otherwise use config
    img_size = img_size if img_size is not None else model_config['img_size']
    patch_size = patch_size if patch_size is not None else model_config['patch_size']
    in_chans = in_chans if in_chans is not None else model_config['in_chans']
    embed_dim = embed_dim if embed_dim is not None else model_config['embed_dim']
    spatial_layers = spatial_layers if spatial_layers is not None else model_config['spatial_layers']
    cross_view_layers = cross_view_layers if cross_view_layers is not None else model_config['cross_view_layers']
    temporal_layers = temporal_layers if temporal_layers is not None else model_config['temporal_layers']
    num_heads = num_heads if num_heads is not None else model_config['num_heads']
    mlp_ratio = mlp_ratio if mlp_ratio is not None else model_config['mlp_ratio']
    dropout = dropout if dropout is not None else model_config['dropout']
    attn_dropout = attn_dropout if attn_dropout is not None else model_config['attn_dropout']
    ego_motion_dim = ego_motion_dim if ego_motion_dim is not None else model_config['ego_motion_dim']
    
    
    
    # Create model - use DeepSpeed version if specified
    model = DeepSpeedModularVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        spatial_layers=spatial_layers,
        cross_view_layers=cross_view_layers,
        temporal_layers=temporal_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        ego_motion_dim=ego_motion_dim,
        config=config
    )
    
    
    return model

def create_loss_function(
    reconstruction_weight=None,
    consistency_weight=None,
    future_weight=None,
    config=None,
):
    """
    Create the self-supervised loss function
    """
    # Get training configuration
    training_config = get_default_training_config()
    if config is not None and 'training' in config:
        # Update with provided config
        for key, value in config['training'].items():
            if key in training_config:
                training_config[key] = value
    
    # Use provided parameters if given, otherwise use config
    reconstruction_weight = reconstruction_weight if reconstruction_weight is not None else training_config['reconstruction_weight']
    consistency_weight = consistency_weight if consistency_weight is not None else training_config['consistency_weight']
    future_weight = future_weight if future_weight is not None else training_config['future_weight']
    
    return SelfSupervisedLoss(
        reconstruction_weight=reconstruction_weight,
        consistency_weight=consistency_weight,
        future_weight=future_weight
    )

def create_optimizer_for_deepspeed(model, lr=1e-4, weight_decay=1e-4, config=None):
    """
    Create optimizer compatible with DeepSpeed (will be handled by DeepSpeed engine)
    This function creates the optimizer parameters that DeepSpeed will use
    """
    # Get training configuration
    training_config = get_default_training_config()
    if config is not None and 'training' in config:
        # Update with provided config
        for key, value in config['training'].items():
            if key in training_config:
                training_config[key] = value
    
    # Get optimizer parameters
    beta1 = training_config.get('beta1', 0.9)
    beta2 = training_config.get('beta2', 0.999)
    eps = training_config.get('eps', 1e-8)
    
    # DeepSpeed will handle optimizer creation, but we can specify parameters
    optimizer_params = {
        "type": "AdamW",
        "params": {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": [beta1, beta2],
            "eps": eps
        }
    }
    
    return optimizer_params

def create_scheduler_for_deepspeed(optimizer_params, warmup_epochs=None, max_epochs=None, min_lr=None, config=None):
    """
    Create learning rate scheduler configuration for DeepSpeed
    """
    # Get training configuration
    training_config = get_default_training_config()
    if config is not None and 'training' in config:
        # Update with provided config
        for key, value in config['training'].items():
            if key in training_config:
                training_config[key] = value
    
    # Use provided parameters if given, otherwise use config
    warmup_epochs = warmup_epochs if warmup_epochs is not None else training_config['warmup_epochs']
    max_epochs = max_epochs if max_epochs is not None else training_config['epochs']
    min_lr = min_lr if min_lr is not None else training_config['min_lr']
    
    # DeepSpeed scheduler configuration
    scheduler_params = {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": min_lr,
            "warmup_max_lr": optimizer_params["params"]["lr"],
            "warmup_num_steps": warmup_epochs,
            "total_num_steps": max_epochs,
            "min_lr": min_lr
        }
    }
    
    return scheduler_params

def initialize_deepspeed_model(
    model, 
    config_dict, 
    model_parameters=None,
    lr=1e-4,
    weight_decay=1e-4,
    config=None
):
    """
    Initialize model with DeepSpeed engine
    """
    # Create optimizer parameters
    optimizer_params = create_optimizer_for_deepspeed(
        model, lr=lr, weight_decay=weight_decay, config=config
    )
    
    # Add optimizer to DeepSpeed config
    if "optimizer" not in config_dict:
        config_dict["optimizer"] = optimizer_params
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        config=config_dict,
        model_parameters=model_parameters or model.parameters(),
    )
    
    return model_engine, optimizer, lr_scheduler

def save_deepspeed_checkpoint(
    model_engine, 
    epoch, 
    loss, 
    save_dir, 
    model_config=None, 
    additional_data=None,
    config=None
):
    """
    Save DeepSpeed checkpoint
    """
    if model_config is None:
        # Get model configuration from defaults or config
        model_config = get_default_model_config()
        if config is not None and 'model' in config:
            for key, value in config['model'].items():
                if key in model_config:
                    model_config[key] = value
    
    # Use DeepSpeed's checkpoint saving
    checkpoint_data = {
        'epoch': epoch,
        'loss': loss,
        'model_config': model_config,
    }
    
    # Add any additional data to the checkpoint
    if additional_data:
        checkpoint_data.update(additional_data)
    
    # Save with DeepSpeed
    model_engine.save_checkpoint(save_dir, client_state=checkpoint_data)
    
    return save_dir

def load_deepspeed_checkpoint(
    model_engine, 
    checkpoint_dir, 
    load_optimizer_states=True,
    load_lr_scheduler_states=True
):
    """
    Load DeepSpeed checkpoint
    """
    _, client_state = model_engine.load_checkpoint(
        checkpoint_dir,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states
    )
    
    return client_state

def create_deepspeed_model_and_engine(
    img_size=None,
    patch_size=None,
    in_chans=None,
    embed_dim=None,
    spatial_layers=None,
    cross_view_layers=None,
    temporal_layers=None,
    num_heads=None,
    mlp_ratio=None,
    dropout=None,
    attn_dropout=None,
    ego_motion_dim=None,
    lr=1e-4,
    weight_decay=1e-4,
    config=None,
    **kwargs
):
    """
    Create model and initialize with DeepSpeed in one step
    """
    # Create the model
    model = create_modular_model(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        spatial_layers=spatial_layers,
        cross_view_layers=cross_view_layers,
        temporal_layers=temporal_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        ego_motion_dim=ego_motion_dim,
        config=config,
        **kwargs
    )
    
    # Count parameters BEFORE DeepSpeed wrapping (ZeRO-3 makes them inaccessible after)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Create DeepSpeed configuration
    deepspeed_config = create_deepspeed_config()
    
    # Initialize with DeepSpeed
    model_engine, optimizer, lr_scheduler = initialize_deepspeed_model(
        model=model,
        config_dict=deepspeed_config,
        lr=lr,
        weight_decay=weight_decay,
        config=config
    )
    
    return model_engine, optimizer, lr_scheduler, deepspeed_config, total_params 
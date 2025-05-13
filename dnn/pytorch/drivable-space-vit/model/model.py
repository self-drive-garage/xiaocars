import torch

from .stereo_transformer import StereoTransformer
from .self_supervised_loss import SelfSupervisedLoss
from .cosine_scheduler_with_warmup import CosineSchedulerWithWarmup

def get_default_model_config():
    """Return default model configuration"""
    return {
        'img_size': 224,
        'patch_size': 16,
        'num_channels': 3,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'ego_motion_dim': 12,
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

def create_model(
    img_size=None,
    patch_size=None,
    in_chans=None,
    embed_dim=None,
    depth=None,
    num_heads=None,
    mlp_ratio=None,
    dropout=None,
    attn_dropout=None,
    ego_motion_dim=None,
    config=None,
    **kwargs
):
    """
    Create and initialize the StereoTransformer model using either provided parameters or config
    """
    # Get model configuration
    model_config = get_default_model_config()
    if config is not None and 'model' in config:
        # Update with provided config
        for key, value in config['model'].items():
            model_config[key] = value
    
    # Use provided parameters if given, otherwise use config
    img_size = img_size if img_size is not None else model_config['img_size']
    patch_size = patch_size if patch_size is not None else model_config['patch_size']
    in_chans = in_chans if in_chans is not None else model_config['num_channels']
    embed_dim = embed_dim if embed_dim is not None else model_config['embed_dim']
    depth = depth if depth is not None else model_config['num_layers']
    num_heads = num_heads if num_heads is not None else model_config['num_heads']
    mlp_ratio = mlp_ratio if mlp_ratio is not None else model_config['mlp_ratio']
    dropout = dropout if dropout is not None else model_config['dropout']
    attn_dropout = attn_dropout if attn_dropout is not None else model_config['attn_dropout']
    ego_motion_dim = ego_motion_dim if ego_motion_dim is not None else model_config['ego_motion_dim']
    
    model = StereoTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        ego_motion_dim=ego_motion_dim,
        config=config,
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

def create_optimizer(model, lr=None, weight_decay=None, config=None):
    """
    Create optimizer for the model
    """
    # Get training configuration
    training_config = get_default_training_config()
    if config is not None and 'training' in config:
        # Update with provided config
        for key, value in config['training'].items():
            training_config[key] = value
    
    # Use provided parameters if given, otherwise use config
    lr = lr if lr is not None else training_config['lr']
    weight_decay = weight_decay if weight_decay is not None else training_config['weight_decay']
    
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

def create_scheduler(optimizer, warmup_epochs=None, max_epochs=None, min_lr=None, config=None):
    """
    Create learning rate scheduler with cosine decay and warmup
    """
    # Get training configuration
    training_config = get_default_training_config()
    if config is not None and 'training' in config:
        # Update with provided config
        for key, value in config['training'].items():
            training_config[key] = value
    
    # Use provided parameters if given, otherwise use config
    warmup_epochs = warmup_epochs if warmup_epochs is not None else training_config['warmup_epochs']
    max_epochs = max_epochs if max_epochs is not None else training_config['epochs']
    min_lr = min_lr if min_lr is not None else training_config['min_lr']
    
    return CosineSchedulerWithWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        min_lr=min_lr
    )

def load_model_from_checkpoint(checkpoint_path, device='cuda', config=None):
    """
    Load model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with the same configuration as saved
    model_config = checkpoint.get('model_config', {})
    model = create_model(**model_config, config=config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint

def save_model_checkpoint(
    model, optimizer, scheduler, epoch, loss, 
    save_path, model_config=None, additional_data=None, config=None
):
    """
    Save model checkpoint
    """
    if model_config is None:
        # Get model configuration from defaults or config
        model_config = get_default_model_config()
        if config is not None and 'model' in config:
            for key, value in config['model'].items():
                model_config[key] = value
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'model_config': model_config,
    }
    
    # Add any additional data to the checkpoint
    if additional_data:
        checkpoint.update(additional_data)
    
    torch.save(checkpoint, save_path)
    
    return save_path

import torch
import torch.nn as nn

from .stereo_transformer import StereoTransformer
from .self_supervised_loss import SelfSupervisedLoss
from .driving_dataset import DrivingDataset, create_dataloader
from .cosine_scheduler_with_warmup import CosineSchedulerWithWarmup

# Constants for model architecture
IMG_SIZE = 224  # Input image size for ViT
PATCH_SIZE = 16  # Patch size for ViT
NUM_CHANNELS = 3  # RGB images
EMBED_DIM = 768  # Embedding dimension
NUM_HEADS = 12  # Number of attention heads
NUM_LAYERS = 12  # Number of transformer layers
MLP_RATIO = 4  # Expansion ratio for MLP
DROPOUT = 0.1  # Dropout probability
EGO_MOTION_DIM = 6  # Ego motion dimensions (speed, acceleration, steering, etc.)

def create_model(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=NUM_CHANNELS,
    embed_dim=EMBED_DIM,
    depth=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    attn_dropout=DROPOUT,
    ego_motion_dim=EGO_MOTION_DIM,
    **kwargs
):
    """
    Create and initialize the StereoTransformer model
    """
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
    )
    
    return model

def create_loss_function(reconstruction_weight=1.0, consistency_weight=1.0, future_weight=0.5):
    """
    Create the self-supervised loss function
    """
    return SelfSupervisedLoss(
        reconstruction_weight=reconstruction_weight,
        consistency_weight=consistency_weight,
        future_weight=future_weight
    )

def create_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """
    Create optimizer for the model
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

def create_scheduler(optimizer, warmup_epochs=10, max_epochs=100, min_lr=1e-6):
    """
    Create learning rate scheduler with cosine decay and warmup
    """
    return CosineSchedulerWithWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        min_lr=min_lr
    )

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with the same configuration as saved
    model_config = checkpoint.get('model_config', {})
    model = create_model(**model_config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint

def save_model_checkpoint(
    model, optimizer, scheduler, epoch, loss, 
    save_path, model_config=None, additional_data=None
):
    """
    Save model checkpoint
    """
    if model_config is None:
        model_config = {
            'img_size': IMG_SIZE,
            'patch_size': PATCH_SIZE,
            'in_chans': NUM_CHANNELS,
            'embed_dim': EMBED_DIM,
            'depth': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'mlp_ratio': MLP_RATIO,
            'dropout': DROPOUT,
            'attn_dropout': DROPOUT,
            'ego_motion_dim': EGO_MOTION_DIM,
        }
    
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

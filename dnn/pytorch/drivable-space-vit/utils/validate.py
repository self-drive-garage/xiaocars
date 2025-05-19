import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def validate(model, loader, loss_fn, device, epoch, config, rank=0):
    """Validate the model using DeepSpeed"""
    model.eval()
    val_loss = 0.0
    total_steps = len(loader)
    
    # Use tqdm for progress bar (only on main process)
    if rank == 0:
        pbar = tqdm(loader, total=total_steps, desc=f"Validation {epoch+1}")
    else:
        pbar = loader
    
    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with task='all' to generate all outputs needed for loss calculation
            outputs = model(batch, task='all')
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Update metrics
            val_loss += loss.item()
            
            # Update progress bar (only on main process)
            if rank == 0:
                pbar.set_postfix({'loss': loss.item()})
    
    # Average validation loss
    val_loss = val_loss / total_steps
    
    # Log validation results (only on main process)
    if rank == 0:
        logger.info(f"Validation Epoch: [{epoch+1}/{config['training']['epochs']}] Loss: {val_loss:.4f}")
        
        # Log individual loss components
        for loss_name, loss_value in loss_dict.items():
            logger.info(f"  Validation {loss_name}: {loss_value:.4f}")
    
    return val_loss


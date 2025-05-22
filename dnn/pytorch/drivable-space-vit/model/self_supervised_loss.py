import torch
import torch.nn as nn
import logging

# Configure logger
logger = logging.getLogger(__name__)

class SelfSupervisedLoss(nn.Module):
    """Combined loss for self-supervised training"""
    def __init__(self, reconstruction_weight=0.00001, consistency_weight=0.00001, future_weight=0.000001):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.future_weight = future_weight
        
        self.reconstruction_loss = nn.MSELoss(reduction='mean')
        self.consistency_loss = nn.CosineSimilarity(dim=1)
        self.future_prediction_loss = nn.MSELoss()
        
        # Add a projection layer to handle dimension mismatch in future prediction
        self.future_projection = None
        
        # Log initialization params
        logger.debug(f"SelfSupervisedLoss::__init__ - Initialized with weights: reconstruction={reconstruction_weight}, consistency={consistency_weight}, future={future_weight}")
        logger.debug(f"ACTUAL WEIGHTS BEING USED: recon={self.reconstruction_weight}, consistency={self.consistency_weight}, future={self.future_weight}")
    
    def forward(self, outputs, batch):
        # Log available keys in outputs
        logger.debug(f"SelfSupervisedLoss::forward - Available keys in outputs: {list(outputs.keys())}")
        logger.debug(f"SelfSupervisedLoss::forward - Available keys in batch: {list(batch.keys())}")
        
        loss_dict = {}
        
        # We'll accumulate real losses here
        total_loss = 0
            
        # Get the last frame from the sequence for reconstruction target
        left_target = batch['left_images'][:, -1]  # (B, C, H, W)
        center_target = batch['center_images'][:, -1]  # (B, C, H, W)
        right_target = batch['right_images'][:, -1]  # (B, C, H, W)

        # Calculate consistency between all pairs of reconstructions
        left_recon = outputs['left_reconstructed']
        center_recon = outputs['center_reconstructed']
        right_recon = outputs['right_reconstructed']
        
        # Then use these normalized values
        left_recon_loss = self.reconstruction_loss(left_recon, left_target)
        center_recon_loss = self.reconstruction_loss(center_recon, center_target)
        right_recon_loss = self.reconstruction_loss(right_recon, right_target)
        
        # Log individual reconstruction losses
        logger.debug(f"SelfSupervisedLoss::forward - Individual reconstruction losses - left: {left_recon_loss.item()}, center: {center_recon_loss.item()}, right: {right_recon_loss.item()}")
        
        # Average the reconstruction loss across all three views
        recon_loss = (left_recon_loss + center_recon_loss + right_recon_loss) / 3.0
        total_loss += self.reconstruction_weight * recon_loss
        loss_dict['reconstruction_loss'] = recon_loss.item()

        # Calculate pairwise cosine similarities and convert to loss (1 - similarity)
        left_center_sim = self.consistency_loss(
            left_recon.reshape(left_recon.size(0), -1),
            center_recon.reshape(center_recon.size(0), -1)
        ).mean()
        
        center_right_sim = self.consistency_loss(
            center_recon.reshape(center_recon.size(0), -1),
            right_recon.reshape(right_recon.size(0), -1)
        ).mean()
        
        left_right_sim = self.consistency_loss(
            left_recon.reshape(left_recon.size(0), -1),
            right_recon.reshape(right_recon.size(0), -1)
        ).mean()
        
        left_center_consistency = 1.0 - left_center_sim
        center_right_consistency = 1.0 - center_right_sim
        left_right_consistency = 1.0 - left_right_sim
        
        # Average the consistency losses
        consistency = (left_center_consistency + center_right_consistency + left_right_consistency) / 3.0
        total_loss += self.consistency_weight * consistency            
        loss_dict['consistency_loss'] = consistency.item()

        # Future prediction loss 
        # future_pred = outputs['future_prediction']  # Expected shape [B, 3*embed_dim]
        # future_target = batch['future_features']    # Target future features
        
        # # Now compute the loss with matching dimensions
        # future_loss = self.future_prediction_loss(future_pred, future_target)
        # total_loss += self.future_weight * future_loss
        # loss_dict['future_prediction_loss'] = future_loss.item()

        logger.debug(f"RAW LOSSES: recon={recon_loss.item()}, consistency={consistency.item()}")

       
        # Record total loss for logging
        if isinstance(total_loss, torch.Tensor):
            loss_dict['total_loss'] = total_loss.item()
            logger.debug(f"SelfSupervisedLoss::forward - Final total loss: {total_loss.item()}")
            # Check if the loss is zero or extremely small
            if abs(total_loss.item()) < 1e-8:
                logger.debug("SelfSupervisedLoss::forward - WARNING: Total loss is effectively zero!")
        else:
            loss_dict['total_loss'] = float(total_loss) if total_loss is not None else 0.0
            logger.debug(f"SelfSupervisedLoss::forward - Final total loss (non-tensor): {loss_dict['total_loss']}")
            
        return total_loss, loss_dict
    
    def _get_device(self, outputs):
        """Helper method to get the device from outputs"""
        return torch.device('cuda')

import torch
import torch.nn as nn

class SelfSupervisedLoss(nn.Module):
    """Combined loss for self-supervised training"""
    def __init__(self, reconstruction_weight=1.0, consistency_weight=1.0, future_weight=0.5):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.future_weight = future_weight
        
        self.reconstruction_loss = nn.MSELoss()
        self.consistency_loss = nn.CosineSimilarity(dim=1)
        self.future_prediction_loss = nn.MSELoss()
    
    def forward(self, outputs, batch):
        loss = 0.0
        loss_dict = {}
        
        # Reconstruction loss (if available)
        if 'left_reconstructed' in outputs and 'right_reconstructed' in outputs:
            # Get the last frame from the sequence for reconstruction target
            left_target = batch['left_images'][:, -1]  # (B, C, H, W)
            right_target = batch['right_images'][:, -1]  # (B, C, H, W)
            
            left_recon_loss = self.reconstruction_loss(outputs['left_reconstructed'], left_target)
            right_recon_loss = self.reconstruction_loss(outputs['right_reconstructed'], right_target)
            recon_loss = (left_recon_loss + right_recon_loss) / 2.0
            
            loss += self.reconstruction_weight * recon_loss
            loss_dict['reconstruction_loss'] = recon_loss.item()
        
        # View consistency loss (if available)
        if 'left_reconstructed' in outputs and 'right_reconstructed' in outputs:
            # Calculate consistency between left and right reconstructions
            # This encourages the model to learn stereo correspondence
            left_recon = outputs['left_reconstructed']
            right_recon = outputs['right_reconstructed']
            
            # Calculate cosine similarity and convert to a loss (1 - similarity)
            consistency = 1.0 - self.consistency_loss(
                left_recon.reshape(left_recon.size(0), -1),
                right_recon.reshape(right_recon.size(0), -1)
            ).mean()
            
            loss += self.consistency_weight * consistency
            loss_dict['consistency_loss'] = consistency.item()
        
        # Future prediction loss (if available)
        if 'future_prediction' in outputs and batch.get('future_features') is not None:
            future_loss = self.future_prediction_loss(
                outputs['future_prediction'],
                batch['future_features']
            )
            
            loss += self.future_weight * future_loss
            loss_dict['future_prediction_loss'] = future_loss.item()
        
        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict

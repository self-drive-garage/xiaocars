import torch
import torch.nn as nn

class SelfSupervisedLoss(nn.Module):
    """Combined loss for self-supervised training"""
    def __init__(self, reconstruction_weight=1.0, consistency_weight=1.0, future_weight=0.005):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.future_weight = future_weight
        
        self.reconstruction_loss = nn.MSELoss()
        self.consistency_loss = nn.CosineSimilarity(dim=1)
        self.future_prediction_loss = nn.MSELoss()
        
        # Add a projection layer to handle dimension mismatch in future prediction
        self.future_projection = None
    
    def forward(self, outputs, batch):
        # Track if we've added any loss components
        has_loss_components = False
        loss_dict = {}
        
        # We'll accumulate real losses here
        total_loss = None
        
        # Reconstruction loss (if available)
        if ('left_reconstructed' in outputs and 
            'center_reconstructed' in outputs and 
            'right_reconstructed' in outputs):
            
            # Get the last frame from the sequence for reconstruction target
            left_target = batch['left_images'][:, -1]  # (B, C, H, W)
            center_target = batch['center_images'][:, -1]  # (B, C, H, W)
            right_target = batch['right_images'][:, -1]  # (B, C, H, W)
            
            left_recon_loss = self.reconstruction_loss(outputs['left_reconstructed'], left_target)
            center_recon_loss = self.reconstruction_loss(outputs['center_reconstructed'], center_target)
            right_recon_loss = self.reconstruction_loss(outputs['right_reconstructed'], right_target)
            
            # Average the reconstruction loss across all three views
            recon_loss = (left_recon_loss + center_recon_loss + right_recon_loss) / 3.0
            
            if total_loss is None:
                total_loss = self.reconstruction_weight * recon_loss
            else:
                total_loss += self.reconstruction_weight * recon_loss
                
            has_loss_components = True
            loss_dict['reconstruction_loss'] = recon_loss.item()
        
        # View consistency loss (if available)
        if ('left_reconstructed' in outputs and 
            'center_reconstructed' in outputs and 
            'right_reconstructed' in outputs):
            
            # Calculate consistency between all pairs of reconstructions
            left_recon = outputs['left_reconstructed']
            center_recon = outputs['center_reconstructed']
            right_recon = outputs['right_reconstructed']
            
            # Calculate pairwise cosine similarities and convert to loss (1 - similarity)
            left_center_consistency = 1.0 - self.consistency_loss(
                left_recon.reshape(left_recon.size(0), -1),
                center_recon.reshape(center_recon.size(0), -1)
            ).mean()
            
            center_right_consistency = 1.0 - self.consistency_loss(
                center_recon.reshape(center_recon.size(0), -1),
                right_recon.reshape(right_recon.size(0), -1)
            ).mean()
            
            left_right_consistency = 1.0 - self.consistency_loss(
                left_recon.reshape(left_recon.size(0), -1),
                right_recon.reshape(right_recon.size(0), -1)
            ).mean()
            
            # Average the consistency losses
            consistency = (left_center_consistency + center_right_consistency + left_right_consistency) / 3.0
            
            if total_loss is None:
                total_loss = self.consistency_weight * consistency
            else:
                total_loss += self.consistency_weight * consistency
                
            has_loss_components = True
            loss_dict['consistency_loss'] = consistency.item()
        
        # Future prediction loss (if available)
        if 'future_prediction' in outputs and batch.get('future_features') is not None:
            future_pred = outputs['future_prediction']
            future_target = batch['future_features']
            
            # Handle dimension mismatch - initialize projection layer if needed
            if self.future_projection is None and future_pred.size(1) != future_target.size(1):
                input_dim = future_pred.size(1)
                output_dim = future_target.size(1)
                self.future_projection = nn.Linear(input_dim, output_dim).to(future_pred.device)
                # Initialize projection with small weights for stability
                nn.init.xavier_uniform_(self.future_projection.weight, gain=0.01)
                nn.init.zeros_(self.future_projection.bias)
                print(f"Created projection layer from {input_dim} to {output_dim} features")
            
            # Apply projection if needed
            if self.future_projection is not None:
                future_pred = self.future_projection(future_pred)
            
            # Now compute the loss with matching dimensions
            future_loss = self.future_prediction_loss(future_pred, future_target)
            
            if total_loss is None:
                total_loss = self.future_weight * future_loss
            else:
                total_loss += self.future_weight * future_loss
                
            has_loss_components = True
            loss_dict['future_prediction_loss'] = future_loss.item()
        
        # If no loss components were added, create a dummy loss with gradients
        if not has_loss_components:
            # Create a dummy loss that's connected to the computation graph
            # Get any tensor from outputs to create a proper loss with gradients
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    # Create a zero loss that's connected to the graph
                    total_loss = 0.0 * torch.sum(value)
                    break
            
            # If we still don't have a valid loss (no gradable tensors found)
            if total_loss is None:
                print("WARNING: No loss components found and no gradable tensors in outputs. Training may fail.")
                # Last resort: create a new parameter and use it for loss
                dummy_param = nn.Parameter(torch.tensor([0.0], device=self._get_device(outputs)))
                total_loss = 0.0 * dummy_param.sum()
        
        # Record total loss for logging
        if isinstance(total_loss, torch.Tensor):
            loss_dict['total_loss'] = total_loss.item()
        else:
            loss_dict['total_loss'] = float(total_loss) if total_loss is not None else 0.0
            
        return total_loss, loss_dict
    
    def _get_device(self, outputs):
        """Helper method to get the device from outputs"""
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math

class CosineSchedulerWithWarmup:
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    Ensures proper transitions between warmup and decay phases.
    """
    def __init__(self, optimizer, warmup_epochs=10, max_epochs=100, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = optimizer.param_groups[0]['lr']  # Initial learning rate
        self.min_lr = min_lr
        self.last_epoch = -1  # Initialize to -1 to ensure proper first step
        
        # Store original learning rates for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None):
        """
        Update learning rate based on epoch.
        Args:
            epoch: Current epoch number (0-indexed)
        """
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = epoch
        
        # Calculate new learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            # Linear warmup phase
            if epoch < self.warmup_epochs:
                # Linear warmup from 0 to base_lr
                lr = base_lr * (epoch / self.warmup_epochs)
            else:
                # Cosine annealing from base_lr to min_lr
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                # Ensure progress is capped at 1.0 to prevent negative values in cosine
                progress = min(1.0, progress)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine_factor
            
            # Update learning rate in optimizer
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
        }
    
    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']

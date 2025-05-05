import math

class CosineSchedulerWithWarmup:
    """Cosine learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr_scale = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(self.min_lr, lr_scale * self.base_lrs[i])
        
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict['base_lrs']

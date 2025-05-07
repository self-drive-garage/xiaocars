import torch
import torch.nn as nn

def get_default_config():
    """Return default configuration parameters for EgoMotionEncoder"""
    return {
        'ego_motion_dim': 6,
        'embed_dim': 768,
        'dropout': 0.1,
    }

class EgoMotionEncoder(nn.Module):
    """Encoder for ego motion data"""
    def __init__(
        self,
        ego_motion_dim=None,
        embed_dim=None,
        dropout=None,
        config=None,
    ):
        super().__init__()
        
        # Get default configuration
        default_config = get_default_config()
        
        # Use provided config if available
        model_config = {}
        if config is not None and 'model' in config:
            model_config = config['model']
        
        # Use provided parameters if given, otherwise use config, fallback to defaults
        self.ego_motion_dim = ego_motion_dim if ego_motion_dim is not None else model_config.get('ego_motion_dim', default_config['ego_motion_dim'])
        self.embed_dim = embed_dim if embed_dim is not None else model_config.get('embed_dim', default_config['embed_dim'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        
        self.encoder = nn.Sequential(
            nn.Linear(self.ego_motion_dim, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.Dropout(dropout_value),
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, ego_motion_dim)
        return self.encoder(x)  # (batch_size, seq_len, embed_dim)

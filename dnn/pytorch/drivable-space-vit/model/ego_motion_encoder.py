import torch
import torch.nn as nn

# Constants for model architecture
EGO_MOTION_DIM = 6  # Ego motion dimensions (speed, acceleration, steering, etc.)
EMBED_DIM = 768  # Embedding dimension
DROPOUT = 0.1  # Dropout probability

class EgoMotionEncoder(nn.Module):
    """Encoder for ego motion data"""
    def __init__(
        self,
        ego_motion_dim=EGO_MOTION_DIM,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.ego_motion_dim = ego_motion_dim
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(ego_motion_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, ego_motion_dim)
        return self.encoder(x)  # (batch_size, seq_len, embed_dim)

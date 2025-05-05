import torch
import torch.nn as nn

# Constants for model architecture
MLP_RATIO = 4  # Expansion ratio for MLP
DROPOUT = 0.1  # Dropout probability

class TemporalTransformerLayer(nn.Module):
    """Transformer layer for temporal attention across frames"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=DROPOUT,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Temporal self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

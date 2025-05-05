import torch
import torch.nn as nn

# Constants for model architecture
MLP_RATIO = 4  # Expansion ratio for MLP
DROPOUT = 0.1  # Dropout probability

class CrossViewTransformerLayer(nn.Module):
    """Transformer layer for cross-view attention between left and right images"""
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
        self.norm1_left = nn.LayerNorm(dim)
        self.norm1_right = nn.LayerNorm(dim)
        self.norm2_left = nn.LayerNorm(dim)
        self.norm2_right = nn.LayerNorm(dim)
        
        # Cross-attention: left to right and right to left
        self.cross_attn_left_to_right = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        self.cross_attn_right_to_left = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        # MLP blocks for both paths
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_left = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        self.mlp_right = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, left_features, right_features):
        # Cross-attention: left to right
        residual_right = right_features
        left_norm = self.norm1_left(left_features)
        right_norm = self.norm1_right(right_features)
        right_attended, _ = self.cross_attn_left_to_right(
            right_norm, left_norm, left_norm, need_weights=False
        )
        right_features = residual_right + right_attended
        
        # Cross-attention: right to left
        residual_left = left_features
        left_norm = self.norm1_left(left_features)
        right_norm = self.norm1_right(right_features)
        left_attended, _ = self.cross_attn_right_to_left(
            left_norm, right_norm, right_norm, need_weights=False
        )
        left_features = residual_left + left_attended
        
        # MLP for left path
        residual_left = left_features
        left_features = self.norm2_left(left_features)
        left_features = self.mlp_left(left_features)
        left_features = residual_left + left_features
        
        # MLP for right path
        residual_right = right_features
        right_features = self.norm2_right(right_features)
        right_features = self.mlp_right(right_features)
        right_features = residual_right + right_features
        
        return left_features, right_features

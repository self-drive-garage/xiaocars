import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionGuidedFuturePredictor(nn.Module):
    """
    Enhanced future prediction module that uses ego motion to guide prediction of future frames
    """
    def __init__(
        self,
        embed_dim,
        ego_motion_dim=12,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ego_motion_dim = ego_motion_dim
        
        # Transform ego motion to feature space
        self.motion_transform = nn.Sequential(
            nn.Linear(ego_motion_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Self-attention for temporal encoding
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention to inject motion information
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Final projection
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, features, ego_motion):
        """
        Predict future features based on current features and ego motion
        
        Args:
            features: Tensor of shape (B, T, E) - sequence of features
            ego_motion: Tensor of shape (B, T, D) - sequence of ego motion data
            
        Returns:
            Tensor of shape (B, E) - predicted future features
        """
        # Get batch size and sequence length
        B, T, E = features.shape
        
        # Transform ego motion to feature space
        motion_features = self.motion_transform(ego_motion)  # (B, T, E)
        
        # Self-attention on visual features (with residual)
        attn_output, _ = self.self_attention(
            query=features,
            key=features,
            value=features
        )
        features = features + attn_output
        features = self.norm1(features)
        
        # Cross-attention with motion features (with residual)
        attn_output, _ = self.cross_attention(
            query=features,
            key=motion_features,
            value=motion_features
        )
        features = features + attn_output
        features = self.norm2(features)
        
        # Feed-forward network (with residual)
        ffn_output = self.ffn(features)
        features = features + ffn_output
        features = self.norm3(features)
        
        # Take the last time step as future prediction
        future_features = features[:, -1]  # (B, E)
        
        # Project to output space
        future_features = self.projection(future_features)  # (B, E)
        
        return future_features

class MotionSpatialTransformer(nn.Module):
    """
    Transformer block that uses ego motion to modulate self-attention
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        # Motion modulation
        self.motion_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
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
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, motion_features=None):
        """
        Forward pass with optional motion features for attention modulation
        
        Args:
            x: Tensor of shape (B, N, D) - visual features
            motion_features: Tensor of shape (B, D) - motion features
            
        Returns:
            Tensor of shape (B, N, D) - updated features
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        
        # Apply motion modulation if available
        if motion_features is not None:
            # Expand motion features to match spatial dimensions
            motion_gate = self.motion_gate(motion_features).unsqueeze(1)  # (B, 1, D)
            attn_output = attn_output * motion_gate
            
        # Add residual connection
        x = x + attn_output
        
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x 
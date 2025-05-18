import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionGuidedFuturePredictor(nn.Module):
    """
    Memory-efficient future prediction module that uses ego motion to guide prediction of future frames
    """
    def __init__(
        self,
        embed_dim,
        ego_motion_dim=9,  # Updated from 15 to 9 (3 velocity + 3 acceleration + 3 angular velocity)
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        
        # Reduce internal dimensions to save memory
        self.embed_dim = embed_dim
        self.ego_motion_dim = ego_motion_dim
        self.internal_dim = embed_dim // 2  # Use half the dimension for internal processing
        
        # Transform ego motion to feature space (with reduced dimensions)
        self.motion_transform = nn.Sequential(
            nn.Linear(ego_motion_dim, self.internal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Reduce input features dimension
        self.input_reduction = nn.Linear(embed_dim, self.internal_dim)
        
        # Simpler attention mechanism with fewer heads and smaller dimensions
        num_reduced_heads = max(1, num_heads // 2)  # At least 1 head
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.internal_dim,
            num_heads=num_reduced_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network (smaller expansion ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.internal_dim, self.internal_dim * 2),  # Reduced expansion ratio
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.internal_dim * 2, self.internal_dim),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.internal_dim)
        self.norm2 = nn.LayerNorm(self.internal_dim)
        
        # Final projection back to original dimension
        self.projection = nn.Linear(self.internal_dim, embed_dim)
    
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
        
        # Reduce feature dimensions
        features = self.input_reduction(features)  # (B, T, E/2)
        
        # Transform ego motion to feature space
        motion_features = self.motion_transform(ego_motion)  # (B, T, E/2)
        
        # Combine features and motion instead of using expensive cross-attention
        combined_features = features + motion_features
        combined_features = self.norm1(combined_features)
        
        # Self-attention on combined features
        attn_output, _ = self.self_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features
        )
        combined_features = combined_features + attn_output
        combined_features = self.norm2(combined_features)
        
        # Feed-forward network
        combined_features = combined_features + self.ffn(combined_features)
        
        # Take the last time step as future prediction
        future_features = combined_features[:, -1]  # (B, E/2)
        
        # Project back to original dimension
        future_features = self.projection(future_features)  # (B, E)
        
        return future_features

class MotionSpatialTransformer(nn.Module):
    """
    Memory-efficient transformer block that uses ego motion to modulate self-attention
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
        
        # Use fewer heads for attention
        num_reduced_heads = max(1, num_heads // 2)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_reduced_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        # Motion modulation (simplified)
        self.motion_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # MLP block with reduced expansion ratio
        mlp_hidden_dim = int(dim * (mlp_ratio / 2))  # Half the expansion ratio
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
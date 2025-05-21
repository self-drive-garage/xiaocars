import torch.nn as nn

def get_default_config():
    """Return default configuration parameters for CrossViewTransformerLayer"""
    return {
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
    }

class CrossViewTransformerLayer(nn.Module):
    """Transformer layer for cross-view attention between left, center, and right images"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=None,
        dropout=None,
        attn_dropout=None,
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
        self.dim = dim
        self.num_heads = num_heads
        mlp_ratio_value = mlp_ratio if mlp_ratio is not None else model_config.get('mlp_ratio', default_config['mlp_ratio'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        attn_dropout_value = attn_dropout if attn_dropout is not None else model_config.get('attn_dropout', default_config['attn_dropout'])
        
        # Normalization layers for all three views
        self.norm1_left = nn.LayerNorm(dim)
        self.norm1_center = nn.LayerNorm(dim)
        self.norm1_right = nn.LayerNorm(dim)
        self.norm2_left = nn.LayerNorm(dim)
        self.norm2_center = nn.LayerNorm(dim)
        self.norm2_right = nn.LayerNorm(dim)
        
        # Cross-attention between all three views (six directions)
        # Left to Center
        self.cross_attn_left_to_center = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # Left to Right
        self.cross_attn_left_to_right = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # Center to Left
        self.cross_attn_center_to_left = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # Center to Right
        self.cross_attn_center_to_right = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # Right to Left
        self.cross_attn_right_to_left = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # Right to Center
        self.cross_attn_right_to_center = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # MLP blocks for all three views
        mlp_hidden_dim = int(dim * mlp_ratio_value)
        
        self.mlp_left = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_value),
        )
        
        self.mlp_center = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_value),
        )
        
        self.mlp_right = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_value),
        )
    
    def forward(self, left_features, center_features, right_features):
        # Apply cross-attention between all three views
        
        # Normalize inputs for attention
        left_norm = self.norm1_left(left_features)
        center_norm = self.norm1_center(center_features)
        right_norm = self.norm1_right(right_features)
        
        # --- Left features attending to others ---
        # Left attending to center
        residual_center = center_features
        center_attended_by_left, _ = self.cross_attn_left_to_center(
            center_norm, left_norm, left_norm, need_weights=False
        )
        center_features = residual_center + center_attended_by_left
        
        # Left attending to right
        residual_right = right_features
        right_attended_by_left, _ = self.cross_attn_left_to_right(
            right_norm, left_norm, left_norm, need_weights=False
        )
        right_features = residual_right + right_attended_by_left
        
        # --- Center features attending to others ---
        # Center attending to left
        residual_left = left_features
        left_attended_by_center, _ = self.cross_attn_center_to_left(
            left_norm, center_norm, center_norm, need_weights=False
        )
        left_features = residual_left + left_attended_by_center
        
        # Center attending to right
        residual_right = right_features
        right_attended_by_center, _ = self.cross_attn_center_to_right(
            right_norm, center_norm, center_norm, need_weights=False
        )
        right_features = residual_right + right_attended_by_center
        
        # --- Right features attending to others ---
        # Right attending to left
        residual_left = left_features
        left_attended_by_right, _ = self.cross_attn_right_to_left(
            left_norm, right_norm, right_norm, need_weights=False
        )
        left_features = residual_left + left_attended_by_right
        
        # Right attending to center
        residual_center = center_features
        center_attended_by_right, _ = self.cross_attn_right_to_center(
            center_norm, right_norm, right_norm, need_weights=False
        )
        center_features = residual_center + center_attended_by_right
        
        # Renormalize for MLP
        left_norm = self.norm2_left(left_features)
        center_norm = self.norm2_center(center_features) 
        right_norm = self.norm2_right(right_features)
        
        # Apply MLP to each view
        # MLP for left path
        residual_left = left_features
        left_mlp_out = self.mlp_left(left_norm)
        left_features = residual_left + left_mlp_out
        
        # MLP for center path
        residual_center = center_features
        center_mlp_out = self.mlp_center(center_norm)
        center_features = residual_center + center_mlp_out
        
        # MLP for right path
        residual_right = right_features
        right_mlp_out = self.mlp_right(right_norm)
        right_features = residual_right + right_mlp_out
        
        return left_features, center_features, right_features

import torch.nn as nn

def get_default_config():
    """Return default configuration parameters for CrossViewTransformerLayer"""
    return {
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
    }

class CrossViewTransformerLayer(nn.Module):
    """Transformer layer for cross-view attention between left and right images"""
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
        
        # Normalization layers
        self.norm1_left = nn.LayerNorm(dim)
        self.norm1_right = nn.LayerNorm(dim)
        self.norm2_left = nn.LayerNorm(dim)
        self.norm2_right = nn.LayerNorm(dim)
        
        # Cross-attention: left to right and right to left
        self.cross_attn_left_to_right = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        self.cross_attn_right_to_left = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # MLP blocks for both paths
        mlp_hidden_dim = int(dim * mlp_ratio_value)
        self.mlp_left = nn.Sequential(
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

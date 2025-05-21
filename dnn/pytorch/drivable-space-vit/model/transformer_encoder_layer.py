import torch
import torch.nn as nn
from utils.train_utils import debug_attention_forward

def get_default_config():
    """Return default configuration parameters for TransformerEncoderLayer"""
    return {
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
    }

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head self-attention and MLP"""
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
        training_config = {}
        if config is not None:
            if 'model' in config:
                model_config = config['model']
            if 'training' in config:
                training_config = config['training']
        
        # Store debug flag
        self.debug_mode = training_config.get('debug', False)
        
        # Use provided parameters if given, otherwise use config, fallback to defaults
        self.dim = dim
        self.num_heads = num_heads
        mlp_ratio_value = mlp_ratio if mlp_ratio is not None else model_config.get('mlp_ratio', default_config['mlp_ratio'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        attn_dropout_value = attn_dropout if attn_dropout is not None else model_config.get('attn_dropout', default_config['attn_dropout'])
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True,
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio_value)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_value),
        )
    
    def forward(self, x, attn_mask=None):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # Use debug attention forward function if debug mode is enabled, otherwise use standard forward
        if self.debug_mode:
            # Use the utility function for debugging
            attn_out, _ = debug_attention_forward(self.attn, x, attn_mask=attn_mask, need_weights=False)
            x = attn_out
        else:
            # Standard forward pass
            x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

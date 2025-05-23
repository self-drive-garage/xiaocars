import torch
import torch.nn as nn
import logging

from .ds_multihead_attention import DeepSpeedCompatibleMultiheadAttention

# Get logger
logger = logging.getLogger(__name__)

def get_default_config():
    """Return default configuration parameters for TransformerEncoderLayer"""
    return {
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
    }

class DeepSpeedTransformerEncoderLayer(nn.Module):
    """DeepSpeed-compatible transformer encoder layer with multi-head self-attention and MLP"""
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
        if config and 'model' in config:
            model_config = config['model']
        else:
            model_config = {}
        
        self.dim = dim
        self.num_heads = num_heads

        mlp_ratio_value = mlp_ratio if mlp_ratio is not None else model_config.get('mlp_ratio', default_config['mlp_ratio'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        attn_dropout_value = attn_dropout if attn_dropout is not None else model_config.get('attn_dropout', default_config['attn_dropout'])
        
        # Normalization layers (Pre-norm architecture for better training stability)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention - use DeepSpeed compatible version
        self.attn = DeepSpeedCompatibleMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout_value,
            batch_first=True
        )
        
        # MLP block with GELU activation
        mlp_hidden_dim = int(dim * mlp_ratio_value)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_value),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        # Initialize MLP layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize layer norms
        for module in [self.norm1, self.norm2]:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass with pre-norm architecture
        Args:
            x: Input tensor [batch, seq_len, dim]
            attn_mask: Optional attention mask
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Self-attention with residual connection (Pre-norm)
        residual = x
        x = self.norm1(x)
        
        # Add shape diagnostics
        logger.debug(f"DeepSpeedTransformerEncoderLayer input x shape: {x.shape}, dtype: {x.dtype}")
        logger.debug(f"DeepSpeedTransformerEncoderLayer x dims: {len(x.shape)}, is_contiguous: {x.is_contiguous()}")
        
        # Additional diagnostics for attention params
        logger.debug(f"Attention params: dim={self.dim}, num_heads={self.num_heads}")
        logger.debug(f"Attention q_proj.weight shape: {self.attn.q_proj.weight.shape}")
        logger.debug(f"Attention out_proj.weight shape: {self.attn.out_proj.weight.shape}")
        
        try:
            # Log shape right before attention call
            logger.debug(f"About to call DeepSpeed attention with input shape: {x.shape}")
            
            x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
            logger.debug(f"DeepSpeed attention output shape: {x.shape}")
        except Exception as e:
            # Log detailed diagnostics on error
            logger.error(f"Error in DeepSpeed attention forward: {str(e)}")
            logger.error(f"Input tensor shape: {x.shape}, ndim: {x.ndim}, dtype: {x.dtype}")
            if hasattr(self.attn, 'out_proj'):
                logger.error(f"out_proj.weight: {self.attn.out_proj.weight.shape}, dtype: {self.attn.out_proj.weight.dtype}")
            raise
        
        x = residual + x
        
        # MLP with residual connection (Pre-norm)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# Alias for backward compatibility
DSTransformerEncoderLayer = DeepSpeedTransformerEncoderLayer 
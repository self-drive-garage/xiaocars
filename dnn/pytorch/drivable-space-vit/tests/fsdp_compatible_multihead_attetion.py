import torch
import torch.nn as nn
import logging

# Get logger
logger = logging.getLogger(__name__)

def get_default_config():
    """Return default configuration parameters for TransformerEncoderLayer"""
    return {
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
    }

class FSDPCompatibleMultiheadAttention(nn.Module):
    """FSDP-compatible wrapper for MultiheadAttention to avoid weight flattening issues"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Separate linear layers instead of using the fused in_proj_weight
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        
        # Initialize weights similar to PyTorch's MultiheadAttention
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters similar to PyTorch's MultiheadAttention"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """Forward pass compatible with PyTorch's MultiheadAttention interface"""
        if not self.batch_first:
            # Convert from seq_len, batch, embed_dim to batch, seq_len, embed_dim
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch, seq_len, embed_dim]
        K = self.k_proj(key)    # [batch, seq_len, embed_dim]
        V = self.v_proj(value)  # [batch, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores += attn_mask
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Reshape back to [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final output projection
        attn_output = self.out_proj(attn_output)
        
        if not self.batch_first:
            # Convert back to seq_len, batch, embed_dim
            attn_output = attn_output.transpose(0, 1)
        
        if need_weights:
            # Average attention weights across heads for compatibility
            avg_attn_weights = attn_weights.mean(dim=1)
            return attn_output, avg_attn_weights
        else:
            return attn_output, None

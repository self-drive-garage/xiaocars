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

class DeepSpeedCompatibleMultiheadAttention(nn.Module):
    """DeepSpeed-compatible wrapper for MultiheadAttention optimized for ZeRO-3"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        # Separate linear layers for better DeepSpeed parameter sharding
        # Using bias=True for better numerical stability
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        
        # Initialize weights for better convergence
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with improved initialization for transformers"""
        # Use Xavier initialization for better gradient flow
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        
        # Output projection gets smaller initialization
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """
        Forward pass compatible with PyTorch's MultiheadAttention interface
        Optimized for DeepSpeed ZeRO-3 parameter sharding
        """
        if not self.batch_first:
            # Convert from seq_len, batch, embed_dim to batch, seq_len, embed_dim
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Project to Q, K, V with efficient computation
        Q = self.q_proj(query)  # [batch, seq_len, embed_dim]
        K = self.k_proj(key)    # [batch, seq_len, embed_dim]
        V = self.v_proj(value)  # [batch, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with scaled dot-product
        scale = (self.head_dim ** -0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Reshape back to [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
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


class DeepSpeedOptimizedMultiheadAttention(nn.Module):
    """
    Enhanced version with additional optimizations for DeepSpeed:
    - Memory efficient attention computation
    - Better gradient checkpointing support
    - Optimized for ZeRO-3 parameter sharding
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True, use_flash_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attention = use_flash_attention
        
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        # Fused QKV projection for better memory efficiency (optional)
        self.use_fused_qkv = True
        
        if self.use_fused_qkv:
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters optimized for transformer training"""
        if self.use_fused_qkv:
            nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
            if self.qkv_proj.bias is not None:
                nn.init.constant_(self.qkv_proj.bias, 0.0)
        else:
            for module in [self.q_proj, self.k_proj, self.v_proj]:
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Output projection
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """
        Memory-efficient forward pass optimized for DeepSpeed
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Efficient QKV computation
        if self.use_fused_qkv and torch.equal(query, key) and torch.equal(key, value):
            # Self-attention case: use fused QKV projection
            qkv = self.qkv_proj(query)  # [batch, seq_len, 3*embed_dim]
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            Q, K, V = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        else:
            # Cross-attention case: separate projections
            if self.use_fused_qkv:
                Q = self.qkv_proj(query)[:, :, :embed_dim]
                K = self.qkv_proj(key)[:, :, embed_dim:2*embed_dim]
                V = self.qkv_proj(value)[:, :, 2*embed_dim:]
            else:
                Q = self.q_proj(query)
                K = self.k_proj(key)
                V = self.v_proj(value)
            
            # Reshape for multi-head attention
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with memory optimization
        scale = (self.head_dim ** -0.5)
        
        # Use memory-efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and not need_weights:
            # Use PyTorch's optimized SDPA (available in PyTorch 2.0+)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            avg_attn_weights = None
        else:
            # Fallback to manual implementation
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores.masked_fill_(attn_mask, float('-inf'))
                else:
                    scores = scores + attn_mask
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            
            if need_weights:
                avg_attn_weights = attn_weights.mean(dim=1)
            else:
                avg_attn_weights = None
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        attn_output = self.out_proj(attn_output)
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, avg_attn_weights 
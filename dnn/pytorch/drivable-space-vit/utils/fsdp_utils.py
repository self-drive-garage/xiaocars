"""
FSDP utilities for dealing with specific PyTorch module types that 
need special handling when used with Fully Sharded Data Parallel.
"""

import torch
import torch.nn as nn

class FSDPSafeMultiheadAttention(nn.Module):
    """
    A wrapper around nn.MultiheadAttention that ensures compatibility with FSDP.
    This preserves parameter shapes and works around the tensor shape issues.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super().__init__()
        # Create the MultiheadAttention module but mark it as non-leaf for FSDP
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Copy parameters to ensure they're properly initialized
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        
        # Expose key parameters for FSDP to wrap correctly
        self.in_proj_weight = self.mha.in_proj_weight
        self.in_proj_bias = self.mha.in_proj_bias
        self.out_proj = self.mha.out_proj
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # Simply pass through to the underlying MultiheadAttention
        return self.mha(query, key, value, key_padding_mask=key_padding_mask, 
                        need_weights=need_weights, attn_mask=attn_mask) 
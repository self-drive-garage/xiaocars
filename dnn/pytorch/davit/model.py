"""
Minimal Vision Transformer for Drivable Space Detection.

This module implements a simplified transformer-based architecture specifically
designed for binary/ternary segmentation of drivable spaces. The model uses:
- Patch-based image encoding
- Transformer blocks for global context
- Simple upsampling decoder
- Efficient attention mechanisms

The architecture is optimized for training on a single 32GB GPU while maintaining
competitive performance with CNN-based approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """
    Converts image into patches and embeds them.
    
    Args:
        img_size: Input image size (height, width)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Dimension of patch embeddings
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 1024),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patches_per_row = img_size[1] // patch_size
        self.patches_per_col = img_size[0] // patch_size
        
        # Projection layer
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        # Ensure input dimensions are divisible by patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}, {W}) must be divisible by patch size {self.patch_size}"
        
        # Project patches: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # Reshape: (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with optional relative position bias.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in qkv projection
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use bias in qkv projection
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate for stochastic depth
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # Attention
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        # Attention block
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Args:
        drop_prob: Probability of dropping path
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class SimpleDecoder(nn.Module):
    """
    Simple upsampling decoder for segmentation.
    
    Args:
        embed_dim: Dimension of transformer embeddings
        num_classes: Number of output classes
        patch_size: Size of patches used in encoding
        hidden_dims: Hidden dimensions for decoder layers
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 3,
        patch_size: int = 16,
        hidden_dims: Tuple[int, ...] = (256, 128, 64)
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Build decoder layers
        layers = []
        in_dim = embed_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.ConvTranspose2d(
                    in_dim, 
                    hidden_dim, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Final projection
        layers.append(
            nn.ConvTranspose2d(
                hidden_dims[-1], 
                num_classes, 
                kernel_size=4, 
                stride=2, 
                padding=1
            )
        )
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(
        self, 
        x: torch.Tensor, 
        patches_per_row: int, 
        patches_per_col: int
    ) -> torch.Tensor:
        """
        Args:
            x: Transformer output of shape (B, N, C)
            patches_per_row: Number of patches per row
            patches_per_col: Number of patches per column
            
        Returns:
            Segmentation map of shape (B, num_classes, H, W)
        """
        B, N, C = x.shape
        
        # Reshape to spatial dimensions
        x = x.transpose(1, 2)  # (B, C, N)
        x = x.reshape(B, C, patches_per_col, patches_per_row)
        
        # Decode to full resolution
        x = self.decoder(x)
        
        return x


class DrivableSpaceTransformer(nn.Module):
    """
    Complete transformer model for drivable space detection.
    
    This model combines:
    - Patch embedding for input encoding
    - Transformer blocks for global context modeling
    - Simple decoder for segmentation output
    
    Args:
        img_size: Input image size (height, width)
        patch_size: Size of image patches
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        embed_dim: Dimension of embeddings
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP dimension expansion ratio
        qkv_bias: Whether to use bias in qkv projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Drop path rate for stochastic depth
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 1024),
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Decoder
        self.decoder = SimpleDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            patch_size=patch_size
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _resize_pos_embed(self, pos_embed: torch.Tensor, new_num_patches: int, 
                         new_h: int, new_w: int) -> torch.Tensor:
        """Resize position embeddings to match new patch grid dimensions."""
        # pos_embed shape: (1, old_num_patches, embed_dim)
        old_num_patches = pos_embed.shape[1]
        embed_dim = pos_embed.shape[2]
        
        # Calculate original grid size
        old_h = self.patch_embed.patches_per_col
        old_w = self.patch_embed.patches_per_row
        
        # Reshape to 2D grid
        pos_embed_grid = pos_embed.reshape(1, old_h, old_w, embed_dim).permute(0, 3, 1, 2)
        # pos_embed_grid shape: (1, embed_dim, old_h, old_w)
        
        # Bilinear interpolation to new size
        pos_embed_resized = F.interpolate(
            pos_embed_grid, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        # pos_embed_resized shape: (1, embed_dim, new_h, new_w)
        
        # Reshape back to sequence
        pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1).reshape(1, new_num_patches, embed_dim)
        
        return pos_embed_resized
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        # Calculate actual patch grid dimensions based on input
        patches_per_col = H // self.patch_embed.patch_size
        patches_per_row = W // self.patch_embed.patch_size
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding (resize if necessary)
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate position embeddings to match actual number of patches
            pos_embed_resized = self._resize_pos_embed(self.pos_embed, x.shape[1], patches_per_col, patches_per_row)
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        # Decode to segmentation map using actual dimensions
        x = self.decoder(x, patches_per_row, patches_per_col)
        
        return x
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> DrivableSpaceTransformer:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DrivableSpaceTransformer model
    """
    model = DrivableSpaceTransformer(
        img_size=tuple(config.get('img_size', [512, 1024])),
        patch_size=config.get('patch_size', 16),
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 3),
        embed_dim=config.get('embed_dim', 384),
        depth=config.get('depth', 6),
        num_heads=config.get('num_heads', 6),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        qkv_bias=config.get('qkv_bias', True),
        drop_rate=config.get('drop_rate', 0.0),
        attn_drop_rate=config.get('attn_drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.1)
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    model = DrivableSpaceTransformer(
        img_size=(512, 1024),
        patch_size=16,
        num_classes=3,
        embed_dim=384,
        depth=6
    )
    
    # Print model info
    print(f"Model created successfully!")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 1024)
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"\nModel parameter memory: {param_memory:.2f} GB")
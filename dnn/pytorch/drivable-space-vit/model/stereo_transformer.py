import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path

from .patch_embed import PatchEmbed
from .transformer_encoder_layer import TransformerEncoderLayer
from .cross_view_transformer_layer import CrossViewTransformerLayer
from .temporal_transfomer_layer import TemporalTransformerLayer
from .ego_motion_encoder import EgoMotionEncoder
from .drivable_space_decoder import DrivableSpaceDecoder

def get_default_model_config():
    """Return default model configuration"""
    return {
        'img_size': 224,
        'patch_size': 16,
        'num_channels': 3,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'ego_motion_dim': 6,
    }

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 2D sine-cosine positional embeddings for patches.
    Args:
        embed_dim: Embedding dimension for each position
        grid_size: Number of patches in each dimension
        cls_token: Whether to add class token position embedding
    Returns:
        Positional embeddings of shape (1, grid_size*grid_size(+1), embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # Here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Create 2D positional embeddings from a grid.
    Args:
        embed_dim: Embedding dimension for each position
        grid: Grid of positions
    Returns:
        Positional embeddings
    """
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Create 1D sine-cosine positional embeddings.
    Args:
        embed_dim: Embedding dimension for each position
        pos: Grid of positions
    Returns:
        Positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class StereoTransformer(nn.Module):
    """
    Main model for stereo vision transformer with temporal modeling and ego motion integration
    """
    def __init__(
        self,
        img_size=None,
        patch_size=None,
        in_chans=None,
        embed_dim=None,
        depth=None,
        num_heads=None,
        mlp_ratio=None,
        dropout=None,
        attn_dropout=None,
        ego_motion_dim=None,
        config=None,
    ):
        super().__init__()
        
        # Get model configuration
        model_config = get_default_model_config()
        if config is not None and 'model' in config:
            # Update with provided config
            for key, value in config['model'].items():
                model_config[key] = value
        
        # Use provided parameters if given, otherwise use config
        self.img_size = img_size if img_size is not None else model_config['img_size']
        self.patch_size = patch_size if patch_size is not None else model_config['patch_size']
        in_chans = in_chans if in_chans is not None else model_config['num_channels']
        self.embed_dim = embed_dim if embed_dim is not None else model_config['embed_dim']
        depth = depth if depth is not None else model_config['num_layers']
        num_heads = num_heads if num_heads is not None else model_config['num_heads']
        mlp_ratio = mlp_ratio if mlp_ratio is not None else model_config['mlp_ratio']
        dropout_value = dropout if dropout is not None else model_config['dropout']
        attn_dropout_value = attn_dropout if attn_dropout is not None else model_config['attn_dropout']
        ego_motion_dim = ego_motion_dim if ego_motion_dim is not None else model_config['ego_motion_dim']
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # Patch embedding for left and right images
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            config=config,
        )
        
        # Position embeddings for spatial transformer
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim),
            requires_grad=False,
        )
        
        # CLS token for each view
        self.cls_token_left = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token_right = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Dropout after pos embed
        self.pos_drop = nn.Dropout(dropout_value)
        
        # Spatial transformer layers for left and right images separately
        self.spatial_transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout_value,
                attn_dropout=attn_dropout_value,
                config=config,
            )
            for _ in range(depth // 3)  # Use 1/3 of layers for spatial processing
        ])
        
        # Cross-view transformer layers for stereo fusion
        self.cross_view_transformer_layers = nn.ModuleList([
            CrossViewTransformerLayer(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout_value,
                attn_dropout=attn_dropout_value,
                config=config,
            )
            for _ in range(depth // 3)  # Use 1/3 of layers for cross-view fusion
        ])
        
        # Temporal transformer layers
        self.temporal_transformer_layers = nn.ModuleList([
            TemporalTransformerLayer(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout_value,
                attn_dropout=attn_dropout_value,
                config=config,
            )
            for _ in range(depth // 3)  # Use 1/3 of layers for temporal modeling
        ])
        
        # Ego motion encoder
        self.ego_motion_encoder = EgoMotionEncoder(
            ego_motion_dim=ego_motion_dim,
            embed_dim=self.embed_dim,
            dropout=dropout_value,
            config=config,
        )
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Decoder for drivable space segmentation
        self.drivable_space_decoder = DrivableSpaceDecoder(
            embed_dim=self.embed_dim,
            img_size=self.img_size,
            patch_size=self.patch_size,
            dropout=dropout_value,
            config=config,
        )
        
        # Image reconstruction decoder (for self-supervised training)
        self.image_reconstruction_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.patch_size * self.patch_size * in_chans),
        )
        
        # Future prediction head (for self-supervised training)
        self.future_prediction_head = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.initialize_pos_embed()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def initialize_pos_embed(self):
        # Initialize positional embeddings for patches
        pos_embed = get_2d_sincos_pos_embed(
            self.embed_dim, 
            int(self.img_size / self.patch_size)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def prepare_tokens(self, left_imgs, right_imgs):
        B = left_imgs.shape[0]
        
        # Extract patch embeddings for left and right images
        left_patches = self.patch_embed(left_imgs)  # (B, N, E)
        right_patches = self.patch_embed(right_imgs)  # (B, N, E)
        
        # Add position embeddings
        left_patches = left_patches + self.pos_embed
        right_patches = right_patches + self.pos_embed
        
        # Add CLS tokens
        cls_left = self.cls_token_left.expand(B, -1, -1)  # (B, 1, E)
        cls_right = self.cls_token_right.expand(B, -1, -1)  # (B, 1, E)
        
        left_patches = torch.cat([cls_left, left_patches], dim=1)  # (B, 1+N, E)
        right_patches = torch.cat([cls_right, right_patches], dim=1)  # (B, 1+N, E)
        
        # Apply dropout
        left_patches = self.pos_drop(left_patches)
        right_patches = self.pos_drop(right_patches)
        
        return left_patches, right_patches
    
    def spatial_encode(self, left_patches, right_patches):
        # Apply spatial transformer layers to each view independently
        for layer in self.spatial_transformer_layers:
            left_patches = layer(left_patches)
            right_patches = layer(right_patches)
        
        return left_patches, right_patches
    
    def cross_view_fusion(self, left_patches, right_patches):
        # Apply cross-view transformer layers for stereo fusion
        for layer in self.cross_view_transformer_layers:
            left_patches, right_patches = layer(left_patches, right_patches)
        
        return left_patches, right_patches
    
    def temporal_encode(self, features):
        # Apply temporal transformer layers
        for layer in self.temporal_transformer_layers:
            features = layer(features)
        
        return features
    
    def forward_features(self, left_imgs, right_imgs, ego_motion=None):
        # left_imgs, right_imgs shape: (B, T, C, H, W)
        # ego_motion shape: (B, T, ego_motion_dim)
        B, T, C, H, W = left_imgs.shape
        
        # Process each frame in the sequence
        left_features_seq = []
        right_features_seq = []
        
        for t in range(T):
            # Extract features for each frame
            left_frame = left_imgs[:, t]  # (B, C, H, W)
            right_frame = right_imgs[:, t]  # (B, C, H, W)
            
            # Prepare tokens
            left_patches, right_patches = self.prepare_tokens(left_frame, right_frame)
            
            # Spatial encoding
            left_spatial, right_spatial = self.spatial_encode(left_patches, right_patches)
            
            # Cross-view fusion
            left_fused, right_fused = self.cross_view_fusion(left_spatial, right_spatial)
            
            # Store features for temporal processing
            left_features_seq.append(left_fused)
            right_features_seq.append(right_fused)
        
        # Stack features across time
        left_features = torch.stack(left_features_seq, dim=1)  # (B, T, 1+N, E)
        right_features = torch.stack(right_features_seq, dim=1)  # (B, T, 1+N, E)
        
        # Integrate ego motion if provided
        if ego_motion is not None:
            ego_features = self.ego_motion_encoder(ego_motion)  # (B, T, E)
            ego_features = ego_features.unsqueeze(2)  # (B, T, 1, E)
            
            # Add ego motion features to the CLS tokens
            left_features[:, :, 0:1, :] = left_features[:, :, 0:1, :] + ego_features
            right_features[:, :, 0:1, :] = right_features[:, :, 0:1, :] + ego_features
        
        # Reshape for temporal transformer
        # Combine left and right features for joint temporal processing
        B, T, N, E = left_features.shape
        
        # For CLS tokens, process them separately through temporal dimension
        left_cls = left_features[:, :, 0].permute(0, 2, 1)  # (B, E, T)
        right_cls = right_features[:, :, 0].permute(0, 2, 1)  # (B, E, T)
        cls_temporal = torch.cat([left_cls, right_cls], dim=1)  # (B, 2*E, T)
        cls_temporal = cls_temporal.permute(0, 2, 1)  # (B, T, 2*E)
        
        # For patch tokens, process each patch position through time
        left_patches = left_features[:, :, 1:]  # (B, T, N-1, E)
        right_patches = right_features[:, :, 1:]  # (B, T, N-1, E)
        
        # Reshape to process each patch position across time
        left_patches = left_patches.permute(0, 2, 1, 3)  # (B, N-1, T, E)
        right_patches = right_patches.permute(0, 2, 1, 3)  # (B, N-1, T, E)
        
        # Apply temporal transformer to CLS tokens (for global temporal understanding)
        cls_features = self.temporal_encode(cls_temporal)  # (B, T, 2*E)
        
        # Take features from the last timestamp for final representation
        final_cls = cls_features[:, -1]  # (B, 2*E)
        
        # Use last frame's patch features for final output (could be modified for different tasks)
        final_left_patches = left_patches[:, :, -1, :]  # (B, N-1, E)
        final_right_patches = right_patches[:, :, -1, :]  # (B, N-1, E)
        
        # Combine CLS token with patches for the final representation
        final_left_cls = final_cls[:, :self.embed_dim].unsqueeze(1)  # (B, 1, E)
        final_right_cls = final_cls[:, self.embed_dim:].unsqueeze(1)  # (B, 1, E)
        
        final_left = torch.cat([final_left_cls, final_left_patches], dim=1)  # (B, 1+N-1, E)
        final_right = torch.cat([final_right_cls, final_right_patches], dim=1)  # (B, 1+N-1, E)
        
        # Apply final normalization
        final_left = self.norm(final_left)
        final_right = self.norm(final_right)
        
        return final_left, final_right, cls_features
    
    def forward(self, batch, task='drivable_space'):
        left_imgs = batch['left_images']  # (B, T, C, H, W)
        right_imgs = batch['right_images']  # (B, T, C, H, W)
        ego_motion = batch.get('ego_motion')  # (B, T, ego_motion_dim) or None
        
        # Extract features
        left_features, right_features, cls_features = self.forward_features(
            left_imgs, right_imgs, ego_motion
        )
        
        # Task-specific outputs
        outputs = {}
        
        if task == 'drivable_space' or task == 'all':
            # Use left camera view for drivable space prediction
            # Skip the CLS token for patch-based prediction
            drivable_space = self.drivable_space_decoder(left_features[:, 1:])  # (B, 1, H, W)
            outputs['drivable_space'] = drivable_space
        
        if task == 'reconstruction' or task == 'all':
            # Reconstruct images for self-supervised training
            B, N, E = left_features.shape
            patch_size = self.patch_size
            in_chans = left_imgs.shape[2]
            
            # Skip the CLS token for reconstruction
            left_reconstructed = self.image_reconstruction_decoder(left_features[:, 1:])
            right_reconstructed = self.image_reconstruction_decoder(right_features[:, 1:])
            
            # Reshape to image dimensions
            left_reconstructed = left_reconstructed.reshape(
                B, self.num_patches, in_chans, patch_size, patch_size
            )
            right_reconstructed = right_reconstructed.reshape(
                B, self.num_patches, in_chans, patch_size, patch_size
            )
            
            # Rearrange patches to form images
            h_dim = w_dim = int(self.img_size / patch_size)
            left_reconstructed = left_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            right_reconstructed = right_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            
            left_reconstructed = left_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, self.img_size, self.img_size
            )
            right_reconstructed = right_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, self.img_size, self.img_size
            )
            
            outputs['left_reconstructed'] = left_reconstructed
            outputs['right_reconstructed'] = right_reconstructed
        
        if task == 'future_prediction' or task == 'all':
            # Predict future features for self-supervised training
            future_prediction = self.future_prediction_head(cls_features[:, -1])  # (B, 2*E)
            outputs['future_prediction'] = future_prediction
        
        return outputs

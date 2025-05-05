#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Implementation for Transformer-Based Self-Supervised Drivable Space Detection

This module contains the model architecture, dataset loading, and training utilities
for self-supervised drivable space detection using stereo vision and ego motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import math
import os
import time
import json
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange, repeat
from functools import partial
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Default constants
IMG_SIZE = 224  # Input image size for ViT
PATCH_SIZE = 16  # Patch size for ViT
NUM_CHANNELS = 3  # RGB images
EMBED_DIM = 768  # Embedding dimension
NUM_HEADS = 12  # Number of attention heads
NUM_LAYERS = 12  # Number of transformer layers
MLP_RATIO = 4  # Expansion ratio for MLP
DROPOUT = 0.1  # Dropout probability
EGO_MOTION_DIM = 6  # Ego motion dimensions (speed, acceleration, steering, etc.)
SEQ_LEN = 5  # Number of frames in sequence


###################
# Helper Functions #
###################

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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    """Load model checkpoint"""
    if not path or not os.path.exists(path):
        return 0, float('inf')
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {path}")
    return checkpoint['epoch'], checkpoint['loss']


class CosineSchedulerWithWarmup:
    """Cosine learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr_scale = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(self.min_lr, lr_scale * self.base_lrs[i])
        
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict['base_lrs']


#####################
# Dataset and Utils #
#####################

class DrivingDataset(Dataset):
    """Dataset for stereo driving data with ego motion"""
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_len: int = SEQ_LEN,
        img_size: int = IMG_SIZE,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.img_size = img_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        
        # Load metadata
        self.metadata_file = self.data_dir / f"{split}_metadata.json"
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter sequences to ensure they have at least seq_len frames
        self.sequences = [seq for seq in self.metadata['sequences'] if len(seq['frames']) >= seq_len]
        logger.info(f"Loaded {len(self.sequences)} sequences for {split}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Randomly select a starting frame, ensuring we have enough frames for the sequence
        start_idx = random.randint(0, len(sequence['frames']) - self.seq_len)
        frames = sequence['frames'][start_idx:start_idx + self.seq_len]
        
        # Load images and ego motion data
        left_images = []
        right_images = []
        ego_motion = []
        
        for frame in frames:
            # Load left and right camera images
            left_img_path = self.data_dir / frame['left_image_path']
            right_img_path = self.data_dir / frame['right_image_path']
            
            left_img = Image.open(left_img_path).convert('RGB')
            right_img = Image.open(right_img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
            
            left_images.append(left_img)
            right_images.append(right_img)
            
            # Extract ego motion data (speed, acceleration, steering, etc.)
            frame_ego_motion = torch.tensor([
                frame['speed'],
                frame['acceleration']['x'],
                frame['acceleration']['y'],
                frame['acceleration']['z'],
                frame['steering_angle'],
                frame['angular_velocity']['z']  # Yaw rate
            ], dtype=torch.float32)
            
            ego_motion.append(frame_ego_motion)
        
        # Stack data
        left_images = torch.stack(left_images)   # (seq_len, C, H, W)
        right_images = torch.stack(right_images) # (seq_len, C, H, W)
        ego_motion = torch.stack(ego_motion)     # (seq_len, ego_motion_dim)
        
        return {
            'left_images': left_images,
            'right_images': right_images,
            'ego_motion': ego_motion,
            'sequence_id': sequence['id'],
        }


def create_dataloader(dataset, batch_size, num_workers, shuffle=True):
    """Create data loader for the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )


#######################
# Model Architecture #
#######################

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding: Convert image to patch embeddings
    """
    def __init__(
        self, 
        img_size=IMG_SIZE, 
        patch_size=PATCH_SIZE, 
        in_chans=NUM_CHANNELS, 
        embed_dim=EMBED_DIM,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # (B, C, H, W) -> (B, E, H//P, W//P) -> (B, H//P * W//P, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head self-attention and MLP"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=DROPOUT,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, attn_mask=None):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class CrossViewTransformerLayer(nn.Module):
    """Transformer layer for cross-view attention between left and right images"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=DROPOUT,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Normalization layers
        self.norm1_left = nn.LayerNorm(dim)
        self.norm1_right = nn.LayerNorm(dim)
        self.norm2_left = nn.LayerNorm(dim)
        self.norm2_right = nn.LayerNorm(dim)
        
        # Cross-attention: left to right and right to left
        self.cross_attn_left_to_right = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        self.cross_attn_right_to_left = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        # MLP blocks for both paths
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_left = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        self.mlp_right = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
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


class TemporalTransformerLayer(nn.Module):
    """Transformer layer for temporal attention across frames"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=DROPOUT,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Temporal self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class EgoMotionEncoder(nn.Module):
    """Encoder for ego motion data"""
    def __init__(
        self,
        ego_motion_dim=EGO_MOTION_DIM,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.ego_motion_dim = ego_motion_dim
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(ego_motion_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, ego_motion_dim)
        return self.encoder(x)  # (batch_size, seq_len, embed_dim)


class DrivableSpaceDecoder(nn.Module):
    """Decoder for drivable space segmentation"""
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = (img_size // patch_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, patch_size * patch_size),  # One value per pixel in patch
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        B, N, E = x.shape
        
        # Decode to patch-level predictions
        x = self.decoder(x)  # (batch_size, num_patches, patch_size*patch_size)
        
        # Reshape to image dimensions
        x = x.reshape(B, self.patch_dim, self.patch_dim, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, self.img_size, self.img_size)
        
        return x


class StereoTransformer(nn.Module):
    """
    Main model for stereo vision transformer with temporal modeling and ego motion integration
    """
    def __init__(
        self,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=NUM_CHANNELS,
        embed_dim=EMBED_DIM,
        depth=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=DROPOUT,
        ego_motion_dim=EGO_MOTION_DIM,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding for left and right images
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        # Position embeddings for spatial transformer
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False,
        )
        
        # CLS token for each view
        self.cls_token_left = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_right = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout after pos embed
        self.pos_drop = nn.Dropout(dropout)
        
        # Spatial transformer layers for left and right images separately
        self.spatial_transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth // 3)  # Use 1/3 of layers for spatial processing
        ])
        
        # Cross-view transformer layers for stereo fusion
        self.cross_view_transformer_layers = nn.ModuleList([
            CrossViewTransformerLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth // 3)  # Use 1/3 of layers for cross-view fusion
        ])
        
        # Temporal transformer layers
        self.temporal_transformer_layers = nn.ModuleList([
            TemporalTransformerLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth // 3)  # Use 1/3 of layers for temporal modeling
        ])
        
        # Ego motion encoder
        self.ego_motion_encoder = EgoMotionEncoder(
            ego_motion_dim=ego_motion_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder for drivable space segmentation
        self.drivable_space_decoder = DrivableSpaceDecoder(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        # Image reconstruction decoder (for self-supervised training)
        self.image_reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, patch_size * patch_size * in_chans),
        )
        
        # Future prediction head (for self-supervised training)
        self.future_prediction_head = nn.Linear(embed_dim, embed_dim)
        
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
        left_cls = left_features[:, :, 0]  # (B, T, E)
        right_cls = right_features[:, :, 0]  # (B, T, E)
        
        # Concatenate CLS tokens from both views
        cls_temporal = torch.cat([left_cls, right_cls], dim=2)  # (B, T, 2*E)
        
        # For patch tokens, process each patch position through time
        left_patches = left_features[:, :, 1:]  # (B, T, N-1, E)
        right_patches = right_features[:, :, 1:]  # (B, T, N-1, E)
        
        # Apply temporal transformer to CLS tokens (for global temporal understanding)
        cls_features = self.temporal_encode(cls_temporal)  # (B, T, 2*E)
        
        # Take features from the last timestamp for final representation
        final_cls = cls_features[:, -1]  # (B, 2*E)
        
        # Use last frame's patch features for final output
        final_left_patches = left_patches[:, -1]  # (B, N-1, E)
        final_right_patches = right_patches[:, -1]  # (B, N-1, E)
        
        # Split the final CLS features back into left and right
        final_left_cls = final_cls[:, :self.embed_dim].unsqueeze(1)  # (B, 1, E)
        final_right_cls = final_cls[:, self.embed_dim:].unsqueeze(1)  # (B, 1, E)
        
        # Combine CLS token with patches for the final representation
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


#######################
# Self-Supervised Loss #
#######################

class SelfSupervisedLoss(nn.Module):
    """Combined loss for self-supervised training"""
    def __init__(self, reconstruction_weight=1.0, consistency_weight=1.0, future_weight=0.5):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.future_weight = future_weight
        
        self.reconstruction_loss = nn.MSELoss()
        self.consistency_loss = nn.CosineSimilarity(dim=1)
        self.future_prediction_loss = nn.MSELoss()
    
    def forward(self, outputs, batch):
        loss = 0.0
        loss_dict = {}
        
        # Reconstruction loss (if available)
        if 'left_reconstructed' in outputs and 'right_reconstructed' in outputs:
            # Get the last frame from the sequence for reconstruction target
            left_target = batch['left_images'][:, -1]  # (B, C, H, W)
            right_target = batch['right_images'][:, -1]  # (B, C, H, W)
            
            left_recon_loss = self.reconstruction_loss(outputs['left_reconstructed'], left_target)
            right_recon_loss = self.reconstruction_loss(outputs['right_reconstructed'], right_target)
            recon_loss = (left_recon_loss + right_recon_loss) / 2.0
            
            loss += self.reconstruction_weight * recon_loss
            loss_dict['reconstruction_loss'] = recon_loss.item()
        
        # View consistency loss (if available)
        if 'left_reconstructed' in outputs and 'right_reconstructed' in outputs:
            # Calculate consistency between left and right reconstructions
            # This encourages the model to learn stereo correspondence
            left_recon = outputs['left_reconstructed']
            right_recon = outputs['right_reconstructed']
            
            # Calculate cosine similarity and convert to a loss (1 - similarity)
            consistency = 1.0 - self.consistency_loss(
                left_recon.reshape(left_recon.size(0), -1),
                right_recon.reshape(right_recon.size(0), -1)
            ).mean()
            
            loss += self.consistency_weight * consistency
            loss_dict['consistency_loss'] = consistency.item()
        
        # Future prediction loss (if available)
        if 'future_prediction' in outputs and batch.get('future_features') is not None:
            future_loss = self.future_prediction_loss(
                outputs['future_prediction'],
                batch['future_features']
            )
            
            loss += self.future_weight * future_loss
            loss_dict['future_prediction_loss'] = future_loss.item()
        
        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict


#######################
# Training Functions #
#######################

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    epoch,
    device,
    loss_fn,
    log_interval=10,
    scaler=None,
    gradient_accumulation=1,
):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    num_batches = len(data_loader)
    
    # Reset gradients at the beginning
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(data_loader):
        # Move data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if requested
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch, task='all')
                loss, loss_dict = loss_fn(outputs, batch)
                # Scale the loss by gradient accumulation steps
                loss = loss / gradient_accumulation
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation steps
            if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == num_batches:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard forward pass
            outputs = model(batch, task='all')
            loss, loss_dict = loss_fn(outputs, batch)
            # Scale the loss by gradient accumulation steps
            loss = loss / gradient_accumulation
            
            # Standard backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation steps
            if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == num_batches:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
        
        # Update total loss (with the un-scaled value for logging)
        total_loss += loss_dict['total_loss'] * gradient_accumulation
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            lr = scheduler.step(epoch + batch_idx / num_batches)
            
            # Calculate time per batch and estimated time remaining
            elapsed_time = time.time() - start_time
            time_per_batch = elapsed_time / (batch_idx + 1)
            remaining_batches = num_batches - (batch_idx + 1)
            eta = time_per_batch * remaining_batches
            
            # Get memory usage
            if device.type == 'cuda':
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
                gpu_mem_str = f", GPU memory: {gpu_mem:.2f}GB"
            else:
                gpu_mem_str = ""
            
            # Log progress
            logger.info(
                f"Epoch: {epoch+1} [{batch_idx+1}/{num_batches} ({100.0 * (batch_idx+1) / num_batches:.0f}%)] "
                f"Loss: {loss_dict['total_loss']:.6f} "
                f"LR: {lr[0]:.6f} "
                f"Time/batch: {time_per_batch:.3f}s "
                f"ETA: {eta / 60:.2f}m{gpu_mem_str}"
            )
            
            # Log individual loss components (optional)
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    logger.info(f"  - {k}: {v:.6f}")
    
    # Return average loss for the epoch
    return total_loss / num_batches


def validate(model, data_loader, device, loss_fn):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(batch, task='all')
            
            # Calculate loss
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Update total loss
            total_loss += loss_dict['total_loss']
    
    # Return average loss
    return total_loss / len(data_loader)


def visualize_predictions(model, data_loader, device, output_dir, num_samples=5):
    """Visualize model predictions"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(batch, task='all')
            
            # Get last frame from sequence
            left_img = batch['left_images'][:, -1]  # (B, C, H, W)
            
            # Get predictions
            drivable_space = outputs.get('drivable_space')  # (B, 1, H, W)
            left_recon = outputs.get('left_reconstructed')  # (B, C, H, W)
            
            # Convert to numpy for visualization
            for b in range(min(left_img.size(0), 3)):  # Visualize up to 3 samples per batch
                # Original image
                img = left_img[b].cpu().permute(1, 2, 0).numpy()
                img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                img = img.astype(np.uint8)
                
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title('Original Image')
                plt.axis('off')
                
                if drivable_space is not None:
                    # Drivable space prediction
                    ds_pred = drivable_space[b, 0].cpu().numpy()
                    plt.subplot(1, 3, 2)
                    plt.imshow(ds_pred, cmap='viridis')
                    plt.title('Drivable Space Prediction')
                    plt.axis('off')
                
                if left_recon is not None:
                    # Reconstructed image
                    recon = left_recon[b].cpu().permute(1, 2, 0).numpy()
                    recon = (recon * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    recon = np.clip(recon, 0, 255).astype(np.uint8)
                    plt.subplot(1, 3, 3)
                    plt.imshow(recon)
                    plt.title('Reconstructed Image')
                    plt.axis('off')
                
                # Save figure
                sample_idx = i * left_img.size(0) + b
                plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}.png'))
                plt.close()


#######################
# Inference Functions #
#######################

def load_model_for_inference(checkpoint_path, device=None):
    """Load model for inference"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = StereoTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=NUM_CHANNELS,
        embed_dim=EMBED_DIM,
        depth=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=0.0,  # No dropout during inference
        attn_dropout=0.0,  # No dropout during inference
        ego_motion_dim=EGO_MOTION_DIM,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model


def predict_drivable_space(model, left_images, right_images, ego_motion=None, device=None):
    """Predict drivable space for a sequence of stereo images"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model is in eval mode
    model.eval()
    
    # Prepare inputs
    batch = {
        'left_images': left_images.to(device),
        'right_images': right_images.to(device),
    }
    
    if ego_motion is not None:
        batch['ego_motion'] = ego_motion.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(batch, task='drivable_space')
    
    # Return drivable space prediction
    return outputs['drivable_space']


def evaluate_on_dataset(model, data_loader, device, output_dir=None):
    """Evaluate model on a dataset"""
    model.eval()
    
    # Initialize metrics
    total_samples = 0
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(batch, task='drivable_space')
            
            # Get predictions
            drivable_space = outputs['drivable_space']  # (B, 1, H, W)
            
            # Update total samples
            total_samples += drivable_space.size(0)
            
            # Save predictions if output directory is provided
            if output_dir:
                for b in range(drivable_space.size(0)):
                    # Get sample ID
                    sample_id = f"{batch_idx}_{b}"
                    if 'sequence_id' in batch:
                        sample_id = f"{batch['sequence_id'][b]}_{b}"
                    
                    # Save drivable space prediction
                    ds_pred = drivable_space[b, 0].cpu().numpy()
                    np.save(os.path.join(output_dir, f"{sample_id}_drivable_space.npy"), ds_pred)
                    
                    # Visualize prediction
                    plt.figure(figsize=(10, 10))
                    plt.imshow(ds_pred, cmap='viridis')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{sample_id}_drivable_space.png"))
                    plt.close()
    
    logger.info(f"Evaluated model on {total_samples} samples")
    return total_samples


def convert_model_to_onnx(model, output_path, img_size=IMG_SIZE, batch_size=1, seq_len=SEQ_LEN):
    """Convert model to ONNX format for deployment"""
    model.eval()
    
    # Create dummy inputs
    left_images = torch.randn(batch_size, seq_len, 3, img_size, img_size)
    right_images = torch.randn(batch_size, seq_len, 3, img_size, img_size)
    ego_motion = torch.randn(batch_size, seq_len, EGO_MOTION_DIM)
    
    # Create batch dictionary
    batch = {
        'left_images': left_images,
        'right_images': right_images,
        'ego_motion': ego_motion,
    }
    
    # Export model to ONNX
    torch.onnx.export(
        model,
        (batch,),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['left_images', 'right_images', 'ego_motion'],
        output_names=['drivable_space'],
        dynamic_axes={
            'left_images': {0: 'batch_size'},
            'right_images': {0: 'batch_size'},
            'ego_motion': {0: 'batch_size'},
            'drivable_space': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Exported model to ONNX format: {output_path}")
    return output_path


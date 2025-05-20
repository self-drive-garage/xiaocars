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
from .ego_motion_encoder import EgoMotionEncoder, MotionGuidedAttention
from .drivable_space_decoder import DrivableSpaceDecoder
from .future_predictor import MotionGuidedFuturePredictor

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
        'ego_motion_dim': 9,
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

class MultiViewTransformer(nn.Module):
    """
    Main model for multi-view vision transformer with temporal modeling and early ego motion integration
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
        
        # Patch embedding for all three views
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            config=config,
        )
        
        # Ego motion encoder - MOVED EARLIER in the model architecture
        self.ego_motion_encoder = EgoMotionEncoder(
            ego_motion_dim=ego_motion_dim,
            embed_dim=self.embed_dim,
            dropout=dropout_value,
            config=config,
        )
        
        # Motion-guided attention for each view
        self.motion_attention = MotionGuidedAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
        )
        
        # Early ego motion conditioning module - NEW
        self.early_motion_conditioning = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        
        # Position embeddings for spatial transformer
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim),
            requires_grad=False,
        )
        
        # CLS token for each view
        self.cls_token_left = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token_center = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token_right = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Dropout after pos embed
        self.pos_drop = nn.Dropout(dropout_value)
        
        # Spatial transformer layers for each view independently
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
        
        # Cross-view transformer layers for multi-view fusion
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
        
        # Temporal projection layer for combined CLS tokens
        self.temporal_projection = nn.Linear(3 * self.embed_dim, self.embed_dim)
        
        # Image reconstruction decoder (simple MLP)
        self.image_reconstruction_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.patch_size * self.patch_size * in_chans),
        )
        
        # Drivable space decoder
        self.drivable_space_decoder = DrivableSpaceDecoder(
            embed_dim=self.embed_dim,
            img_size=self.img_size,
            patch_size=self.patch_size,
            dropout=dropout_value,
            config=config,
        )
        
        # Future prediction head
        self.future_prediction_head = MotionGuidedFuturePredictor(
            embed_dim=self.embed_dim,
            ego_motion_dim=ego_motion_dim,
            dropout=dropout_value
            # ,
            # config=config,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize position embedding
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
    
    def prepare_tokens_with_motion(self, left_imgs, center_imgs, right_imgs, ego_features=None):
        """Prepare tokens for each view with early ego motion integration"""
        B = left_imgs.shape[0]
        
        # Extract patch embeddings for all three views
        left_patches = self.patch_embed(left_imgs)  # (B, N, E)
        center_patches = self.patch_embed(center_imgs)  # (B, N, E)
        right_patches = self.patch_embed(right_imgs)  # (B, N, E)
        
        # Add position embeddings
        left_patches = left_patches + self.pos_embed
        center_patches = center_patches + self.pos_embed
        right_patches = right_patches + self.pos_embed
        
        # Add CLS tokens
        cls_left = self.cls_token_left.expand(B, -1, -1)  # (B, 1, E)
        cls_center = self.cls_token_center.expand(B, -1, -1)  # (B, 1, E)
        cls_right = self.cls_token_right.expand(B, -1, -1)  # (B, 1, E)
        
        left_patches = torch.cat([cls_left, left_patches], dim=1)  # (B, 1+N, E)
        center_patches = torch.cat([cls_center, center_patches], dim=1)  # (B, 1+N, E)
        right_patches = torch.cat([cls_right, right_patches], dim=1)  # (B, 1+N, E)
        
        # EARLY EGO MOTION INTEGRATION - condition patches with motion features
        if ego_features is not None:
            # Transform ego features for conditioning
            motion_cond = self.early_motion_conditioning(ego_features)  # (B, E)
            
            # Add motion features to all patches including CLS token
            motion_cond = motion_cond.unsqueeze(1)  # (B, 1, E)
            left_patches = left_patches + motion_cond
            center_patches = center_patches + motion_cond
            right_patches = right_patches + motion_cond
        
        # Apply dropout
        left_patches = self.pos_drop(left_patches)
        center_patches = self.pos_drop(center_patches)
        right_patches = self.pos_drop(right_patches)
        
        return left_patches, center_patches, right_patches
    
    def spatial_encode(self, left_patches, center_patches, right_patches):
        # Add debugging code to log tensor shapes
        if hasattr(torch.distributed, 'get_rank') and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank == 0:  # Only print from rank 0 to avoid flooding logs
                print(f"DEBUG: Input tensor shapes before spatial encoding - left: {left_patches.shape}, center: {center_patches.shape}, right: {right_patches.shape}")
                # Check tensor properties to help diagnose FSDP issues
                for name, tensor in [("left", left_patches), ("center", center_patches), ("right", right_patches)]:
                    print(f"DEBUG: {name} tensor - dim: {tensor.dim()}, dtype: {tensor.dtype}, device: {tensor.device}, requires_grad: {tensor.requires_grad}")
        
        # Ensure that tensors have correct dimensionality for attention processing
        for tensor_name, tensor in [("left_patches", left_patches), ("center_patches", center_patches), ("right_patches", right_patches)]:
            if tensor.dim() < 3 and tensor.numel() > 0:
                if tensor.dim() == 2 and tensor.size(1) == self.embed_dim:
                    # Case where we have [batch_size, embed_dim] but need [batch_size, seq_len, embed_dim]
                    if tensor_name == "left_patches":
                        left_patches = tensor.unsqueeze(1)
                    elif tensor_name == "center_patches":
                        center_patches = tensor.unsqueeze(1)
                    elif tensor_name == "right_patches":
                        right_patches = tensor.unsqueeze(1)
        
        # Apply spatial transformer layers to each view independently
        left_patches_output = left_patches
        center_patches_output = center_patches
        right_patches_output = right_patches
        
        for i, layer in enumerate(self.spatial_transformer_layers):
            try:
                left_patches_output = layer(left_patches_output)
                center_patches_output = layer(center_patches_output)
                right_patches_output = layer(right_patches_output)
            except RuntimeError as e:
                if "mat2 must be a matrix" in str(e):
                    # Log error info and shapes for debugging
                    if hasattr(torch.distributed, 'get_rank') and torch.distributed.is_initialized():
                        rank = torch.distributed.get_rank()
                        if rank == 0:
                            print(f"DEBUG: Error in spatial_encode at layer {i}: {str(e)}")
                            print(f"DEBUG: Tensor shapes - left: {left_patches_output.shape}, center: {center_patches_output.shape}, right: {right_patches_output.shape}")
                    # Re-raise to see full traceback
                    raise
                else:
                    # Re-raise other errors
                    raise
        
        return left_patches_output, center_patches_output, right_patches_output
    
    def cross_view_fusion(self, left_patches, center_patches, right_patches):
        # Apply cross-view transformer layers for multi-view fusion
        # Process pairs of views and then combine
        for layer in self.cross_view_transformer_layers:
            # Fuse left and center views
            left_fused, center_fused = layer(left_patches, center_patches)
            # Fuse center and right views
            center_fused, right_fused = layer(center_fused, right_patches)
            # Fuse right and left views to complete the cycle
            right_fused, left_fused = layer(right_fused, left_fused)
            
            left_patches, center_patches, right_patches = left_fused, center_fused, right_fused
        
        return left_patches, center_patches, right_patches
    
    def temporal_encode(self, features):
        # Check if features have triple width (combined left/center/right CLS tokens)
        if features.shape[-1] == 3 * self.embed_dim:
            # Project from 3*embed_dim to embed_dim
            features = self.temporal_projection(features)
        
        # Apply temporal transformer layers
        for layer in self.temporal_transformer_layers:
            features = layer(features)
        
        return features
    
    def forward_features(self, left_imgs, center_imgs, right_imgs, ego_motion=None):
        # left_imgs, center_imgs, right_imgs shape: (B, T, C, H, W)
        # ego_motion shape: (B, T, ego_motion_dim)
        B, T, C, H, W = left_imgs.shape
        
        # EARLY EGO MOTION PROCESSING (Approach 1)
        ego_features_seq = None
        if ego_motion is not None:
            # Process ego motion for the entire sequence first
            ego_features_seq = self.ego_motion_encoder(ego_motion)  # (B, T, E)
        
        # Process each frame in the sequence
        left_features_seq = []
        center_features_seq = []
        right_features_seq = []
        
        for t in range(T):
            # Extract features for each frame
            left_frame = left_imgs[:, t]  # (B, C, H, W)
            center_frame = center_imgs[:, t]  # (B, C, H, W)
            right_frame = right_imgs[:, t]  # (B, C, H, W)
            
            # Get ego features for this timestep
            ego_features_t = None
            if ego_features_seq is not None:
                ego_features_t = ego_features_seq[:, t]  # (B, E)
            
            # Prepare tokens with early motion integration
            left_patches, center_patches, right_patches = self.prepare_tokens_with_motion(
                left_frame, center_frame, right_frame, ego_features_t
            )
            
            # Spatial encoding
            left_spatial, center_spatial, right_spatial = self.spatial_encode(left_patches, center_patches, right_patches)
            
            # Cross-view fusion
            left_fused, center_fused, right_fused = self.cross_view_fusion(left_spatial, center_spatial, right_spatial)
            
            # Store features for temporal processing
            left_features_seq.append(left_fused)
            center_features_seq.append(center_fused)
            right_features_seq.append(right_fused)
        
        # Stack features across time
        left_features = torch.stack(left_features_seq, dim=1)  # (B, T, 1+N, E)
        center_features = torch.stack(center_features_seq, dim=1)  # (B, T, 1+N, E)
        right_features = torch.stack(right_features_seq, dim=1)  # (B, T, 1+N, E)
        
        # ADDITIONAL EGO MOTION INTEGRATION for temporal processing
        if ego_motion is not None and ego_features_seq is not None:
            # Use ego motion for motion-guided attention on temporal features
            # Apply motion-guided attention to visual features
            left_motion_attn = self.motion_attention(left_features, ego_features_seq)  # (B, T, N, 1)
            center_motion_attn = self.motion_attention(center_features, ego_features_seq)  # (B, T, N, 1)
            right_motion_attn = self.motion_attention(right_features, ego_features_seq)  # (B, T, N, 1)
            
            # Apply attention weights to features with residual connection
            left_features = left_features * left_motion_attn + left_features
            center_features = center_features * center_motion_attn + center_features
            right_features = right_features * right_motion_attn + right_features
        
        # Reshape for temporal transformer
        # Combine left, center, and right features for joint temporal processing
        B, T, N, E = left_features.shape
        
        # For CLS tokens, process them separately through temporal dimension
        left_cls = left_features[:, :, 0].permute(0, 2, 1)  # (B, E, T)
        center_cls = center_features[:, :, 0].permute(0, 2, 1)  # (B, E, T)
        right_cls = right_features[:, :, 0].permute(0, 2, 1)  # (B, E, T)
        cls_temporal = torch.cat([left_cls, center_cls, right_cls], dim=1)  # (B, 3*E, T)
        cls_temporal = cls_temporal.permute(0, 2, 1)  # (B, T, 3*E)
        
        # For patch tokens, process each patch position through time
        left_patches = left_features[:, :, 1:]  # (B, T, N-1, E)
        center_patches = center_features[:, :, 1:]  # (B, T, N-1, E)
        right_patches = right_features[:, :, 1:]  # (B, T, N-1, E)
        
        # Reshape to process each patch position across time
        left_patches = left_patches.permute(0, 2, 1, 3)  # (B, N-1, T, E)
        center_patches = center_patches.permute(0, 2, 1, 3)  # (B, N-1, T, E)
        right_patches = right_patches.permute(0, 2, 1, 3)  # (B, N-1, T, E)
        
        # Apply temporal transformer to CLS tokens (for global temporal understanding)
        cls_features = self.temporal_encode(cls_temporal)  # (B, T, E) after projection
        
        # Store original combined features for future prediction
        combined_cls_features = cls_temporal  # (B, T, 3*E)
        
        # Take features from the last timestamp for final representation
        final_cls = cls_features[:, -1]  # (B, E)
        
        # Use last frame's patch features for final output (could be modified for different tasks)
        final_left_patches = left_patches[:, :, -1, :]  # (B, N-1, E)
        final_center_patches = center_patches[:, :, -1, :]  # (B, N-1, E)
        final_right_patches = right_patches[:, :, -1, :]  # (B, N-1, E)
        
        # Split the final CLS token back into view components
        final_left_cls = final_cls.unsqueeze(1)  # (B, 1, E)
        final_center_cls = final_cls.unsqueeze(1)  # (B, 1, E)
        final_right_cls = final_cls.unsqueeze(1)  # (B, 1, E)
        
        final_left = torch.cat([final_left_cls, final_left_patches], dim=1)  # (B, 1+N-1, E)
        final_center = torch.cat([final_center_cls, final_center_patches], dim=1)  # (B, 1+N-1, E)
        final_right = torch.cat([final_right_cls, final_right_patches], dim=1)  # (B, 1+N-1, E)
        
        # Apply final normalization
        final_left = self.norm(final_left)
        final_center = self.norm(final_center)
        final_right = self.norm(final_right)
        
        return final_left, final_center, final_right, combined_cls_features
    
    def forward(self, batch, task='drivable_space'):
        left_imgs = batch['left_images']  # (B, T, C, H, W)
        center_imgs = batch['center_images']  # (B, T, C, H, W)
        right_imgs = batch['right_images']  # (B, T, C, H, W)
        ego_motion = batch.get('ego_motion')  # (B, T, ego_motion_dim) or None
        
        # Extract features
        left_features, center_features, right_features, cls_features = self.forward_features(
            left_imgs, center_imgs, right_imgs, ego_motion
        )
        
        # Task-specific outputs
        outputs = {}
        
        if task == 'drivable_space' or task == 'all':
            # Extract motion context for drivable space prediction
            if ego_motion is not None:
                # Use the last timestamp's ego motion as context
                motion_context = ego_motion[:, -1]  # (B, ego_motion_dim)
                # Encode it to feature space
                motion_context = self.ego_motion_encoder(motion_context.unsqueeze(1)).squeeze(1)  # (B, embed_dim)
            else:
                motion_context = None
                
            # Use center camera view for drivable space prediction with motion context
            # Skip the CLS token for patch-based prediction
            drivable_space = self.drivable_space_decoder(
                center_features[:, 1:],  # Remove CLS token - use center view
                motion_context
            )  # (B, 1, H, W)
            
            outputs['drivable_space'] = drivable_space
        
        if task == 'reconstruction' or task == 'all':
            # Reconstruct images for self-supervised training
            B, N, E = left_features.shape
            patch_size = self.patch_size
            in_chans = left_imgs.shape[2]
            
            # Skip the CLS token for reconstruction
            left_reconstructed = self.image_reconstruction_decoder(left_features[:, 1:])
            center_reconstructed = self.image_reconstruction_decoder(center_features[:, 1:])
            right_reconstructed = self.image_reconstruction_decoder(right_features[:, 1:])
            
            # Reshape to image dimensions
            left_reconstructed = left_reconstructed.reshape(
                B, self.num_patches, in_chans, patch_size, patch_size
            )
            center_reconstructed = center_reconstructed.reshape(
                B, self.num_patches, in_chans, patch_size, patch_size
            )
            right_reconstructed = right_reconstructed.reshape(
                B, self.num_patches, in_chans, patch_size, patch_size
            )
            
            # Rearrange patches to form images
            h_dim = w_dim = int(self.img_size / patch_size)
            left_reconstructed = left_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            center_reconstructed = center_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            right_reconstructed = right_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            
            left_reconstructed = left_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, self.img_size, self.img_size
            )
            center_reconstructed = center_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, self.img_size, self.img_size
            )
            right_reconstructed = right_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, self.img_size, self.img_size
            )
            
            outputs['left_reconstructed'] = left_reconstructed
            outputs['center_reconstructed'] = center_reconstructed
            outputs['right_reconstructed'] = right_reconstructed
        
        if task == 'future_prediction' or task == 'all':
            # Enhanced future prediction with motion guidance
            if ego_motion is not None:
                # Use the enhanced future predictor that incorporates ego motion
                future_prediction = self.future_prediction_head(cls_features, ego_motion)  # (B, 3*E)
            else:
                # Fallback to simple prediction if no ego motion data
                # Create a zero tensor matching the ego_motion_dim size (now 9 instead of 15)
                zeros = torch.zeros_like(cls_features[:, :, :9])
                future_prediction = self.future_prediction_head(cls_features, zeros)
            
            outputs['future_prediction'] = future_prediction
        
        return outputs

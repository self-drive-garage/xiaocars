import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from .patch_embed import PatchEmbed
from .transformer_encoder_layer import TransformerEncoderLayer
from .cross_view_transformer_layer import CrossViewTransformerLayer
from .temporal_transfomer_layer import TemporalTransformerLayer
from .ego_motion_encoder import EgoMotionEncoder
from .drivable_space_decoder import DrivableSpaceDecoder
from .future_predictor import MotionGuidedFuturePredictor

# Get logger
logger = logging.getLogger(__name__)

class SpatialTransformerModule(nn.Module):
    """Spatial transformer module for processing individual views"""
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        num_layers,
        num_heads,
        mlp_ratio,
        dropout,
        attn_dropout,
        config
    ):
        super().__init__()
        
        # Compute number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False
        )
        
        # Initialize position embedding with sine-cosine
        pos_embed = self._get_sincos_pos_embed(embed_dim, int(img_size / patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
  
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                config=config
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _get_sincos_pos_embed(self, embed_dim, grid_size):
        """Generate sine-cosine positional embedding."""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        # Use half of dimensions to encode grid_h
        assert embed_dim % 2 == 0
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb
    
    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """Generate 1D sine-cosine positional embedding."""
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    def forward(self, x, motion_features):
        """
        Args:
            x: Input images [B, C, H, W]
            motion_features: Optional motion features [B, D]
        Returns:
            features: Transformer features [B, N+1, D]
        """
        # Add diagnostic logging
        logger.debug(f"SpatialTransformerModule input x shape: {x.shape}, dtype: {x.dtype}")
        logger.debug(f"motion_features shape: {motion_features.shape}, dtype: {motion_features.dtype}")
        
        B = x.shape[0]
        
        # Extract patches
        x = self.patch_embed(x)  # [B, N, D]
        logger.debug(f"After patch_embed shape: {x.shape}")
        
        # Add position embedding
        x = x + self.pos_embed  # [B, N, D]
        logger.debug(f"After adding pos_embed shape: {x.shape}")
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        logger.debug(f"CLS token shape: {cls_token.shape}")
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N, D]
        logger.debug(f"After adding CLS token shape: {x.shape}")
        
        # Update CLS token with motion information
        x[:, 0] = x[:, 0] + motion_features  # Add motion to CLS token only
        
        # Apply transformer layers
        logger.debug(f"Before transformer layers, x shape: {x.shape}")
        for i, layer in enumerate(self.transformer_layers):
            try:
                logger.debug(f"Applying transformer layer {i}, input shape: {x.shape}")
                x = layer(x)  # [B, 1+N, D]
                logger.debug(f"After transformer layer {i}, output shape: {x.shape}")
            except Exception as e:
                logger.error(f"Error in transformer layer {i}: {str(e)}")
                logger.error(f"Layer input shape: {x.shape}, ndim: {x.ndim}")
                raise
        
        # Apply layer norm
        x = self.norm(x)  # [B, 1+N, D]
        logger.debug(f"Final output shape after norm: {x.shape}")
        
        return x


class CrossViewTransformerModule(nn.Module):
    """Cross-view attention module for fusing features from different views"""
    def __init__(
        self,
        embed_dim=768,
        num_layers=4,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
        attn_dropout=0.1
    ):
        super().__init__()
        
        # Cross-view transformer layers
        self.transformer_layers = nn.ModuleList([
            CrossViewTransformerLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm_left = nn.LayerNorm(embed_dim)
        self.norm_center = nn.LayerNorm(embed_dim)
        self.norm_right = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, left_features, center_features, right_features):
        """
        Args:
            left_features: Features from left view [B, N+1, D]
            center_features: Features from center view [B, N+1, D]
            right_features: Features from right view [B, N+1, D]
        Returns:
            fused_left: Fused left features [B, N+1, D]
            fused_center: Fused center features [B, N+1, D]
            fused_right: Fused right features [B, N+1, D]
        """
        # Apply cross-view transformer layers
        left_fused, center_fused, right_fused = left_features, center_features, right_features
        
        for layer in self.transformer_layers:
            left_fused, center_fused, right_fused = layer(left_fused, center_fused, right_fused)
        
        # Apply layer norm
        left_fused = self.norm_left(left_fused)
        center_fused = self.norm_center(center_fused)
        right_fused = self.norm_right(right_fused)
        
        return left_fused, center_fused, right_fused


class TemporalTransformerModule(nn.Module):
    """Temporal transformer module for modeling sequences"""
    def __init__(
        self,
        embed_dim=768,
        num_layers=4,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
        attn_dropout=0.1
    ):
        super().__init__()
        
        # Projection layer for combined CLS tokens
        self.cls_projection = nn.Linear(3 * embed_dim, embed_dim)
        
        # Temporal transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalTransformerLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, cls_features):
        """
        Args:
            cls_features: Sequence of combined CLS tokens [B, T, 3*D]
        Returns:
            temporal_features: Temporal features [B, T, D]
        """
        # Project from 3*embed_dim to embed_dim
        features = self.cls_projection(cls_features)  # [B, T, D]
        
        # Apply temporal transformer layers
        for layer in self.transformer_layers:
            features = layer(features)  # [B, T, D]
        
        # Apply layer norm
        features = self.norm(features)  # [B, T, D]
        
        return features


class ModularVisionTransformer(nn.Module):
    """
    Modular vision transformer for multi-view, temporal understanding
    with explicit components for FSDP-friendly distributed training
    """
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        spatial_layers,
        cross_view_layers,
        temporal_layers,
        num_heads,
        mlp_ratio,
        dropout,
        attn_dropout,
        ego_motion_dim,
        config
    ):
        super().__init__()
        
        # Ego motion encoder
        self.ego_motion_encoder = EgoMotionEncoder(
            ego_motion_dim=ego_motion_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Spatial transformer for each view
        self.spatial_transformer = SpatialTransformerModule(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_layers=spatial_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            config=config
        )
        
        # Cross-view transformer
        self.cross_view_transformer = CrossViewTransformerModule(
            embed_dim=embed_dim,
            num_layers=cross_view_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformerModule(
            embed_dim=embed_dim,
            num_layers=temporal_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Drivable space decoder
        self.drivable_space_decoder = DrivableSpaceDecoder(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            dropout=dropout
        )
        
        # Future prediction head
        self.future_prediction_head = MotionGuidedFuturePredictor(
            embed_dim=embed_dim,
            ego_motion_dim=ego_motion_dim,
            dropout=dropout
        )
        
        # Image reconstruction decoder
        self.image_reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        )
        
        # Save parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
    
    def forward(self, batch, task='drivable_space'):
        """
        Args:
            batch: Dictionary with:
                - left_images: [B, T, C, H, W]
                - center_images: [B, T, C, H, W]
                - right_images: [B, T, C, H, W]
                - ego_motion: [B, T, ego_motion_dim] (optional)
            task: Task to perform ('drivable_space', 'reconstruction', 'future_prediction', 'all')
        Returns:
            outputs: Dictionary with task-specific outputs
        """
        left_imgs = batch['left_images']  # [B, T, C, H, W]
        center_imgs = batch['center_images']  # [B, T, C, H, W]
        right_imgs = batch['right_images']  # [B, T, C, H, W]
        ego_motion = batch['ego_motion']  # [B, T, ego_motion_dim]
        
        logger.debug(f"ModularVisionTransformer input shapes - left_imgs: {left_imgs.shape}, center_imgs: {center_imgs.shape}, right_imgs: {right_imgs.shape}")
        logger.debug(f"ego_motion shape: {ego_motion.shape}, dtype: {ego_motion.dtype}")
        
        B, T, C, H, W = left_imgs.shape
        
 
        # Process ego motion one frame at a time to avoid shape mismatch
        ego_features_list = []
        for t in range(T):
            # Process each frame separately
            motion_frame = ego_motion[:, t]  # [B, ego_motion_dim]
            logger.debug(f"Processing ego motion frame {t}, shape: {motion_frame.shape}")
            
            # Use encode_frame which handles 2D input correctly
            try:
                frame_features = self.ego_motion_encoder.encode_frame(motion_frame)  # [B, embed_dim]
                logger.debug(f"Encoded ego motion frame {t}, output shape: {frame_features.shape}")
                ego_features_list.append(frame_features)
            except Exception as e:
                logger.error(f"Error encoding ego motion frame {t}: {str(e)}")
                logger.error(f"motion_frame: shape={motion_frame.shape}, dtype={motion_frame.dtype}, ndim={motion_frame.ndim}")
                raise
            
        # Stack all processed features
        ego_features_seq = torch.stack(ego_features_list, dim=1)  # [B, T, embed_dim]
        logger.debug(f"Stacked ego_features_seq shape: {ego_features_seq.shape}")
    
        # Process each frame in the sequence
        left_features_seq = []
        center_features_seq = []
        right_features_seq = []
        cls_features_seq = []
        
        for t in range(T):
            # Extract current frame
            left_frame = left_imgs[:, t]  # [B, C, H, W]
            center_frame = center_imgs[:, t]  # [B, C, H, W]
            right_frame = right_imgs[:, t]  # [B, C, H, W]
            
            logger.debug(f"Frame {t} shapes - left: {left_frame.shape}, center: {center_frame.shape}, right: {right_frame.shape}")
            
            # Get ego features for this timestep
            ego_features_t = ego_features_seq[:, t]  # [B, embed_dim]
            logger.debug(f"ego_features_t for frame {t} shape: {ego_features_t.shape}, dtype: {ego_features_t.dtype}")
            
            # Process each view with spatial transformer
            logger.debug(f"Calling spatial_transformer with left_frame shape: {left_frame.shape}")
            logger.debug(f"ego_features_t shape: {ego_features_t.shape}, dtype: {ego_features_t.dtype}, ndim: {ego_features_t.ndim}")
            
            left_features = self.spatial_transformer(left_frame, ego_features_t)  # [B, 1+N, D]
            logger.debug(f"left_features after spatial_transformer: {left_features.shape}")
            
            center_features = self.spatial_transformer(center_frame, ego_features_t)  # [B, 1+N, D]
            right_features = self.spatial_transformer(right_frame, ego_features_t) 
            
            # Fuse views with cross-view transformer
            left_fused, center_fused, right_fused = self.cross_view_transformer(
                left_features, center_features, right_features
            )  # [B, 1+N, D]
            
            # Store features for temporal processing
            left_features_seq.append(left_fused)
            center_features_seq.append(center_fused)
            right_features_seq.append(right_fused)
            
            # Combine CLS tokens for temporal processing
            left_cls = left_fused[:, 0]  # [B, D]
            center_cls = center_fused[:, 0]  # [B, D]
            right_cls = right_fused[:, 0]  # [B, D]
            combined_cls = torch.cat([left_cls, center_cls, right_cls], dim=1)  # [B, 3*D]
            cls_features_seq.append(combined_cls)
        
        # Stack features across time
        left_features = torch.stack(left_features_seq, dim=1)  # [B, T, 1+N, D]
        center_features = torch.stack(center_features_seq, dim=1)  # [B, T, 1+N, D]
        right_features = torch.stack(right_features_seq, dim=1)  # [B, T, 1+N, D]
        cls_features = torch.stack(cls_features_seq, dim=1)  # [B, T, 3*D]
        
        # Apply temporal transformer to CLS features
        temporal_features = self.temporal_transformer(cls_features)  # [B, T, D]
        
        # Use the last timestamp for final prediction
        final_temporal = temporal_features[:, -1]  # [B, D]
        
        # Use last frame's features
        final_left = left_features[:, -1]  # [B, 1+N, D]
        final_center = center_features[:, -1]  # [B, 1+N, D]
        final_right = right_features[:, -1]  # [B, 1+N, D]
        
        # Task-specific outputs
        outputs = {}
        
        if task == 'drivable_space' or task == 'all':
            # Extract motion context for drivable space prediction
            if ego_motion is not None:
                # Use the last timestamp's ego motion as context
                motion_context = ego_motion[:, -1]  # [B, ego_motion_dim]
                # Encode it to feature space using encode_frame which handles 2D input correctly
                motion_context = self.ego_motion_encoder.encode_frame(motion_context)  # [B, embed_dim]
            else:
                motion_context = None
            
            # Use center camera view for drivable space prediction with motion context
            # Skip the CLS token for patch-based prediction
            drivable_space = self.drivable_space_decoder(
                final_center[:, 1:],  # Remove CLS token - use center view
                motion_context
            )  # [B, 1, H, W]
            
            outputs['drivable_space'] = drivable_space
        
        if task == 'reconstruction' or task == 'all':
            # Reconstruct images for self-supervised training
            patch_size = self.patch_size
            in_chans = self.in_chans
            img_size = self.img_size
            num_patches = (img_size // patch_size) ** 2
            
            # Skip the CLS token for reconstruction
            left_reconstructed = self.image_reconstruction_decoder(final_left[:, 1:])
            center_reconstructed = self.image_reconstruction_decoder(final_center[:, 1:])
            right_reconstructed = self.image_reconstruction_decoder(final_right[:, 1:])
            
            # Reshape to image dimensions
            left_reconstructed = left_reconstructed.reshape(
                B, num_patches, in_chans, patch_size, patch_size
            )
            center_reconstructed = center_reconstructed.reshape(
                B, num_patches, in_chans, patch_size, patch_size
            )
            right_reconstructed = right_reconstructed.reshape(
                B, num_patches, in_chans, patch_size, patch_size
            )
            
            # Rearrange patches to form images
            h_dim = w_dim = int(img_size / patch_size)
            left_reconstructed = left_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            center_reconstructed = center_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            right_reconstructed = right_reconstructed.reshape(B, h_dim, w_dim, in_chans, patch_size, patch_size)
            
            left_reconstructed = left_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, img_size, img_size
            )
            center_reconstructed = center_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, img_size, img_size
            )
            right_reconstructed = right_reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(
                B, in_chans, img_size, img_size
            )
            
            outputs['left_reconstructed'] = left_reconstructed
            outputs['center_reconstructed'] = center_reconstructed
            outputs['right_reconstructed'] = right_reconstructed
        
        if task == 'future_prediction' or task == 'all':
            # Enhanced future prediction with motion guidance
            if ego_motion is not None:
                # Use the enhanced future predictor that incorporates ego motion
                future_prediction = self.future_prediction_head(cls_features, ego_motion)  # [B, 3*D]
            else:
                # Fallback to simple prediction if no ego motion data
                # Create a zero tensor matching the ego_motion_dim size
                zeros = torch.zeros_like(cls_features[:, :, :ego_motion.shape[-1]])
                future_prediction = self.future_prediction_head(cls_features, zeros)
            
            outputs['future_prediction'] = future_prediction
        
        return outputs 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


from .ds_transformer_encoder_layer import DeepSpeedTransformerEncoderLayer
from model.patch_embed import PatchEmbed
from model.cross_view_transformer_layer import CrossViewTransformerLayer
from model.temporal_transfomer_layer import TemporalTransformerLayer
from model.ego_motion_encoder import EgoMotionEncoder
from model.drivable_space_decoder import DrivableSpaceDecoder
from model.future_predictor import MotionGuidedFuturePredictor

# Get logger
logger = logging.getLogger(__name__)

class DeepSpeedSpatialTransformerModule(nn.Module):
    """DeepSpeed-compatible spatial transformer module for processing individual views"""
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
        
        # Transformer layers - use DeepSpeed compatible version
        self.transformer_layers = nn.ModuleList([
            DeepSpeedTransformerEncoderLayer(
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
        logger.debug(f"DeepSpeedSpatialTransformerModule input x shape: {x.shape}, dtype: {x.dtype}")
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
        logger.debug(f"Before DeepSpeed transformer layers, x shape: {x.shape}")
        for i, layer in enumerate(self.transformer_layers):
            try:
                logger.debug(f"Applying DeepSpeed transformer layer {i}, input shape: {x.shape}")
                x = layer(x)  # [B, 1+N, D]
                logger.debug(f"After DeepSpeed transformer layer {i}, output shape: {x.shape}")
            except Exception as e:
                logger.error(f"Error in DeepSpeed transformer layer {i}: {str(e)}")
                logger.error(f"Layer input shape: {x.shape}, ndim: {x.ndim}")
                raise
        
        # Apply layer norm
        x = self.norm(x)  # [B, 1+N, D]
        logger.debug(f"Final output shape after norm: {x.shape}")
        
        return x


class DeepSpeedModularVisionTransformer(nn.Module):
    """
    DeepSpeed-compatible modular vision transformer for multi-view, temporal understanding
    with explicit components optimized for DeepSpeed ZeRO-3 distributed training
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
        
        # Spatial transformer for each view - use DeepSpeed compatible version
        self.spatial_transformer = DeepSpeedSpatialTransformerModule(
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
        
        # Cross-view transformer (can keep the same for now)
        self.cross_view_transformer = CrossViewTransformerLayer(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Temporal transformer (can keep the same for now)
        self.temporal_transformer = TemporalTransformerLayer(
            dim=embed_dim,
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
            embed_dim,
            num_heads,
            dropout
        )
        
        # CLS token projection layer to map from 3*embed_dim to embed_dim
        self.cls_projection = nn.Linear(3 * embed_dim, embed_dim)
        
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
        
        # Explicitly convert all modules to FP16 for DeepSpeed compatibility
        # This ensures modules imported from the original model/ directory are also converted
        self._convert_to_fp16()
    
    def _convert_to_fp16(self):
        """Convert all modules to FP16 for DeepSpeed compatibility"""
        target_dtype = torch.float16
        
        # Convert all sub-modules to FP16
        for module in [
            self.ego_motion_encoder,
            self.cross_view_transformer, 
            self.temporal_transformer,
            self.drivable_space_decoder,
            self.future_prediction_head,
            self.cls_projection,
            self.image_reconstruction_decoder
        ]:
            module.to(target_dtype)
            
        # Also convert parameters and buffers of the spatial transformer
        self.spatial_transformer.to(target_dtype)
        
        logger.info(f"Converted all modules to {target_dtype} for DeepSpeed compatibility")
    
    def forward(self, batch, task='all'):
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
        
        logger.debug(f"DeepSpeedModularVisionTransformer input shapes - left_imgs: {left_imgs.shape}, center_imgs: {center_imgs.shape}, right_imgs: {right_imgs.shape}")
        logger.debug(f"ego_motion shape: {ego_motion.shape}, dtype: {ego_motion.dtype}")
        
        B, T, C, H, W = left_imgs.shape
        
        # Get the target dtype for all computations (FP16 for DeepSpeed)
        target_dtype = next(self.parameters()).dtype
 
        # Process ego motion one frame at a time to avoid shape mismatch
        ego_features_list = []
        for t in range(T):
            # Process each frame separately
            motion_frame = ego_motion[:, t]  # [B, ego_motion_dim]
            
            # Convert to the same dtype as model parameters (FP16 for DeepSpeed)
            if target_dtype != motion_frame.dtype:
                motion_frame = motion_frame.to(target_dtype)
            
            logger.debug(f"Processing ego motion frame {t}, shape: {motion_frame.shape}, dtype: {motion_frame.dtype}")
            
            # Use encode_frame which handles 2D input correctly
            try:
                frame_features = self.ego_motion_encoder.encode_frame(motion_frame)  # [B, embed_dim]
                # Ensure output dtype
                if frame_features.dtype != target_dtype:
                    frame_features = frame_features.to(target_dtype)
                logger.debug(f"Encoded ego motion frame {t}, output shape: {frame_features.shape}")
                ego_features_list.append(frame_features)
            except Exception as e:
                logger.error(f"Error encoding ego motion frame {t}: {str(e)}")
                logger.error(f"motion_frame: shape={motion_frame.shape}, dtype={motion_frame.dtype}, ndim={motion_frame.ndim}")
                logger.error(f"ego_motion_encoder param dtype: {next(self.ego_motion_encoder.parameters()).dtype}")
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
            
            # Convert image frames to the model's target dtype
            if left_frame.dtype != target_dtype:
                left_frame = left_frame.to(target_dtype)
                center_frame = center_frame.to(target_dtype)
                right_frame = right_frame.to(target_dtype)
            
            logger.debug(f"Frame {t} shapes - left: {left_frame.shape}, center: {center_frame.shape}, right: {right_frame.shape}")
            logger.debug(f"Frame {t} dtypes - left: {left_frame.dtype}, center: {center_frame.dtype}, right: {right_frame.dtype}")
            
            # Get ego features for this timestep
            ego_features_t = ego_features_seq[:, t]  # [B, embed_dim]
            logger.debug(f"ego_features_t for frame {t} shape: {ego_features_t.shape}, dtype: {ego_features_t.dtype}")
            
            # Process each view with spatial transformer
            logger.debug(f"Calling DeepSpeed spatial_transformer with left_frame shape: {left_frame.shape}")
            logger.debug(f"ego_features_t shape: {ego_features_t.shape}, dtype: {ego_features_t.dtype}, ndim: {ego_features_t.ndim}")
            
            left_features = self.spatial_transformer(left_frame, ego_features_t)  # [B, 1+N, D]
            logger.debug(f"left_features after DeepSpeed spatial_transformer: {left_features.shape}")
            
            center_features = self.spatial_transformer(center_frame, ego_features_t)  # [B, 1+N, D]
            right_features = self.spatial_transformer(right_frame, ego_features_t) 
            
            # Fuse views with cross-view transformer
            left_fused, center_fused, right_fused = self.cross_view_transformer(
                left_features, center_features, right_features
            )  # [B, 1+N, D]
            
            # Ensure cross-view transformer outputs are in correct dtype
            if left_fused.dtype != target_dtype:
                left_fused = left_fused.to(target_dtype)
                center_fused = center_fused.to(target_dtype)
                right_fused = right_fused.to(target_dtype)
            
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
        
        # Project combined CLS features from 3*embed_dim to embed_dim
        B, T, _ = cls_features.shape
        cls_features_flat = cls_features.view(-1, cls_features.size(-1))  # [B*T, 3*D]
        projected_cls = self.cls_projection(cls_features_flat)  # [B*T, D]
        projected_cls = projected_cls.view(B, T, -1)  # [B, T, D]
        
        # Ensure projection output is in correct dtype
        if projected_cls.dtype != target_dtype:
            projected_cls = projected_cls.to(target_dtype)
        
        logger.debug(f"cls_features shape: {cls_features.shape}, projected_cls shape: {projected_cls.shape}")
        
        # Apply temporal transformer to projected CLS features
        temporal_features = self.temporal_transformer(projected_cls)  # [B, T, D]
        
        # Ensure temporal transformer output is in correct dtype
        if temporal_features.dtype != target_dtype:
            temporal_features = temporal_features.to(target_dtype)
        
        # Use the last timestamp for final prediction
        final_temporal = temporal_features[:, -1]  # [B, D]
        
        # Use last frame's features
        final_left = left_features[:, -1]  # [B, 1+N, D]
        final_center = center_features[:, -1]  # [B, 1+N, D]
        final_right = right_features[:, -1]  # [B, 1+N, D]
        
        # Task-specific outputs
        outputs = {}
        
        if task == 'drivable_space' or task == 'all':
            # Use the last timestamp's ego motion as context
            motion_context = ego_motion[:, -1]  # [B, ego_motion_dim]
            
            # Convert to the target dtype
            if target_dtype != motion_context.dtype:
                motion_context = motion_context.to(target_dtype)
            
            # Encode it to feature space using encode_frame which handles 2D input correctly
            motion_context = self.ego_motion_encoder.encode_frame(motion_context)  # [B, embed_dim]
            
            # Ensure motion context is in correct dtype
            if motion_context.dtype != target_dtype:
                motion_context = motion_context.to(target_dtype)
            
            # Use center camera view for drivable space prediction with motion context
            # Skip the CLS token for patch-based prediction
            drivable_space = self.drivable_space_decoder(
                final_center[:, 1:],  # Remove CLS token - use center view
                motion_context
            )  # [B, 1, H, W]
            
            # Ensure output is in the correct dtype
            if drivable_space.dtype != target_dtype:
                drivable_space = drivable_space.to(target_dtype)
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
            
            # Ensure outputs are in the correct dtype
            if left_reconstructed.dtype != target_dtype:
                left_reconstructed = left_reconstructed.to(target_dtype)
                center_reconstructed = center_reconstructed.to(target_dtype)
                right_reconstructed = right_reconstructed.to(target_dtype)
            
            outputs['left_reconstructed'] = left_reconstructed
            outputs['center_reconstructed'] = center_reconstructed
            outputs['right_reconstructed'] = right_reconstructed
        
        if task == 'future_prediction' or task == 'all':
            # Enhanced future prediction with motion guidance
            future_prediction = self.future_prediction_head(cls_features)  # [B, T, future_dim]
            
            # Ensure output is in the correct dtype
            if future_prediction.dtype != target_dtype:
                future_prediction = future_prediction.to(target_dtype)
            
            outputs['future_prediction'] = future_prediction
        
        # Final safety check: ensure all outputs are in the correct dtype
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.dtype != target_dtype:
                outputs[key] = value.to(target_dtype)
        
        return outputs 
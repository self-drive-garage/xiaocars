import torch
import torch.nn as nn
import torch.nn.functional as F

def get_default_config():
    """Return default configuration parameters for DrivableSpaceDecoder"""
    return {
        'embed_dim': 768,
        'img_size': 224,
        'patch_size': 16,
        'dropout': 0.1,
    }

class MotionAwareUpsampling(nn.Module):
    """Motion-aware upsampling block for better drivable space prediction"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DrivableSpaceDecoder(nn.Module):
    """Enhanced decoder for drivable space segmentation with motion awareness"""
    def __init__(
        self,
        embed_dim=None,
        img_size=None,
        patch_size=None,
        dropout=None,
        config=None,
    ):
        super().__init__()
        
        # Get default configuration
        default_config = get_default_config()
        
        # Use provided config if available
        model_config = {}
        if config is not None and 'model' in config:
            model_config = config['model']
        
        # Use provided parameters if given, otherwise use config, fallback to defaults
        self.embed_dim = embed_dim if embed_dim is not None else model_config.get('embed_dim', default_config['embed_dim'])
        self.img_size = img_size if img_size is not None else model_config.get('img_size', default_config['img_size'])
        self.patch_size = patch_size if patch_size is not None else model_config.get('patch_size', default_config['patch_size'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = (self.img_size // self.patch_size)
        
        # Initial projection from transformer features to spatial features
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 4, self.patch_size * self.patch_size),
        )
        
        # Convolutional decoder for progressive upsampling
        hidden_dim = 128
        self.initial_conv = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        
        # Progressive upsampling blocks
        self.decoder_stages = nn.ModuleList([
            MotionAwareUpsampling(hidden_dim, hidden_dim // 2, scale_factor=2),
            MotionAwareUpsampling(hidden_dim // 2, hidden_dim // 4, scale_factor=2),
            MotionAwareUpsampling(hidden_dim // 4, hidden_dim // 8, scale_factor=2),
        ])
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
        
        # Context integration with skip connections
        self.context_gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x, motion_context=None):
        # x shape: (batch_size, num_patches, embed_dim)
        B, N, E = x.shape
        
        # Initial projection to patch-level predictions
        x = self.projection(x)  # (batch_size, num_patches, patch_size*patch_size)
        
        # Reshape to initial spatial representation
        x = x.reshape(B, self.patch_dim, self.patch_dim, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, self.img_size, self.img_size)
        
        # Apply convolutional decoder
        x = self.initial_conv(x)
        
        # Apply motion context if available
        if motion_context is not None:
            # Reshape motion context to spatial dimensions - assume (B, embed_dim)
            motion_spatial = motion_context.reshape(B, -1, 1, 1)
            motion_spatial = motion_spatial.expand(-1, -1, x.size(2), x.size(3))
            
            # Gate mechanism to control motion influence
            gate = torch.sigmoid(self.context_gate)
            x = x * (1 + gate * motion_spatial)
        
        # Progressive upsampling with residual connections
        features = []
        for decoder_stage in self.decoder_stages:
            features.append(x)
            x = decoder_stage(x)
            
            # Add residual if resolution matches
            if x.size(2) == features[-1].size(2) * 2:
                upsampled_prev = F.interpolate(features[-1], scale_factor=2, mode='bilinear', align_corners=False)
                x = x + upsampled_prev
        
        # Final prediction
        x = self.final_conv(x)
        
        # Resize to match input image size if needed
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return x
    
    def forward_legacy(self, x):
        """Legacy forward method for backward compatibility"""
        # x shape: (batch_size, num_patches, embed_dim)
        B, N, E = x.shape
        
        # Decode to patch-level predictions
        x = self.projection(x)  # (batch_size, num_patches, patch_size*patch_size)
        
        # Reshape to image dimensions
        x = x.reshape(B, self.patch_dim, self.patch_dim, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, self.img_size, self.img_size)
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_default_config():
    """Return default configuration parameters for DrivableSpaceDecoder"""
    return {
        'embed_dim': 512,  # Reduced from 768
        'img_size': 192,   # Reduced from 224
        'patch_size': 16,
        'dropout': 0.1,
        'hidden_dim': 128,
    }

class MemoryEfficientUpsampling(nn.Module):
    """Memory-efficient upsampling block for drivable space prediction"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        
        # Reduce channels before upsampling to save memory
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Process features at lower resolution first
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        # Then upsample (more memory efficient)
        x = self.upsample(x)
        return x

class DrivableSpaceDecoder(nn.Module):
    """Memory-efficient decoder for drivable space segmentation"""
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
        self.hidden_dim = model_config.get('hidden_dim', default_config['hidden_dim'])
        dropout_value = dropout if dropout is not None else model_config.get('dropout', default_config['dropout'])
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = (self.img_size // self.patch_size)
        
        # Simplified projection to reduce memory
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 4, self.patch_size * self.patch_size),
        )
        
        # Reduced channel dimensions throughout
        
        self.initial_conv = nn.Conv2d(1, self.hidden_dim, kernel_size=3, padding=1)
        
        # Motion context projection layer
        self.motion_projection = nn.Linear(self.embed_dim, self.hidden_dim)
        
        # Use uniform channel sizes to reduce memory footprint
        channels = [self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 2, self.hidden_dim // 2]
        self.decoder_stages = nn.ModuleList([
            MemoryEfficientUpsampling(channels[0], channels[1], scale_factor=2),
            MemoryEfficientUpsampling(channels[1], channels[2], scale_factor=2),
            MemoryEfficientUpsampling(channels[2], channels[3], scale_factor=2),
        ])
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(channels[3], 1, kernel_size=1)
        
        # Context integration
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
            # Project motion context to match hidden dimension
            motion_context = self.motion_projection(motion_context)
            
            # Apply as channel-wise scaling
            motion_spatial = motion_context.reshape(B, -1, 1, 1)
            
            # Gate mechanism
            gate = torch.sigmoid(self.context_gate)
            x = x * (1 + gate * motion_spatial)
        
        # Process through upsampling blocks - memory efficient
        for decoder_stage in self.decoder_stages:
            x = decoder_stage(x)
        
        # Final prediction
        x = self.final_conv(x)
        
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

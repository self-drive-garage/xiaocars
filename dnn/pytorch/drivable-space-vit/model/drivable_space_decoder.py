import torch
import torch.nn as nn

def get_default_config():
    """Return default configuration parameters for DrivableSpaceDecoder"""
    return {
        'embed_dim': 768,
        'img_size': 224,
        'patch_size': 16,
        'dropout': 0.1,
    }

class DrivableSpaceDecoder(nn.Module):
    """Decoder for drivable space segmentation"""
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
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(self.embed_dim // 4, self.patch_size * self.patch_size),  # One value per pixel in patch
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

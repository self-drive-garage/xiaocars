import torch
import torch.nn as nn

def get_default_config():
    """Return default configuration parameters for PatchEmbed"""
    return {
        'img_size': 224,
        'patch_size': 16,
        'num_channels': 3,
        'embed_dim': 768,
    }

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding: Convert image to patch embeddings
    """
    def __init__(
        self, 
        img_size=None, 
        patch_size=None, 
        in_chans=None, 
        embed_dim=None,
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
        self.img_size = img_size if img_size is not None else model_config.get('img_size', default_config['img_size'])
        self.patch_size = patch_size if patch_size is not None else model_config.get('patch_size', default_config['patch_size'])
        in_chans = in_chans if in_chans is not None else model_config.get('num_channels', default_config['num_channels'])
        embed_dim = embed_dim if embed_dim is not None else model_config.get('embed_dim', default_config['embed_dim'])
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # (B, C, H, W) -> (B, E, H//P, W//P) -> (B, H//P * W//P, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

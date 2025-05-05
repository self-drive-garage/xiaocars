import torch
import torch.nn as nn

# Constants for model architecture
IMG_SIZE = 224  # Input image size for ViT
PATCH_SIZE = 16  # Patch size for ViT
NUM_CHANNELS = 3  # RGB images
EMBED_DIM = 768  # Embedding dimension

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

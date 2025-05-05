import torch
import torch.nn as nn

# Constants for model architecture
EMBED_DIM = 768  # Embedding dimension
IMG_SIZE = 224  # Input image size for ViT
PATCH_SIZE = 16  # Patch size for ViT
DROPOUT = 0.1  # Dropout probability

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

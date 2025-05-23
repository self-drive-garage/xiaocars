import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from model.modular_model import create_modular_model
from model.driving_dataset import DrivingDataset

# Load config
with open('config/config_deepspeed.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = create_modular_model(config=config)

# Load checkpoint
checkpoint_path = 'outputs/deepspeed/checkpoint_epoch_19/pytorch_model.bin/pytorch_model.bin'
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# Create dataset
dataset = DrivingDataset(
    data_dir='datasets/argoversev2',
    split='val',
    seq_len=config['dataset']['seq_len'],
    img_size=config['model']['img_size'],
    random_sequence=False,
    cache_images=False,
    config=config,
)

# Get a single sample
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

with torch.no_grad():
    for batch in data_loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch, task='all')
        
        # Get data
        center_img = batch['center_images'][0, -1]  # Last frame
        drivable_space = outputs['drivable_space'][0, 0]  # Remove batch and channel dims
        center_recon = outputs['center_reconstructed'][0]
        
        # Denormalize images
        mean = torch.tensor([0.4, 0.4, 0.4], device=center_img.device)
        std = torch.tensor([0.25, 0.25, 0.25], device=center_img.device)
        
        def denormalize(img):
            img_denorm = img.clone()
            for t, m, s in zip(img_denorm, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(img_denorm, 0, 1)
        
        center_img_denorm = denormalize(center_img)
        center_recon_denorm = denormalize(center_recon)
        
        # Create figure with better visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        axes[0, 0].imshow(center_img_denorm.permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title('Original Center Image (256x256)')
        axes[0, 0].axis('off')
        
        # Drivable space - downsample for visualization
        # Downsample from 2048x2048 to 256x256 for better visualization
        drivable_space_vis = torch.nn.functional.interpolate(
            drivable_space.unsqueeze(0).unsqueeze(0), 
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )[0, 0]
        
        im = axes[0, 1].imshow(drivable_space_vis.cpu().numpy(), cmap='viridis')
        axes[0, 1].set_title(f'Drivable Space (downsampled to 256x256)\nOriginal: 2048x2048')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Reconstruction
        axes[1, 0].imshow(center_recon_denorm.permute(1, 2, 0).cpu().numpy())
        axes[1, 0].set_title('Center Reconstruction')
        axes[1, 0].axis('off')
        
        # Show a zoomed patch of drivable space
        patch_size = 128
        patch = drivable_space[:patch_size, :patch_size].cpu().numpy()
        im2 = axes[1, 1].imshow(patch, cmap='viridis')
        axes[1, 1].set_title(f'Drivable Space Patch (top-left {patch_size}x{patch_size})')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to test_visualization.png")
        print(f"Drivable space stats: min={drivable_space.min():.3f}, max={drivable_space.max():.3f}, mean={drivable_space.mean():.3f}, std={drivable_space.std():.3f}")
        
        break 
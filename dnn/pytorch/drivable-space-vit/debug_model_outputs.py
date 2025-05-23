import torch
import numpy as np
from pathlib import Path
import yaml
from model.modular_model import create_modular_model
from model.driving_dataset import DrivingDataset, create_dataloader

def debug_model_outputs():
    """Debug script to analyze model outputs"""
    
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
    
    print(f"\nModel loaded successfully")
    
    # Create a small dataset
    dataset = DrivingDataset(
        data_dir='datasets/argoversev2',
        split='val',
        seq_len=config['dataset']['seq_len'],
        img_size=config['model']['img_size'],
        random_sequence=False,
        cache_images=False,
        config=config,
    )
    
    # Get a single batch
    # Directly create DataLoader to avoid timeout issue with num_workers=0
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0
    )
    
    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            print("\n=== Input Analysis ===")
            print(f"Center images shape: {batch['center_images'].shape}")
            print(f"Center images range: [{batch['center_images'].min():.3f}, {batch['center_images'].max():.3f}]")
            print(f"Center images mean: {batch['center_images'].mean():.3f}")
            print(f"Center images std: {batch['center_images'].std():.3f}")
            
            # Forward pass
            outputs = model(batch, task='all')
            
            print("\n=== Output Analysis ===")
            
            # Check drivable space output
            if 'drivable_space' in outputs:
                ds = outputs['drivable_space']
                print(f"\nDrivable space shape: {ds.shape}")
                print(f"Drivable space range: [{ds.min():.3f}, {ds.max():.3f}]")
                print(f"Drivable space mean: {ds.mean():.3f}")
                print(f"Drivable space std: {ds.std():.3f}")
                print(f"Drivable space unique values: {torch.unique(ds).shape[0]}")
                
                # Check if output is constant or repetitive
                if ds.std() < 0.01:
                    print("WARNING: Drivable space output has very low variance!")
                
                # Check pattern
                if ds.shape[-1] > 10 and ds.shape[-2] > 10:
                    patch = ds[0, 0, :10, :10].cpu().numpy()
                    print(f"Sample patch (top-left 10x10):\n{patch}")
            
            # Check reconstruction output
            if 'center_reconstructed' in outputs:
                recon = outputs['center_reconstructed']
                print(f"\nReconstruction shape: {recon.shape}")
                print(f"Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
                print(f"Reconstruction mean: {recon.mean():.3f}")
                print(f"Reconstruction std: {recon.std():.3f}")
                
                if recon.std() < 0.01:
                    print("WARNING: Reconstruction has very low variance!")
                
                # Compare with input
                input_img = batch['center_images'][0, -1]  # Last frame
                mse = ((recon[0] - input_img) ** 2).mean()
                print(f"Reconstruction MSE: {mse:.6f}")
            
            # Check other outputs
            for key, value in outputs.items():
                if key not in ['drivable_space', 'center_reconstructed'] and isinstance(value, torch.Tensor):
                    print(f"\n{key} shape: {value.shape}")
                    print(f"{key} range: [{value.min():.3f}, {value.max():.3f}]")
            
            # Check loss components if available
            print("\n=== Loss Analysis ===")
            from model.self_supervised_loss import SelfSupervisedLoss
            loss_fn = SelfSupervisedLoss(
                reconstruction_weight=config['training']['reconstruction_weight'],
                consistency_weight=config['training']['consistency_weight'],
                future_weight=config['training']['future_weight']
            )
            
            losses = loss_fn(outputs, batch)
            print(f"Total loss: {losses['total_loss'].item():.6f}")
            for key, value in losses.items():
                if key != 'total_loss' and isinstance(value, torch.Tensor):
                    print(f"{key}: {value.item():.6f}")
            
            break  # Just analyze one batch
    
    # Check specific layer outputs
    print("\n=== Layer Weight Analysis ===")
    for name, param in model.named_parameters():
        if 'final_conv' in name or 'decoder_stages.2' in name:
            print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")

if __name__ == "__main__":
    debug_model_outputs() 
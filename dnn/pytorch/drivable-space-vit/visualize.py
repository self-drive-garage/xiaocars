import os
import sys
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepspeed

from model.modular_model import load_model_from_checkpoint
from model.driving_dataset import DrivingDataset, create_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_deepspeed_checkpoint(checkpoint_path, device, config):
    """Load a DeepSpeed checkpoint for inference
    
    Args:
        checkpoint_path: Path to the DeepSpeed checkpoint file
        device: Device to load the model on
        config: Model configuration
        
    Returns:
        model: The loaded model
        checkpoint_info: Dictionary with checkpoint metadata
    """
    # Import the model creation function
    from model.modular_model import create_modular_model
    
    checkpoint_path = Path(checkpoint_path)
    
    # Check if this is a consolidated checkpoint (from zero_to_fp32.py)
    if checkpoint_path.name == 'pytorch_model.bin' and checkpoint_path.exists():
        logger.info("Loading consolidated checkpoint from zero_to_fp32.py")
        try:
            # Create model
            model = create_modular_model(config=config)
            
            # Load the consolidated checkpoint - it contains the state dict directly
            state_dict = torch.load(checkpoint_path, map_location=device)
            
            # The consolidated checkpoint contains the model parameters directly
            model.load_state_dict(state_dict, strict=False)
            
            checkpoint_info = {
                'epoch': 'unknown (consolidated)',
                'loss': 'unknown (consolidated)'
            }
            
            logger.info(f"Successfully loaded consolidated checkpoint with {len(state_dict)} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load consolidated checkpoint: {e}")
            raise RuntimeError(f"Unable to load consolidated checkpoint: {checkpoint_path}")
    
    else:
        # Handle sharded DeepSpeed checkpoints
        # Create the model first
        model = create_modular_model(config=config)
        
        # For DeepSpeed inference, we need to use init_inference
        # This will handle gathering the partitioned parameters
        try:
            # Try to infer checkpoint directory from the file path
            checkpoint_dir = checkpoint_path.parent
            
            # Initialize DeepSpeed for inference
            # This will automatically gather the partitioned weights
            model = deepspeed.init_inference(
                model,
                checkpoint=str(checkpoint_dir),
                replace_with_kernel_inject=False,  # Don't use kernel injection for visualization
                tensor_parallel={"tp_size": 1},  # No tensor parallelism for inference
                dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            )
            
            # Get the module from DeepSpeed engine
            model = model.module
            
            # Try to load checkpoint metadata if available
            checkpoint_info = {
                'epoch': 'unknown',
                'loss': 'unknown'
            }
            
            # Look for client state file
            client_state_file = checkpoint_dir / 'checkpoint_state.json'
            if client_state_file.exists():
                import json
                with open(client_state_file, 'r') as f:
                    client_state = json.load(f)
                    checkpoint_info['epoch'] = client_state.get('epoch', 'unknown')
                    checkpoint_info['loss'] = client_state.get('loss', 'unknown')
            
            logger.info(f"Successfully loaded DeepSpeed checkpoint using init_inference")
            
        except Exception as e:
            logger.warning(f"Failed to load with init_inference: {e}")
            logger.info("Attempting alternative loading method...")
            
            try:
                # Alternative: Try using deepspeed's zero_to_fp32 utilities
                from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
                
                # Create model
                model = create_modular_model(config=config)
                
                # Load from zero checkpoint
                checkpoint_dir = checkpoint_path.parent.parent  # Go up to the checkpoint root
                model = load_state_dict_from_zero_checkpoint(model, str(checkpoint_dir))
                
                checkpoint_info = {
                    'epoch': 'unknown', 
                    'loss': 'unknown'
                }
                
                logger.info(f"Successfully loaded using zero_to_fp32 utilities")
                
            except Exception as e2:
                logger.error(f"Failed to load with zero_to_fp32: {e2}")
                
                # Last resort: Try direct state dict loading if it's a consolidated checkpoint
                try:
                    # Create model
                    model = create_modular_model(config=config)
                    
                    # Load the checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Handle different checkpoint formats
                    if 'module' in checkpoint:
                        state_dict = checkpoint['module']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        # Assume the checkpoint is the state dict directly
                        state_dict = checkpoint
                    
                    model.load_state_dict(state_dict, strict=False)
                    
                    checkpoint_info = {
                        'epoch': checkpoint.get('epoch', 'unknown'),
                        'loss': checkpoint.get('loss', 'unknown')
                    }
                    
                    logger.info(f"Successfully loaded using direct state dict")
                    
                except Exception as e3:
                    logger.error(f"All loading methods failed: {e3}")
                    raise RuntimeError(f"Unable to load DeepSpeed checkpoint: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model, checkpoint_info


def visualize_predictions(model, data_loader, device, output_dir, num_samples=10, rank=0):
    """Visualize model predictions
    
    Args:
        model: The model to generate predictions
        data_loader: DataLoader containing validation or test data
        device: Device to run model on (cuda or cpu)
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
        rank: Process rank (for distributed setup, only rank 0 visualizes)
    """
    # Importing visualization libraries here to avoid dependencies if not used
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    model.eval()
    
    # Only visualize on main process
    if rank != 0:
        return
        
    samples_visualized = 0
    with torch.no_grad():
        for batch in data_loader:
            if samples_visualized >= num_samples:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with task='all' to generate all outputs
            outputs = model(batch, task='all')
            
            # Visualize each sample in the batch
            for i in range(min(batch['center_images'].size(0), num_samples - samples_visualized)):
                # Get current sample - only use center camera
                center_img = batch['center_images'][i, -1]  # Last frame in sequence
                
                # Get reconstruction if available
                center_recon = outputs.get('center_reconstructed', None)
                
                # Get drivable space prediction if available
                drivable_space = outputs.get('drivable_space', None)
                
                # Create figure - smaller now that we only have center view
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Denormalize images for visualization
                # Use the same normalization values as in the dataset
                mean = torch.tensor([0.4, 0.4, 0.4], device=center_img.device)
                std = torch.tensor([0.25, 0.25, 0.25], device=center_img.device)
                
                # Function to denormalize an image
                def denormalize(img):
                    img_denorm = img.clone()
                    for t, m, s in zip(img_denorm, mean, std):
                        t.mul_(s).add_(m)
                    return torch.clamp(img_denorm, 0, 1)
                
                # Plot original center image (denormalized)
                center_img_denorm = denormalize(center_img)
                axes[0, 0].imshow(center_img_denorm.permute(1, 2, 0).cpu().numpy())
                axes[0, 0].set_title('Center Image')
                axes[0, 0].axis('off')
                
                # Plot drivable space prediction if available
                if drivable_space is not None:
                    # Get the original drivable space map
                    drivable_map = drivable_space[i].squeeze()  # Remove channel dimension if present
                    
                    # Get the input image size for reference
                    input_height, input_width = center_img.shape[-2:]
                    
                    # Check if downsampling is needed (if output is larger than input)
                    if drivable_map.shape[0] > input_height or drivable_map.shape[1] > input_width:
                        # Downsample to match input image size for better visualization
                        drivable_map_vis = torch.nn.functional.interpolate(
                            drivable_map.unsqueeze(0).unsqueeze(0),
                            size=(input_height, input_width),
                            mode='bilinear',
                            align_corners=False
                        )[0, 0].cpu().numpy()
                        
                        axes[0, 1].imshow(drivable_map_vis, cmap='viridis')
                        axes[0, 1].set_title(f'Drivable Space Prediction\n(downsampled from {drivable_map.shape[0]}x{drivable_map.shape[1]})')
                    else:
                        # Use original size if not larger than input
                        axes[0, 1].imshow(drivable_map.cpu().numpy(), cmap='viridis')
                        axes[0, 1].set_title('Drivable Space Prediction')
                    
                    axes[0, 1].axis('off')
                
                # Plot reconstruction if available (denormalized)
                if center_recon is not None:
                    center_recon_denorm = denormalize(center_recon[i])
                    axes[1, 0].imshow(center_recon_denorm.permute(1, 2, 0).cpu().numpy())
                    axes[1, 0].set_title('Center Reconstruction')
                    axes[1, 0].axis('off')
                
                # Empty plot for the fourth position or use for additional info
                axes[1, 1].axis('off')
                
                # Save figure
                fig.tight_layout()
                plt.savefig(output_dir / f'sample_{samples_visualized}.png')
                plt.close(fig)
                
                samples_visualized += 1
                if samples_visualized >= num_samples:
                    break


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize model predictions from a checkpoint')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Optional arguments with defaults
    parser.add_argument('--config_path', type=str, default='config',
                        help='Path to configuration directory')
    parser.add_argument('--config_name', type=str, default='config',
                        help='Name of configuration file')
    parser.add_argument('--data_dir', type=str, default='datasets/argoversev2',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Path to save visualization images')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    
    return parser.parse_args()


def seed_everything(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main function to run visualization"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Initialize Hydra without using the decorator
    try:
        with hydra.initialize(version_base=None, config_path=args.config_path):
            hydra_config = hydra.compose(config_name=args.config_name)
            config = OmegaConf.to_container(hydra_config, resolve=True)
    except Exception as e:
        logger.error(f"Failed to load Hydra config: {e}")
        # Try to load config directly from YAML if Hydra fails
        try:
            config_file = Path(args.config_path) / f"{args.config_name}.yaml"
            config = load_config(config_file)
        except Exception as e2:
            logger.error(f"Failed to load config from YAML: {e2}")
            sys.exit(1)
    
    logger.info(f"Configuration loaded")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info(f"Using CPU")
    
    # Check if this is a DeepSpeed checkpoint
    checkpoint_path = Path(args.checkpoint)
    is_deepspeed_checkpoint = (
        'zero_pp_rank' in checkpoint_path.name or 
        'mp_rank' in checkpoint_path.name or
        # Check if it's a consolidated checkpoint in a DeepSpeed directory structure
        (checkpoint_path.name == 'pytorch_model.bin' and 
         'checkpoint_epoch' in str(checkpoint_path.parent.parent) and
         (checkpoint_path.parent.parent / 'zero_to_fp32.py').exists())
    )
    
    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    try:
        if is_deepspeed_checkpoint:
            logger.info("Detected DeepSpeed checkpoint format")
            model, checkpoint_info = load_deepspeed_checkpoint(args.checkpoint, device=device, config=config)
            logger.info(f"Model loaded from DeepSpeed checkpoint - epoch {checkpoint_info['epoch']}, loss {checkpoint_info['loss']}")
        else:
            model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device, config=config)
            logger.info(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)
    
    # Create dataset
    logger.info(f"Creating {args.split} dataset")
    dataset = DrivingDataset(
        data_dir=args.data_dir,
        split=args.split,
        seq_len=config['dataset']['seq_len'],
        img_size=config['model']['img_size'],
        random_sequence=False,  # Always deterministic for visualization
        cache_images=config['dataset'].get('cache_images', False),
        config=config,
    )
    
    logger.info(f"{args.split.capitalize()} dataset size: {len(dataset)}")
    
    # Create dataloader
    data_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Run visualization
    logger.info(f"Running visualization on {args.num_samples} samples")
    visualize_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
        output_dir=output_path,
        num_samples=args.num_samples,
        rank=0,  # Always run as main process
    )
    
    logger.info(f"Visualization complete. Images saved to {output_path}")


if __name__ == "__main__":
    main() 
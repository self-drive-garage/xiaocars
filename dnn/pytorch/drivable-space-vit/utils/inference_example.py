import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model.model import load_model_from_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Drivable Space ViT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--left_images', type=str, required=True, nargs='+',
                        help='Paths to left camera images (sequence)')
    parser.add_argument('--right_images', type=str, required=True, nargs='+',
                        help='Paths to right camera images (sequence)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save output visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--visualization', action='store_true',
                        help='Save visualization of drivable space')
    
    return parser.parse_args()

def load_image_sequence(image_paths, transform):
    """Load and preprocess a sequence of images"""
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)
    
    return torch.stack(images)

def visualize_drivable_space(image, drivable_space, output_path):
    """Visualize drivable space prediction overlaid on input image"""
    # Convert tensors to numpy arrays for visualization
    image = image.permute(1, 2, 0).cpu().numpy()
    # Denormalize image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    # Convert drivable space prediction to binary mask
    drivable_space = drivable_space.squeeze().cpu().numpy()
    
    # Create visualization with overlay
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(drivable_space, cmap='viridis')
    plt.title('Drivable Space')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Create overlay
    overlay = image.copy()
    mask = drivable_space > 0.5  # Threshold for visualization
    overlay_color = np.zeros_like(overlay)
    overlay_color[..., 1] = 1.0  # Green channel
    
    # Apply alpha blending
    alpha = 0.5
    overlay[mask] = overlay[mask] * (1 - alpha) + overlay_color[mask] * alpha
    
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f'Visualization saved to {output_path}')

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load model
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    logger.info(f'Model loaded from: {args.checkpoint}')
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    img_size = model_config.get('img_size', 224)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Check if sequences have the same length
    if len(args.left_images) != len(args.right_images):
        logger.error('Left and right image sequences must have the same length')
        return
    
    # Load images
    left_images = load_image_sequence(args.left_images, transform)
    right_images = load_image_sequence(args.right_images, transform)
    
    # Create batch
    batch = {
        'left_images': left_images.unsqueeze(0).to(device),    # Add batch dimension
        'right_images': right_images.unsqueeze(0).to(device),  # Add batch dimension
    }
    
    # Run inference
    with torch.no_grad():
        outputs = model(batch, task='drivable_space')
    
    # Get drivable space prediction
    drivable_space = outputs['drivable_space']
    
    # Save results
    logger.info(f'Drivable space prediction shape: {drivable_space.shape}')
    
    # Optional visualization
    if args.visualization:
        # Get the last input image and its prediction
        last_left_image = left_images[-1]
        last_prediction = drivable_space[0]  # Remove batch dimension
        
        # Visualize
        output_path = output_dir / 'drivable_space_visualization.png'
        visualize_drivable_space(last_left_image, last_prediction, output_path)
    
    logger.info('Inference completed successfully')

if __name__ == '__main__':
    main() 
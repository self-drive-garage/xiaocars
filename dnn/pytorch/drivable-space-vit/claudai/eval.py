#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation and Inference Script for Transformer-Based Self-Supervised Drivable Space Detection

This script provides utilities for evaluating a trained model and running inference
on new data for drivable space detection.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import from our model module
from model import (
    StereoTransformer,
    DrivingDataset,
    create_dataloader,
    load_model_for_inference,
    predict_drivable_space,
    evaluate_on_dataset,
    convert_model_to_onnx
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate drivable space detection model')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory for evaluation (optional)')
    
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to evaluate on (default: val)')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation (default: 8)')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading (default: 4)')
    
    # Single inference parameters
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Path to input directory for single sequence inference (optional)')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Path to output directory (default: output)')
    
    parser.add_argument('--viz', action='store_true',
                        help='Visualize predictions')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    parser.add_argument('--seq_len', type=int, default=5,
                        help='Sequence length (default: 5)')
    
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension (default: 768)')
    
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers (default: 12)')
    
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads (default: 12)')
    
    # Export options
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    parser.add_argument('--save_overlay', action='store_true',
                        help='Save overlay visualization of drivable space on original image')
    
    parser.add_argument('--save_video', action='store_true',
                        help='Create a video of the results')
    
    return parser.parse_args()


def run_inference_on_sequence(model, input_dir, output_dir, img_size=224, seq_len=5, save_overlay=False):
    """Run inference on a sequence of stereo images"""
    logger.info(f"Running inference on sequence in {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files from input directory
    input_dir = Path(input_dir)
    left_files = sorted([f for f in os.listdir(input_dir) if f.startswith('left_')])
    right_files = sorted([f for f in os.listdir(input_dir) if f.startswith('right_')])
    
    # Verify that we have matching pairs
    if len(left_files) != len(right_files):
        logger.warning(f"Mismatched number of left ({len(left_files)}) and right ({len(right_files)}) images")
    
    # Ensure we have enough frames
    if len(left_files) < seq_len:
        logger.error(f"Not enough frames in sequence: {len(left_files)} < {seq_len}")
        return
    
    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process sequence in windows of seq_len frames
    for i in range(len(left_files) - seq_len + 1):
        # Load sequence of images
        left_seq = []
        right_seq = []
        
        for j in range(seq_len):
            # Load left and right images
            left_img = Image.open(input_dir / left_files[i + j]).convert('RGB')
            right_img = Image.open(input_dir / right_files[i + j]).convert('RGB')
            
            # Save original size for later
            if j == seq_len - 1:  # Last frame is the one we predict for
                orig_size = left_img.size
            
            # Apply transformation
            left_img = transform(left_img)
            right_img = transform(right_img)
            
            left_seq.append(left_img)
            right_seq.append(right_img)
        
        # Stack images
        left_images = torch.stack(left_seq).unsqueeze(0)  # (1, seq_len, C, H, W)
        right_images = torch.stack(right_seq).unsqueeze(0)  # (1, seq_len, C, H, W)
        
        # Create placeholder ego motion (zeros)
        ego_motion = torch.zeros(1, seq_len, 6)
        
        # Run inference
        drivable_space = predict_drivable_space(
            model, left_images, right_images, ego_motion, device=model.device
        )
        
        # Get prediction
        ds_pred = drivable_space[0, 0].cpu().numpy()
        
        # Normalize for visualization
        ds_norm = (ds_pred - ds_pred.min()) / (ds_pred.max() - ds_pred.min())
        
        # Save prediction
        frame_idx = i + seq_len - 1  # Index of last frame in sequence
        np.save(os.path.join(output_dir, f"pred_{frame_idx:04d}.npy"), ds_pred)
        
        # Save visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(ds_norm, cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pred_{frame_idx:04d}.png"))
        plt.close()
        
        # Save overlay if requested
        if save_overlay:
            # Load original image
            orig_img = cv2.imread(str(input_dir / left_files[frame_idx]))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Resize prediction to match original image
            ds_resized = cv2.resize(ds_norm, orig_size)
            
            # Create colormap
            colormap = plt.get_cmap('viridis')
            ds_color = (colormap(ds_resized) * 255).astype(np.uint8)[:, :, :3]
            
            # Create overlay
            overlay = cv2.addWeighted(orig_img, 0.7, ds_color, 0.3, 0)
            
            # Save overlay
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"overlay_{frame_idx:04d}.png"))
            plt.close()
    
    logger.info(f"Inference completed. Results saved to {output_dir}")


def create_video_from_images(image_dir, output_file, pattern="overlay_*.png", fps=10):
    """Create a video from a sequence of images"""
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob(pattern)))
    
    if not image_files:
        logger.error(f"No images found matching pattern {pattern} in {image_dir}")
        return False
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    height, width, layers = first_img.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add each image to video
    for img_file in tqdm(image_files, desc="Creating video"):
        img = cv2.imread(str(img_file))
        video.write(img)
    
    # Release video writer
    video.release()
    
    logger.info(f"Video saved to {output_file}")
    return True


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model_for_inference(args.model, device)
    
    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameter count: {param_count / 1e6:.2f}M")
    
    # Export model to ONNX if requested
    if args.export_onnx:
        onnx_path = output_dir / 'model.onnx'
        convert_model_to_onnx(
            model, 
            str(onnx_path), 
            img_size=args.img_size, 
            batch_size=1, 
            seq_len=args.seq_len
        )
    
    # Run evaluation on dataset if provided
    if args.data_dir:
        logger.info(f"Evaluating model on {args.split} split of {args.data_dir}")
        
        # Create dataset
        eval_dataset = DrivingDataset(
            data_dir=args.data_dir,
            split=args.split,
            seq_len=args.seq_len,
            img_size=args.img_size,
        )
        
        # Create data loader
        eval_loader = create_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        
        # Evaluate model
        eval_dir = output_dir / 'evaluation' if args.viz else None
        evaluate_on_dataset(model, eval_loader, device, output_dir=eval_dir)
    
    # Run inference on single sequence if provided
    if args.input_dir:
        logger.info(f"Running inference on sequence in {args.input_dir}")
        
        # Create output directory for inference
        inference_dir = output_dir / 'inference'
        inference_dir.mkdir(exist_ok=True)
        
        # Run inference
        run_inference_on_sequence(
            model, 
            args.input_dir, 
            inference_dir, 
            img_size=args.img_size, 
            seq_len=args.seq_len,
            save_overlay=args.save_overlay,
        )
        
        # Create video if requested
        if args.save_video:
            logger.info("Creating result video")
            
            # Choose pattern based on whether overlay was saved
            pattern = "overlay_*.png" if args.save_overlay else "pred_*.png"
            
            # Create video
            video_path = output_dir / 'result.mp4'
            create_video_from_images(inference_dir, str(video_path), pattern=pattern)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation Scripts for Transformer-Based Self-Supervised Drivable Space Detection

This script provides utilities for converting common autonomous driving datasets
(nuScenes and Argoverse v2) to the format required by our model.
"""

import os
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Target directory structure
# dataset_root/
# ├── train_metadata.json
# ├── val_metadata.json
# ├── test_metadata.json
# │   ├── images/
# │   │   ├── seq_0001/
# │   │   │   ├── left_000001.jpg
# │   │   │   ├── right_000001.jpg
# │   │   │   ├── left_000002.jpg
# │   │   │   └── ...
# │   │   │
# │   │   ├── seq_0002/
# │   │   └── ...
# │   │
# │   └── ego_motion/
# │       ├── seq_0001.json
# │       ├── seq_0002.json
# │       └── ...


def ensure_directory(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path


def process_image(src_path, dst_path, size=None):
    """Process and save an image, optionally resizing it."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Load and process image
        with Image.open(src_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if specified
            if size is not None:
                img = img.resize(size, Image.LANCZOS)
            
            # Save the processed image
            img.save(dst_path, 'JPEG', quality=95)
        
        return True
    except Exception as e:
        logger.error(f"Error processing image {src_path}: {e}")
        return False


def process_sequence_images(seq_data, src_root, dst_root, size=None, num_workers=4):
    """Process all images in a sequence using parallel processing."""
    # Prepare image processing tasks
    tasks = []
    for frame in seq_data['frames']:
        for camera in ['left', 'right']:
            src_path = os.path.join(src_root, frame[f'{camera}_path'])
            dst_path = os.path.join(dst_root, 'images', seq_data['id'], f"{camera}_{frame['timestamp']}.jpg")
            tasks.append((src_path, dst_path, size))
    
    # Process images in parallel
    with mp.Pool(num_workers) as pool:
        process_func = partial(process_image, size=size)
        results = list(tqdm(pool.starmap(process_func, [(t[0], t[1]) for t in tasks]), 
                          total=len(tasks), 
                          desc=f"Processing images for sequence {seq_data['id']}"))
    
    # Return number of successfully processed images
    return sum(results)


####################################
# Custom Dataset Creation Functions
####################################

def create_custom_dataset(src_dir, output_root, split_ratio=(0.8, 0.1, 0.1), img_size=None, num_workers=4):
    """
    Create a custom dataset from a directory of stereo image sequences.
    
    Expected directory structure:
    src_dir/
    ├── sequence_001/
    │   ├── left_0001.jpg
    │   ├── right_0001.jpg
    │   ├── left_0002.jpg
    │   ├── right_0002.jpg
    │   └── ...
    ├── sequence_002/
    └── ...
    
    Args:
        src_dir: Source directory containing sequence directories
        output_root: Output directory
        split_ratio: Tuple (train, val, test) ratio
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers for image processing
    """
    logger.info(f"Creating custom dataset from {src_dir} to {output_root}")
    
    # Validate split ratio
    assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"
    
    # Create output directory for ego motion
    ensure_directory(os.path.join(output_root, 'ego_motion'))
    
    # Get sequence directories
    sequence_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    if not sequence_dirs:
        logger.error(f"No sequence directories found in {src_dir}")
        return False
    
    logger.info(f"Found {len(sequence_dirs)} sequence directories")
    
    # Randomize sequence order
    np.random.shuffle(sequence_dirs)
    
    # Split sequences into train/val/test
    n_train = int(len(sequence_dirs) * split_ratio[0])
    n_val = int(len(sequence_dirs) * split_ratio[1])
    
    train_sequences = sequence_dirs[:n_train]
    val_sequences = sequence_dirs[n_train:n_train+n_val]
    test_sequences = sequence_dirs[n_train+n_val:]
    
    splits = {
        'train': train_sequences,
        'val': val_sequences,
        'test': test_sequences
    }
    
    logger.info(f"Split sequences: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")
    
    # Process each split
    for split_name, sequences in splits.items():
        metadata = {'sequences': []}
        
        for seq_name in tqdm(sequences, desc=f"Processing {split_name} sequences"):
            seq_dir = os.path.join(src_dir, seq_name)
            
            # Get image files
            files = sorted(os.listdir(seq_dir))
            left_files = [f for f in files if f.startswith('left_')]
            right_files = [f for f in files if f.startswith('right_')]
            
            # Match left and right frames
            left_timestamps = [int(f.split('_')[1].split('.')[0]) for f in left_files]
            right_timestamps = [int(f.split('_')[1].split('.')[0]) for f in right_files]
            
            # Find common timestamps
            common_timestamps = sorted(list(set(left_timestamps).intersection(set(right_timestamps))))
            
            if len(common_timestamps) < 5:
                logger.warning(f"Skipping sequence {seq_name} with only {len(common_timestamps)} paired frames")
                continue
            
            # Create sequence data
            sequence_id = f"seq_{seq_name}"
            sequence_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Initialize ego motion data
            # For custom datasets, if no ego motion is available, we create placeholder data
            ego_motion_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Process frames
            for i, timestamp in enumerate(common_timestamps):
                left_idx = left_timestamps.index(timestamp)
                right_idx = right_timestamps.index(timestamp)
                
                # Get absolute paths to original images
                left_path = os.path.abspath(os.path.join(seq_dir, left_files[left_idx]))
                right_path = os.path.abspath(os.path.join(seq_dir, right_files[right_idx]))
                
                # Generate placeholder ego motion if not available
                # In a real application, you would read this from sensors or compute from visual odometry
                if i > 0:
                    # Placeholder values - in real data, compute from actual measurements
                    speed = 5.0  # m/s
                    steering_angle = 0.02  # radians
                    accel = {'x': 0.1, 'y': 0.0, 'z': 0.05}
                    angular_vel = {'x': 0.0, 'y': 0.0, 'z': 0.01}
                else:
                    speed = 0.0
                    steering_angle = 0.0
                    accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    angular_vel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                
                # Store ego motion data
                ego_motion_data['frames'].append({
                    'timestamp': timestamp,
                    'speed': speed,
                    'steering_angle': steering_angle,
                    'acceleration': accel,
                    'angular_velocity': angular_vel
                })
                
                # Store frame data with absolute paths to original images
                frame_data = {
                    'left_image_path': left_path,
                    'right_image_path': right_path,
                    'timestamp': timestamp,
                    'speed': speed,
                    'acceleration': accel,
                    'steering_angle': steering_angle,
                    'angular_velocity': angular_vel
                }
                
                sequence_data['frames'].append(frame_data)
            
            # Add sequence to metadata
            metadata['sequences'].append(sequence_data)
            
            # Save ego motion data
            with open(os.path.join(output_root, 'ego_motion', f"{sequence_id}.json"), 'w') as f:
                json.dump(ego_motion_data, f, indent=2)
        
        # Save metadata
        with open(os.path.join(output_root, f"{split_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(metadata['sequences'])} sequences for {split_name} split")
    
    return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert datasets for self-supervised drivable space detection')
    
    parser.add_argument('--dataset_type', type=str, choices=['nuscenes', 'argoverse2', 'custom'], required=True,
                        help='Type of dataset to convert')
    
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to source dataset directory (base directory for dataset)')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for metadata and ego motion data')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers for processing')
    
    parser.add_argument('--nuscenes_can_dir', type=str, default=None,
                        help='Path to nuScenes CAN bus data directory (if different from source_dir)')
    
    parser.add_argument('--nuscenes_version', type=str, default='v1.0-mini',
                        help='NuScenes dataset version (default: v1.0-mini)')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Set up logging to file
    log_file = os.path.join(args.output_dir, f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Converting {args.dataset_type} dataset from {args.source_dir} to {args.output_dir}")
    logger.info(f"Using pointers to original image files instead of copying images")
    
    # Convert based on dataset type
    if args.dataset_type == 'nuscenes':

        # Import the converter modules
        try:
            from nuscenes_converter import convert_nuscenes
        except ImportError:
            # Handle the case where the module might be in the same directory
            from .nuscenes_converter import convert_nuscenes
            
        # For nuScenes, determine the CAN data directory (if specified)
        nuscenes_can_dir = args.nuscenes_can_dir if args.nuscenes_can_dir else args.source_dir
        
        logger.info(f"Using nuScenes dataset version: {args.nuscenes_version}")
        logger.info(f"Using nuScenes CAN data from: {nuscenes_can_dir}")
        
        # Inject the appropriate paths into the convert_nuscenes function
        def nusc_convert_with_paths():
            # Monkey patch the NuScenesCanBus and NuScenes initialization in the convert_nuscenes function
            from nuscenes.nuscenes import NuScenes
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus
            original_nuscenes_init = NuScenes.__init__
            original_can_bus_init = NuScenesCanBus.__init__
            
            # Override the CAN bus API initialization to use the specified CAN directory
            def patched_can_bus_init(self, dataroot=None, **kwargs):
                return original_can_bus_init(self, dataroot=nuscenes_can_dir, **kwargs)
            
            # Override the NuScenes initialization to use the specified version and source directory
            def patched_nuscenes_init(self, version=None, dataroot=None, **kwargs):
                return original_nuscenes_init(self, version=args.nuscenes_version, dataroot=args.source_dir, **kwargs)
            
            # Apply the patches
            NuScenesCanBus.__init__ = patched_can_bus_init
            NuScenes.__init__ = patched_nuscenes_init
            
            # Call the convert function
            result = convert_nuscenes(
                args.source_dir,
                args.output_dir,
                num_workers=args.num_workers
            )
            
            # Restore original implementations
            NuScenesCanBus.__init__ = original_can_bus_init
            NuScenes.__init__ = original_nuscenes_init
            
            return result
        
        nusc_convert_with_paths()
    elif args.dataset_type == 'argoverse2':

        try:
            from argoverse_converter import convert_argoverse2
        except ImportError:
            # Handle the case where the module might be in the same directory
            from .argoverse_converter import convert_argoverse2

        convert_argoverse2(
            args.source_dir,
            args.output_dir,
            num_workers=args.num_workers
        )
    elif args.dataset_type == 'custom':
        create_custom_dataset(
            args.source_dir,
            args.output_dir,
            num_workers=args.num_workers
        )
    else:
        logger.error(f"Unknown dataset type: {args.dataset_type}")
        return
    
    logger.info("Dataset conversion completed")
    logger.info(f"Metadata and ego motion data saved to {args.output_dir}")
    logger.info("Image files remain in their original locations and are referenced by absolute paths")


if __name__ == "__main__":
    main()
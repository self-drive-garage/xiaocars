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
# ├── images/
# │   ├── seq_0001/
# │   │   ├── left_000001.jpg
# │   │   ├── right_000001.jpg
# │   │   ├── left_000002.jpg
# │   │   └── ...
# │   ├── seq_0002/
# │   └── ...
# └── ego_motion/
#     ├── seq_0001.json
#     ├── seq_0002.json
#     └── ...


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


#############################
# nuScenes Dataset Conversion
#############################

def convert_nuscenes(nuscenes_root, output_root, splits=None, img_size=None, num_workers=4):
    """
    Convert nuScenes dataset to the required format.
    
    Args:
        nuscenes_root: Path to nuScenes dataset root
        output_root: Path to output directory
        splits: Dictionary mapping 'train', 'val', 'test' to lists of scene tokens
               If None, use default nuScenes splits
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers for image processing
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError:
        logger.error("nuScenes API not found. Please install it with: pip install nuscenes-devkit")
        return False
    
    logger.info(f"Converting nuScenes dataset from {nuscenes_root} to {output_root}")
    
    # Initialize nuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)
    
    # Get default splits if not provided
    if splits is None:
        splits_scenes = create_splits_scenes()
        splits = {
            'train': splits_scenes['train'],
            'val': splits_scenes['val'],
            'test': splits_scenes['test']
        }
    
    # Create output directories
    ensure_directory(os.path.join(output_root, 'images'))
    ensure_directory(os.path.join(output_root, 'ego_motion'))
    
    # Process each split
    for split_name, scene_names in splits.items():
        logger.info(f"Processing {split_name} split with {len(scene_names)} scenes")
        
        # Filter scenes
        scenes = [scene for scene in nusc.scene if scene['name'] in scene_names]
        
        # Initialize metadata
        metadata = {'sequences': []}
        
        # Process each scene
        for scene in tqdm(scenes, desc=f"Processing {split_name} scenes"):
            # Get first sample
            sample_token = scene['first_sample_token']
            
            # Initialize sequence data
            sequence_id = f"scene_{scene['token'][-8:]}"
            sequence_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Initialize ego motion data
            ego_motion_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Process all samples in the scene
            while sample_token:
                sample = nusc.get('sample', sample_token)
                
                # Get front camera data (CAM_FRONT)
                cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                cam_front_left_data = nusc.get('sample_data', sample['data']['CAM_FRONT_LEFT'])
                cam_front_right_data = nusc.get('sample_data', sample['data']['CAM_FRONT_RIGHT'])
                
                # Use CAM_FRONT_LEFT and CAM_FRONT_RIGHT as our stereo pair
                # This isn't ideal as they're not a true stereo pair, but it's the closest approximation
                left_path = cam_front_left_data['filename']
                right_path = cam_front_right_data['filename']
                
                # Get ego pose data
                cam_front_ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
                
                # Get timestamp
                timestamp = int(cam_front_data['timestamp'])
                
                # Compute ego motion data
                # For real applications, you'd want to transform velocities to camera frame
                # and include angular velocities by differentiating orientation
                if len(ego_motion_data['frames']) > 0:
                    prev_pose = ego_motion_data['frames'][-1]['pose']
                    time_diff = (timestamp - prev_pose['timestamp']) / 1e6  # Convert to seconds
                    
                    # Compute translational velocity
                    trans_vel = [
                        (cam_front_ego_pose['translation'][i] - prev_pose['translation'][i]) / time_diff
                        for i in range(3)
                    ]
                    
                    # Simple steering angle approximation (heading change)
                    heading_change = prev_pose['rotation'][0] - cam_front_ego_pose['rotation'][0]
                    steering_angle = heading_change / time_diff if time_diff > 0 else 0
                    
                    # Compute approximate acceleration (finite difference)
                    accel = [0, 0, 0]
                    if len(ego_motion_data['frames']) > 1:
                        prev_vel = ego_motion_data['frames'][-1]['velocity']
                        accel = [
                            (trans_vel[i] - prev_vel[i]) / time_diff
                            for i in range(3)
                        ]
                    
                    # Compute approximate angular velocity
                    # For simplicity, we'll only use yaw rate
                    angular_vel = [0, 0, heading_change / time_diff if time_diff > 0 else 0]
                else:
                    trans_vel = [0, 0, 0]
                    accel = [0, 0, 0]
                    steering_angle = 0
                    angular_vel = [0, 0, 0]
                
                # Store ego motion data
                ego_motion_data['frames'].append({
                    'timestamp': timestamp,
                    'pose': {
                        'translation': cam_front_ego_pose['translation'],
                        'rotation': cam_front_ego_pose['rotation'],
                        'timestamp': timestamp
                    },
                    'velocity': trans_vel,
                    'acceleration': accel,
                    'steering_angle': steering_angle,
                    'angular_velocity': angular_vel
                })
                
                # Store frame data
                frame_data = {
                    'left_image_path': f"images/{sequence_id}/left_{timestamp}.jpg",
                    'right_image_path': f"images/{sequence_id}/right_{timestamp}.jpg",
                    'left_path': left_path,
                    'right_path': right_path,
                    'timestamp': timestamp,
                    'speed': np.linalg.norm(trans_vel),
                    'acceleration': {
                        'x': accel[0],
                        'y': accel[1],
                        'z': accel[2]
                    },
                    'steering_angle': steering_angle,
                    'angular_velocity': {
                        'x': angular_vel[0],
                        'y': angular_vel[1],
                        'z': angular_vel[2]
                    }
                }
                
                sequence_data['frames'].append(frame_data)
                
                # Move to next sample
                sample_token = sample['next']
            
            # Only include sequences with at least 5 frames
            if len(sequence_data['frames']) >= 5:
                # Add sequence to metadata
                metadata['sequences'].append(sequence_data)
                
                # Save ego motion data
                with open(os.path.join(output_root, 'ego_motion', f"{sequence_id}.json"), 'w') as f:
                    json.dump(ego_motion_data, f, indent=2)
                
                # Process sequence images
                process_sequence_images(
                    sequence_data, 
                    nuscenes_root, 
                    output_root, 
                    size=img_size,
                    num_workers=num_workers
                )
            else:
                logger.warning(f"Skipping sequence {sequence_id} with only {len(sequence_data['frames'])} frames")
        
        # Save metadata
        with open(os.path.join(output_root, f"{split_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(metadata['sequences'])} sequences for {split_name} split")
    
    return True


#################################
# Argoverse v2 Dataset Conversion
#################################

def convert_argoverse2(argoverse_root, output_root, splits=None, img_size=None, num_workers=4):
    """
    Convert Argoverse 2 dataset to the required format.
    
    Args:
        argoverse_root: Path to Argoverse 2 dataset root
        output_root: Path to output directory
        splits: Dictionary mapping 'train', 'val', 'test' to lists of log IDs
                If None, use default Argoverse 2 splits
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers for image processing
    """
    try:
        from av2.datasets.sensor.sensor_dataloader import SensorDataLoader
        from av2.utils.io import read_feather
    except ImportError:
        logger.error("Argoverse 2 API not found. Please install it with: pip install av2")
        return False
    
    logger.info(f"Converting Argoverse 2 dataset from {argoverse_root} to {output_root}")
    
    # Set up path to sensor dataset
    sensor_path = os.path.join(argoverse_root, 'sensor')
    
    # Get default splits if not provided
    if splits is None:
        # Argoverse 2 has train/val/test splits defined by directory structure
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_path = os.path.join(sensor_path, split_name)
            if os.path.exists(split_path):
                splits[split_name] = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    
    # Create output directories
    ensure_directory(os.path.join(output_root, 'images'))
    ensure_directory(os.path.join(output_root, 'ego_motion'))
    
    # Process each split
    for split_name, log_ids in splits.items():
        logger.info(f"Processing {split_name} split with {len(log_ids)} logs")
        
        # Initialize metadata
        metadata = {'sequences': []}
        
        # Process each log
        for log_id in tqdm(log_ids, desc=f"Processing {split_name} logs"):
            # Initialize sensor dataloader
            log_dir = os.path.join(sensor_path, split_name, log_id)
            if not os.path.exists(log_dir):
                logger.warning(f"Log directory not found: {log_dir}")
                continue
            
            try:
                dataloader = SensorDataLoader(log_dir, with_camera=True)
            except Exception as e:
                logger.error(f"Error loading log {log_id}: {e}")
                continue
            
            # Initialize sequence data
            sequence_id = f"log_{log_id}"
            sequence_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Initialize ego motion data
            ego_motion_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Get camera names
            # Argoverse 2 has ring cameras in array: ['ring_front_center', 'ring_front_right', ...]
            # We'll use front_left and front_right as our stereo pair
            camera_names = dataloader.get_camera_names()
            left_camera = 'ring_front_left'
            right_camera = 'ring_front_right'
            
            if left_camera not in camera_names or right_camera not in camera_names:
                logger.warning(f"Required cameras not found in log {log_id}. Available: {camera_names}")
                continue
            
            # Get synchronized timestamps
            cam_timestamps = dataloader.get_cam_timestamps(camera_names)
            
            # Filter for timestamps where both cameras have data
            left_timestamps = set(cam_timestamps[left_camera])
            right_timestamps = set(cam_timestamps[right_camera])
            common_timestamps = sorted(list(left_timestamps.intersection(right_timestamps)))
            
            # Process each timestamp
            for i, timestamp in enumerate(common_timestamps):
                # Get image paths
                left_path = dataloader.get_image_path(left_camera, timestamp)
                right_path = dataloader.get_image_path(right_camera, timestamp)
                
                # Get ego vehicle pose
                pose_data = dataloader.get_city_SE3_ego(timestamp)
                
                # Compute ego motion
                if i > 0:
                    prev_timestamp = common_timestamps[i-1]
                    prev_pose = dataloader.get_city_SE3_ego(prev_timestamp)
                    
                    # Time difference in seconds
                    time_diff = (timestamp - prev_timestamp) / 1e9  # Convert nanoseconds to seconds
                    
                    # Compute translation difference
                    trans_diff = pose_data.translation - prev_pose.translation
                    
                    # Compute translational velocity
                    trans_vel = trans_diff / time_diff if time_diff > 0 else np.zeros(3)
                    
                    # Get rotation difference
                    rot_diff = prev_pose.inverse().compose(pose_data).rotation.as_euler("xyz", degrees=False)
                    
                    # Approximate steering angle (yaw rate)
                    steering_angle = rot_diff[2] / time_diff if time_diff > 0 else 0
                    
                    # Compute angular velocity
                    angular_vel = rot_diff / time_diff if time_diff > 0 else np.zeros(3)
                    
                    # Compute acceleration
                    accel = np.zeros(3)
                    if i > 1:
                        prev_prev_timestamp = common_timestamps[i-2]
                        prev_prev_pose = dataloader.get_city_SE3_ego(prev_prev_timestamp)
                        
                        prev_time_diff = (prev_timestamp - prev_prev_timestamp) / 1e9
                        prev_trans_diff = prev_pose.translation - prev_prev_pose.translation
                        prev_vel = prev_trans_diff / prev_time_diff if prev_time_diff > 0 else np.zeros(3)
                        
                        accel = (trans_vel - prev_vel) / time_diff if time_diff > 0 else np.zeros(3)
                else:
                    trans_vel = np.zeros(3)
                    steering_angle = 0
                    angular_vel = np.zeros(3)
                    accel = np.zeros(3)
                
                # Compute speed (magnitude of velocity)
                speed = np.linalg.norm(trans_vel)
                
                # Store ego motion data
                ego_motion_data['frames'].append({
                    'timestamp': timestamp,
                    'pose': {
                        'translation': pose_data.translation.tolist(),
                        'rotation': pose_data.rotation.as_quat().tolist(),
                        'timestamp': timestamp
                    },
                    'velocity': trans_vel.tolist(),
                    'acceleration': accel.tolist(),
                    'steering_angle': steering_angle,
                    'angular_velocity': angular_vel.tolist()
                })
                
                # Store frame data
                frame_data = {
                    'left_image_path': f"images/{sequence_id}/left_{timestamp}.jpg",
                    'right_image_path': f"images/{sequence_id}/right_{timestamp}.jpg",
                    'left_path': left_path,
                    'right_path': right_path,
                    'timestamp': timestamp,
                    'speed': float(speed),
                    'acceleration': {
                        'x': float(accel[0]),
                        'y': float(accel[1]),
                        'z': float(accel[2])
                    },
                    'steering_angle': float(steering_angle),
                    'angular_velocity': {
                        'x': float(angular_vel[0]),
                        'y': float(angular_vel[1]),
                        'z': float(angular_vel[2])
                    }
                }
                
                sequence_data['frames'].append(frame_data)
            
            # Only include sequences with at least 5 frames
            if len(sequence_data['frames']) >= 5:
                # Add sequence to metadata
                metadata['sequences'].append(sequence_data)
                
                # Save ego motion data
                with open(os.path.join(output_root, 'ego_motion', f"{sequence_id}.json"), 'w') as f:
                    json.dump(ego_motion_data, f, indent=2)
                
                # Process sequence images
                process_sequence_images(
                    sequence_data, 
                    argoverse_root, 
                    output_root, 
                    size=img_size,
                    num_workers=num_workers
                )
            else:
                logger.warning(f"Skipping sequence {sequence_id} with only {len(sequence_data['frames'])} frames")
        
        # Save metadata
        with open(os.path.join(output_root, f"{split_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(metadata['sequences'])} sequences for {split_name} split")
    
    return True


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
    
    # Create output directories
    ensure_directory(os.path.join(output_root, 'images'))
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
                
                left_path = os.path.join(seq_dir, left_files[left_idx])
                right_path = os.path.join(seq_dir, right_files[right_idx])
                
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
                
                # Store frame data
                frame_data = {
                    'left_image_path': f"images/{sequence_id}/left_{timestamp}.jpg",
                    'right_image_path': f"images/{sequence_id}/right_{timestamp}.jpg",
                    'left_path': left_path,
                    'right_path': right_path,
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
            
            # Process sequence images - for custom dataset, we need to adjust the process function
            for frame in tqdm(sequence_data['frames'], desc=f"Processing images for {sequence_id}"):
                # Process left image
                left_dst = os.path.join(output_root, frame['left_image_path'])
                os.makedirs(os.path.dirname(left_dst), exist_ok=True)
                process_image(frame['left_path'], left_dst, size=img_size)
                
                # Process right image
                right_dst = os.path.join(output_root, frame['right_image_path'])
                os.makedirs(os.path.dirname(right_dst), exist_ok=True)
                process_image(frame['right_path'], right_dst, size=img_size)
        
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
                        help='Path to source dataset directory')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    
    parser.add_argument('--img_size', type=int, nargs=2, default=None,
                        help='Optional image resize dimensions (width height)')
    
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                        help='Number of parallel workers for image processing')
    
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
    
    # Convert based on dataset type
    if args.dataset_type == 'nuscenes':
        convert_nuscenes(
            args.source_dir,
            args.output_dir,
            img_size=args.img_size,
            num_workers=args.num_workers
        )
    elif args.dataset_type == 'argoverse2':
        convert_argoverse2(
            args.source_dir,
            args.output_dir,
            img_size=args.img_size,
            num_workers=args.num_workers
        )
    elif args.dataset_type == 'custom':
        create_custom_dataset(
            args.source_dir,
            args.output_dir,
            img_size=args.img_size,
            num_workers=args.num_workers
        )
    else:
        logger.error(f"Unknown dataset type: {args.dataset_type}")
        return
    
    logger.info("Dataset conversion completed")


if __name__ == "__main__":
    main()

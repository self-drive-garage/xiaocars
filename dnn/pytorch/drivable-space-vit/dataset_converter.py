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
from pyquaternion import Quaternion

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
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus
        
        # Initialize CAN bus API if available
        can_bus_available = True
        try:
            nusc_can = NuScenesCanBus(dataroot="/datasets/nuscenes")
        except Exception as e:
            logger.warning(f"CAN bus data not available: {e}")
            can_bus_available = False
    except ImportError:
        logger.error("nuScenes API not found. Please install it with: pip install nuscenes-devkit")
        return False
    
    logger.info(f"Converting nuScenes dataset from {nuscenes_root} to {output_root}")
    
    # Initialize nuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot="/datasets/nuscenes/v1.0-mini", verbose=True)
    
    # Get default splits if not provided
    if splits is None:
        splits_scenes = create_splits_scenes()
        splits = {
            'train': splits_scenes['train'],
            'val': splits_scenes['val'],
            'test': splits_scenes['test']
        }
    
    # Create output directory for ego motion data
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
            
            # Initialize ego motion data (for detailed ego motion, saved separately)
            ego_motion_data = {
                'id': sequence_id,
                'frames': []
            }
            
            # Get CAN bus data for this scene if available
            vehicle_monitor_data = None
            pose_data = None
            if can_bus_available:
                try:
                    # Get vehicle monitor data (for steering angle)
                    vehicle_monitor_data = nusc_can.get_messages(scene['name'], 'vehicle_monitor')
                    
                    # Get pose data (50Hz data with velocity, acceleration, etc.)
                    pose_data = nusc_can.get_messages(scene['name'], 'pose')
                    if pose_data:
                        logger.info(f"Pose data available for scene {scene['name']} ({len(pose_data)} samples)")
                    
                    logger.info(f"CAN bus data available for scene {scene['name']}")
                except Exception as e:
                    logger.warning(f"Failed to get CAN bus data for scene {scene['name']}: {e}")
            
            # Process all samples in the scene
            while sample_token:
                sample = nusc.get('sample', sample_token)
                
                # Get front camera data (CAM_FRONT)
                cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                cam_front_left_data = nusc.get('sample_data', sample['data']['CAM_FRONT_LEFT'])
                cam_front_right_data = nusc.get('sample_data', sample['data']['CAM_FRONT_RIGHT'])
                
                # Use CAM_FRONT_LEFT and CAM_FRONT_RIGHT as our stereo pair
                # This isn't ideal as they're not a true stereo pair, but it's the closest approximation
                left_path = os.path.join(nuscenes_root, cam_front_left_data['filename'])
                right_path = os.path.join(nuscenes_root, cam_front_right_data['filename'])
                
                # Get ego pose data
                cam_front_ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
                
                # Get timestamp
                timestamp = int(cam_front_data['timestamp'])
                
                # Method 1: Try to get data from 50Hz pose data (preferred)
                trans_vel = [0, 0, 0]
                accel = [0, 0, 0]
                angular_vel = [0, 0, 0]
                steering_angle = 0
                used_pose_data = False
                
                if pose_data is not None:
                    # Find the closest pose data to this timestamp
                    closest_pose_idx = nusc_can.get_closest_message_idx(pose_data, timestamp)
                    
                    if closest_pose_idx is not None:
                        closest_pose = pose_data[closest_pose_idx]
                        # Check if the pose timestamp is close enough to our camera timestamp
                        time_diff_ms = abs(closest_pose['utime'] - timestamp) / 1000  # Convert to ms
                        
                        if time_diff_ms < 25:  # Within 25ms (50Hz data means ~20ms between samples)
                            # Get velocity, acceleration and rotation rate directly from pose data
                            trans_vel = closest_pose['vel']
                            accel = closest_pose['accel']
                            angular_vel = closest_pose['rotation_rate']
                            
                            # Note: pose data doesn't have steering angle, we'll get that separately
                            used_pose_data = True
                            logger.debug(f"Using pose data for velocity and acceleration")
                
                # Method 2: Fall back to calculating from ego poses if pose data not available
                if not used_pose_data and len(ego_motion_data['frames']) > 0:
                    prev_pose = ego_motion_data['frames'][-1]['pose']
                    time_diff = (timestamp - prev_pose['timestamp']) / 1e6  # Convert to seconds
                    
                    # Compute translational velocity
                    trans_vel = [
                        (cam_front_ego_pose['translation'][i] - prev_pose['translation'][i]) / time_diff
                        for i in range(3)
                    ]
                    
                    # Compute approximate acceleration (finite difference)
                    accel = [0, 0, 0]
                    if len(ego_motion_data['frames']) > 1:
                        prev_vel = ego_motion_data['frames'][-1]['velocity']
                        accel = [
                            (trans_vel[i] - prev_vel[i]) / time_diff
                            for i in range(3)
                        ]
                    
                    # Compute approximate angular velocity using quaternion difference
                    prev_quat = Quaternion(prev_pose['rotation'])
                    curr_quat = Quaternion(cam_front_ego_pose['rotation'])
                    quat_diff = prev_quat.inverse * curr_quat
                    roll, pitch, yaw = quat_diff.yaw_pitch_roll
                    angular_vel = [
                        roll / time_diff if time_diff > 0 else 0,
                        pitch / time_diff if time_diff > 0 else 0,
                        yaw / time_diff if time_diff > 0 else 0
                    ]
                    
                    logger.debug(f"Using differential calculation for velocity and acceleration")
                
                # Get steering angle - first try CAN bus data, then fall back to quaternion method
                steering_angle = 0.0
                steering_angle_source = "default"
                
                # Method 1: Try to get steering angle from vehicle_monitor CAN bus data if available
                if vehicle_monitor_data is not None:
                    closest_vehicle_idx = nusc_can.get_closest_message_idx(vehicle_monitor_data, timestamp)
                    if closest_vehicle_idx is not None:
                        # Convert steering angle from degrees to radians
                        steering_angle_deg = vehicle_monitor_data[closest_vehicle_idx]['steering']
                        steering_angle = np.radians(steering_angle_deg)
                        steering_angle_source = "can_bus"
                        logger.debug(f"Using CAN bus steering angle: {steering_angle} rad")
                
                # Method 2: If CAN bus data not available, use quaternion-based heading change
                if steering_angle_source == "default" and len(ego_motion_data['frames']) > 0:
                    prev_pose = ego_motion_data['frames'][-1]['pose']
                    time_diff = (timestamp - prev_pose['timestamp']) / 1e6  # Convert to seconds
                    
                    # Convert quaternions to Euler angles to get proper heading change
                    prev_quat = Quaternion(prev_pose['rotation'])
                    curr_quat = Quaternion(cam_front_ego_pose['rotation'])
                    heading_change = (prev_quat.inverse * curr_quat).yaw_pitch_roll[0]  # This gives heading change in radians
                    steering_angle = heading_change / time_diff if time_diff > 0 else 0
                    steering_angle_source = "quaternion"
                    logger.debug(f"Using quaternion-based steering angle: {steering_angle} rad")
                
                # Calculate speed (magnitude of velocity)
                speed = float(np.linalg.norm(trans_vel))
                
                # Store ego motion data (detailed version for ego_motion directory)
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
                    'angular_velocity': angular_vel,
                    'data_source': {
                        'motion': 'pose_data' if used_pose_data else 'calculated',
                        'steering': steering_angle_source
                    }
                })
                
                # Create frame data for metadata.json using absolute paths to original images
                frame_data = {
                    'left_image_path': left_path,
                    'right_image_path': right_path,
                    'timestamp': timestamp,
                    'speed': speed,
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
                
                # Move to next sample
                sample_token = sample['next']
            
            # Only include sequences with at least 5 frames
            if len(sequence_data['frames']) >= 5:
                # Add sequence to metadata
                metadata['sequences'].append(sequence_data)
                
                # Save ego motion data
                with open(os.path.join(output_root, 'ego_motion', f"{sequence_id}.json"), 'w') as f:
                    json.dump(ego_motion_data, f, indent=2)
            else:
                logger.warning(f"Skipping sequence {sequence_id} with only {len(sequence_data['frames'])} frames")
        
        # Save metadata
        with open(os.path.join(output_root, f"{split_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(metadata['sequences'])} sequences for {split_name} split")
    
    return True


def process_sequence_images_nuscenes(seq_data, src_root, dst_root, size=None, num_workers=4):
    """Process all images in a sequence using parallel processing for nuScenes data."""
    # Prepare image processing tasks
    tasks = []
    for frame in seq_data['frames']:
        # Use the internal paths that we temporarily stored
        if 'left_path' in frame and 'right_path' in frame:
            left_src_path = os.path.join(src_root, frame['left_path'])
            right_src_path = os.path.join(src_root, frame['right_path'])
            
            timestamp = frame['timestamp'] if 'timestamp' in frame else 'unknown'
            sequence_id = seq_data['id']
            
            left_dst_path = os.path.join(dst_root, f"images/{sequence_id}/left_{timestamp}.jpg")
            right_dst_path = os.path.join(dst_root, f"images/{sequence_id}/right_{timestamp}.jpg")
            
            tasks.append((left_src_path, left_dst_path, size))
            tasks.append((right_src_path, right_dst_path, size))
    
    # Process images in parallel
    with mp.Pool(num_workers) as pool:
        process_func = partial(process_image, size=size)
        results = list(tqdm(pool.starmap(process_func, [(t[0], t[1]) for t in tasks]), 
                          total=len(tasks), 
                          desc=f"Processing images for sequence {seq_data['id']}"))
    
    # Return number of successfully processed images
    return sum(results)


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
    
    # Create output directory for ego motion
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
                # Get absolute image paths
                left_path = os.path.abspath(dataloader.get_image_path(left_camera, timestamp))
                right_path = os.path.abspath(dataloader.get_image_path(right_camera, timestamp))
                
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
                
                # Store frame data with absolute paths to original images
                frame_data = {
                    'left_image_path': left_path,
                    'right_image_path': right_path,
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
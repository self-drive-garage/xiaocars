#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation Scripts for Transformer-Based Self-Supervised Drivable Space Detection
NuScenes Dataset Converter
"""

import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime
from pyquaternion import Quaternion
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

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
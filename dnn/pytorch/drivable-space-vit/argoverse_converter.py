#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation Scripts for Transformer-Based Self-Supervised Drivable Space Detection
Argoverse 2 Dataset Converter
"""

import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from PIL import Image
import multiprocessing as mp
from functools import partial

try:
    from pathlib import Path
    from av2.datasets.sensor.constants import StereoCameras
    from av2.datasets.sensor.sensor_dataloader import SensorDataloader
except ImportError as e:
    logger.error(f"Argoverse 2 API not found. Please install it with: pip install av2")

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

def convert_argoverse2(argoverse_root, output_root, splits=None, img_size=None, num_workers=4):
    """
    Convert Argoverse 2 dataset to the required format using paths to original images.
    
    Args:
        argoverse_root: Path to Argoverse 2 dataset root
        output_root: Path to output directory
        splits: Dictionary mapping 'train', 'val', 'test' to lists of log IDs
                If None, use default Argoverse 2 splits
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers for image processing
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
 
    
    start_time = datetime.now()
    logger.info(f"Converting Argoverse 2 dataset from {argoverse_root} to {output_root}")
    logger.info(f"Using stereo cameras and storing original image paths in metadata")
    
    # Create output directory for ego motion data only (not images)
    ensure_directory(os.path.join(output_root, 'ego_motion'))
    
    # Define which cameras to use
    cam_names = (StereoCameras.STEREO_FRONT_LEFT, StereoCameras.STEREO_FRONT_RIGHT)
    
    # Set up paths
    data_dir = Path(argoverse_root)
    
    # Get default splits if not provided 
    if splits is None:
        # Just use directory structure for splits
        logger.info(f"Using directory structure for splits")
        splits = {}
        for split_name in ['train', 'val']:
            split_path = data_dir / split_name
            if split_path.exists():
                splits[split_name] = [d.name for d in split_path.glob("*") if d.is_dir()]
    
    # Statistics tracking
    total_logs = sum(len(log_ids) for log_ids in splits.values())
    if total_logs == 0:
        logger.error("No logs found in any split")
        return False
        
    successful_logs = 0
    total_frames = 0
    
    # Process each split
    for split_name, log_ids in splits.items():
        logger.info(f"Processing {split_name} split with {len(log_ids)} logs")
        
        # Initialize metadata
        metadata = {'sequences': []}
        
        # Initialize the SensorDataloader for the entire split
        try:
            logger.info(f"Initializing dataset for {split_name} split")
            split_path = data_dir / split_name
            
            logger.info(f"Using dataset directory: {split_path}")
            
            # Group frames by log ID
            log_frames = {}
            log_ego_data = {}
            
            # First pass: collect frames for each log
            logger.info(f"Collecting frames for {len(log_ids)} logs in {split_name} split")
            
            # Manually iterate through logs and timestamps
            for log_id in tqdm(log_ids, desc=f"Scanning {split_name} logs"):
                try:
                    # Get available timestamps for this log
                    log_path = split_path / log_id
                    if not log_path.exists():
                        logger.warning(f"Log path does not exist: {log_path}")
                        continue
                        
                    # Initialize sequence data structures
                    sequence_id = f"log_{log_id}"
                    log_frames[log_id] = {
                        'id': sequence_id,
                        'frames': []
                    }
                    log_ego_data[log_id] = {
                        'id': sequence_id,
                        'frames': []
                    }
                    
                    # Find the camera directories and timestamps
                    left_cam_dir = log_path / "sensors" / "cameras" / f"{StereoCameras.STEREO_FRONT_LEFT.value}"
                    right_cam_dir = log_path / "sensors" / "cameras" / f"{StereoCameras.STEREO_FRONT_RIGHT.value}"
                    
                    if not left_cam_dir.exists() or not right_cam_dir.exists():
                        logger.warning(f"Camera directories not found for log {log_id}")
                        continue
                        
                    # Get available timestamps (from image filenames)
                    left_timestamps = [int(p.stem) for p in left_cam_dir.glob("*.jpg")]
                    right_timestamps = [int(p.stem) for p in right_cam_dir.glob("*.jpg")]
                    
                    # Find common timestamps between left and right cameras
                    common_timestamps = sorted(list(set(left_timestamps).intersection(set(right_timestamps))))
                    
                    if not common_timestamps:
                        logger.warning(f"No common timestamps found for log {log_id}")
                        continue
                        
                    # Downsample frames to reduce density (e.g., every 10th frame)
                    # Adjust the sampling rate as needed
                    sampling_rate = 10  # Change this to control frame density
                    sampled_timestamps = common_timestamps[::sampling_rate]
                    
                    logger.info(f"Found {len(common_timestamps)} total frames, using {len(sampled_timestamps)} sampled frames for log {log_id}")
                    
                    # Process each timestamp from our downsampled list
                    for timestamp_ns in sampled_timestamps:
                        # Construct the absolute image paths
                        left_src_path = str(os.path.abspath(left_cam_dir / f"{timestamp_ns}.jpg"))
                        right_src_path = str(os.path.abspath(right_cam_dir / f"{timestamp_ns}.jpg"))
                        
                        # Skip if files don't exist
                        if not os.path.exists(left_src_path) or not os.path.exists(right_src_path):
                            continue
                        
                        # Create frame data with absolute paths to original images
                        frame_data = {
                            'left_image_path': left_src_path,  # Direct path to original image
                            'right_image_path': right_src_path,  # Direct path to original image
                            'timestamp': timestamp_ns,
                            # Use placeholder motion data for now
                            'speed': 0.0,
                            'acceleration': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'steering_angle': 0.0,
                            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        }
                        
                        log_frames[log_id]['frames'].append(frame_data)
                        
                        # Add placeholder ego motion data
                        ego_frame_data = {
                            'timestamp': timestamp_ns,
                            'pose': {
                                'translation': [0.0, 0.0, 0.0],
                                'rotation': [1.0, 0.0, 0.0, 0.0],  # Quaternion w,x,y,z
                                'timestamp': timestamp_ns
                            },
                            'velocity': [0.0, 0.0, 0.0],
                            'acceleration': [0.0, 0.0, 0.0],
                            'steering_angle': 0.0,
                            'angular_velocity': [0.0, 0.0, 0.0]
                        }
                        log_ego_data[log_id]['frames'].append(ego_frame_data)
                        
                except Exception as e:
                    logger.error(f"Error processing log {log_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Second pass: sort frames by timestamp and calculate ego motion
            logger.info(f"Processing ego motion for collected frames")
            for log_id, sequence_data in tqdm(log_frames.items(), desc=f"Processing ego motion"):
                # Sort frames by timestamp
                sequence_data['frames'].sort(key=lambda f: f['timestamp'])
                
                # Reset the ego motion data with the sorted timestamps
                log_ego_data[log_id]['frames'] = []
                
                # Process ego motion for sorted frames
                for i, frame in enumerate(sequence_data['frames']):
                    timestamp_ns = frame['timestamp']
                    
                    # Calculate ego motion based on time difference between frames
                    if i > 0:
                        prev_frame = sequence_data['frames'][i-1]
                        prev_timestamp = prev_frame['timestamp']
                        
                        # Time difference in seconds
                        time_diff = (timestamp_ns - prev_timestamp) / 1e9
                        
                        # For demonstration, let's simulate some simple motion
                        # In a real implementation, you'd compute this from sensor data or visual odometry
                        if time_diff > 1e-6:
                            # Simulate a constant forward velocity of 5 m/s
                            speed = 5.0
                            trans_vel = [0.0, speed, 0.0]  # [x, y, z] - y is forward in Argoverse
                            
                            # Simple acceleration model
                            accel = [0.0, 0.0, 0.0]
                            
                            # Simple steering model (alternating)
                            steering_angle = 0.1 * np.sin(i / 10.0)
                            
                            # Simple angular velocity
                            angular_vel = [0.0, 0.0, steering_angle]
                        else:
                            speed = 0.0
                            trans_vel = [0.0, 0.0, 0.0]
                            accel = [0.0, 0.0, 0.0]
                            steering_angle = 0.0
                            angular_vel = [0.0, 0.0, 0.0]
                    else:
                        # First frame - no motion
                        speed = 0.0
                        trans_vel = [0.0, 0.0, 0.0]
                        accel = [0.0, 0.0, 0.0]
                        steering_angle = 0.0
                        angular_vel = [0.0, 0.0, 0.0]
                    
                    # Update the frame with calculated motion data
                    frame['speed'] = speed
                    frame['acceleration'] = {
                        'x': float(accel[0]),
                        'y': float(accel[1]),
                        'z': float(accel[2])
                    }
                    frame['steering_angle'] = float(steering_angle)
                    frame['angular_velocity'] = {
                        'x': float(angular_vel[0]),
                        'y': float(angular_vel[1]),
                        'z': float(angular_vel[2])
                    }
                    
                    # Store ego motion data
                    ego_frame_data = {
                        'timestamp': timestamp_ns,
                        'pose': {
                            # Simple incremental pose based on velocity
                            'translation': [0.0, i * 0.1, 0.0],  # [x, y, z]
                            'rotation': [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z] quaternion
                            'timestamp': timestamp_ns
                        },
                        'velocity': trans_vel,
                        'acceleration': accel,
                        'steering_angle': float(steering_angle),
                        'angular_velocity': angular_vel
                    }
                    log_ego_data[log_id]['frames'].append(ego_frame_data)
            
            # Third pass: save processed logs and add to metadata
            logger.info(f"Saving processed logs")
            for log_id, sequence_data in tqdm(log_frames.items(), desc=f"Saving {split_name} logs"):
                # Only save logs with enough frames
                if len(sequence_data['frames']) >= 5:
                    # Save ego motion data
                    ego_data = log_ego_data[log_id]
                    with open(os.path.join(output_root, 'ego_motion', f"{sequence_data['id']}.json"), 'w') as f:
                        json.dump(ego_data, f, indent=2)
                    
                    # Add to metadata
                    metadata['sequences'].append(sequence_data)
                    successful_logs += 1
                    total_frames += len(sequence_data['frames'])
                    
                    logger.info(f"Saved log {log_id} with {len(sequence_data['frames'])} frames")
                else:
                    logger.warning(f"Skipping log {log_id}: Not enough frames ({len(sequence_data['frames'])})")
                
        except Exception as e:
            logger.error(f"Error processing {split_name} split: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
            
        # Save metadata for this split
        with open(os.path.join(output_root, f"{split_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(metadata['sequences'])} sequences for {split_name} split")
    
    # Calculate conversion statistics
    success_rate = successful_logs / total_logs * 100 if total_logs > 0 else 0
    elapsed_time = datetime.now() - start_time
    
    logger.info(f"Argoverse 2 conversion complete:")
    logger.info(f"  Success rate: {successful_logs}/{total_logs} logs ({success_rate:.1f}%)")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Time elapsed: {elapsed_time}")
    
    return True 
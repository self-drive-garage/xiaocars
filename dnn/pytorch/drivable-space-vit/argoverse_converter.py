#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation for Transformer-Based Self-Supervised Drivable Space Detection
Argoverse 2 Dataset Converter with Proper Ego Motion Integration
"""

import os
import json
import numpy as np
from pathlib import Path
from rich.progress import track
import logging
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from scipy.spatial.transform import Rotation
from av2.utils.typing import NDArrayByte, NDArrayInt
from collections import defaultdict
import traceback
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from av2.datasets.sensor.constants import StereoCameras
    from av2.datasets.sensor.sensor_dataloader import SensorDataloader
except ImportError as e:
    print(f"Argoverse 2 API not found. Please install it with: pip install av2")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a fixed version of SensorDataloader that properly handles iteration
class FixedSensorDataloader(SensorDataloader):
    """Fixed version of SensorDataloader that properly handles iteration.
    
    The original SensorDataloader.__next__ method doesn't check if the index
    is out of bounds, which can lead to errors when using it in a for loop.
    This class fixes that issue.
    """

    def __next__(self):
        """Return the next datum in the dataset, or raise StopIteration if done."""
        if self._ptr >= self.num_sweeps:
            # Reset the pointer and raise StopIteration to signal the end
            self._ptr = 0
            # print("StopIteration >>>>>>>>>>>>>>")
            raise StopIteration
        
        # print(f"self._ptr: {self._ptr}")
        result = self.__getitem__(self._ptr)
        self._ptr += 1
        return result
    
    def get_item_range(self, start_idx, end_idx):
        """Get a range of items from the dataset.
        
        Args:
            start_idx: Starting index (inclusive)
            end_idx: Ending index (exclusive)
            
        Returns:
            List of data items
        """
        return [self.__getitem__(i) for i in range(start_idx, min(end_idx, self.num_sweeps))]

def ensure_directory(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path

def compute_ego_motion_features(poses):
    """
    Compute rich ego motion features from pose sequence.
    
    Args:
        poses: List of pose dictionaries with translation and rotation data
        
    Returns:
        Dictionary mapping timestamps to enhanced ego motion features
    """
    
    ego_motion_features = {}
    sorted_poses = sorted(poses, key=lambda x: x['timestamp'])
    
    for i, pose in enumerate(sorted_poses):
        timestamp = pose['timestamp']
        
        # Base features from current pose
        feature = {
            'translation': pose['translation'],
            'rotation': pose['rotation'],
            'image_path': pose['image_path'],
            'camera': pose['camera'],
            'velocity': [0.0, 0.0, 0.0],
            'acceleration': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }
        
        # Calculate velocity and acceleration (if enough frames)
        if i > 0:
            prev_pose = sorted_poses[i-1]
            dt = (timestamp - prev_pose['timestamp']) / 1e9  # seconds
            
            if dt > 1e-6:  # Avoid division by zero or very small dt
                # Linear velocity
                vel = [(pose['translation'][j] - prev_pose['translation'][j]) / dt for j in range(3)]
                feature['velocity'] = vel
                
                # Angular velocity using quaternions (if rotation is represented as quaternion)
                if len(pose['rotation']) == 4:  # quaternion [x,y,z,w]
                    q1 = np.array(prev_pose['rotation'])
                    q2 = np.array(pose['rotation'])
                    
                    # Convert to rotation matrices
                    r1 = Rotation.from_quat(q1)
                    r2 = Rotation.from_quat(q2)
                    
                    # Calculate angular velocity
                    r_diff = r2 * r1.inv()
                    angles = r_diff.as_euler('xyz')
                    feature['angular_velocity'] = [a / dt for a in angles]
                
                # Calculate acceleration if possible
                if i > 1:
                    prev_timestamp = sorted_poses[i-1]['timestamp']
                    if prev_timestamp in ego_motion_features:
                        prev_vel = ego_motion_features[prev_timestamp]['velocity']
                        feature['acceleration'] = [(vel[j] - prev_vel[j]) / dt for j in range(3)]
        
        ego_motion_features[timestamp] = feature
    
    return ego_motion_features

def save_metadata_to_parquet(metadata_entries, output_root, split_name, append=False):
    """Helper function to save metadata to a parquet file."""
    parquet_file = os.path.join(output_root, f"{split_name}_metadata.parquet")
    
    # Convert list of dictionaries to pandas DataFrame
    df = pd.DataFrame(metadata_entries)
    
    # Handle list-type columns properly for parquet
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            # Convert list columns to string representation for now
            # This approach keeps the original structure and can be parsed back
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    if append and os.path.exists(parquet_file):
        # Read existing data
        existing_df = pd.read_parquet(parquet_file)
        # Append new data
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save to parquet
    df.to_parquet(parquet_file, index=False)
    
    logger.info(f"Saved metadata for {split_name} split with {len(metadata_entries)} stereo pairs to {parquet_file}")
    return parquet_file

def process_batch(batch_data, log_pose_history, stereo_pairs, split_name, argoverse_root):
    """Process a batch of data to extract pose information.
    
    Args:
        batch_data: List of data items from the dataset
        log_pose_history: Dictionary to store pose history by log ID
        stereo_pairs: Dictionary to store stereo image pairs
        split_name: Name of the dataset split
        argoverse_root: Path to Argoverse dataset root
        
    Returns:
        Set of processed log IDs
    """
    processed_log_ids = set()
    
    for datum in batch_data:
        try:
            sweep = datum.sweep
            annotations = datum.annotations
            log_id = datum.log_id
            processed_log_ids.add(log_id)  # Keep track of unique log IDs
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            synchronized_imagery = datum.synchronized_imagery
            
            if synchronized_imagery is None:
                continue
            
            # Store image data for both cameras at this timestamp
            timestamp_key = f"{log_id}_{sweep.timestamp_ns}"
            
            for cam_name, cam in synchronized_imagery.items():
                if cam_name not in (StereoCameras.STEREO_FRONT_LEFT.value, StereoCameras.STEREO_FRONT_RIGHT.value):
                    continue
                
                if (
                    cam.timestamp_ns in timestamp_city_SE3_ego_dict
                    and sweep.timestamp_ns in timestamp_city_SE3_ego_dict
                ):
                    # Get the SE3 transformation for this timestamp
                    city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam.timestamp_ns]
                    
                    # Extract rotation and translation from SE3
                    rotation = city_SE3_ego_cam_t.rotation
                    translation = city_SE3_ego_cam_t.translation
                    
                    # Get the image path
                    image_filename = f"{cam.timestamp_ns}.jpg"
                    image_path = os.path.join(f"{argoverse_root}/{split_name}/{log_id}/sensors/cameras/{cam_name}/", image_filename)
                    
                    # Store image path by camera type
                    stereo_pairs[timestamp_key][cam_name] = image_path
                    
                    # Create pose entry
                    pose_entry = {
                        'timestamp': cam.timestamp_ns,
                        'translation': translation.tolist(),
                        'rotation': [
                            [rotation[0, 0], rotation[0, 1], rotation[0, 2]],
                            [rotation[1, 0], rotation[1, 1], rotation[1, 2]],
                            [rotation[2, 0], rotation[2, 1], rotation[2, 2]]
                        ],
                        'image_path': image_path,
                        'log_id': log_id,
                        'camera': cam_name
                    }
                    
                    # Add to pose history for this log
                    log_pose_history[log_id].append(pose_entry)
        except Exception as e:
            logger.error(f"Error processing log {log_id}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    return processed_log_ids

def generate_metadata_entries(stereo_pairs, log_pose_history, ego_motion_by_log, log_ids=None):
    """Generate metadata entries from stereo pairs and pose history.
    
    Args:
        stereo_pairs: Dictionary of stereo image pairs
        log_pose_history: Dictionary of pose history by log ID
        ego_motion_by_log: Dictionary of ego motion features by log ID
        log_ids: Optional list of log IDs to process (for parallel processing)
        
    Returns:
        List of metadata entries
    """
    metadata_entries = []
    
    # Filter stereo pairs by log ID if specified
    pairs_to_process = {}
    if log_ids:
        for timestamp_key, cameras in stereo_pairs.items():
            log_id = timestamp_key.split('_')[0]
            if log_id in log_ids:
                pairs_to_process[timestamp_key] = cameras
    else:
        pairs_to_process = stereo_pairs
    
    for timestamp_key, cameras in pairs_to_process.items():
        try:
            log_id, sweep_timestamp = timestamp_key.split('_')
            
            left_image_path = cameras[StereoCameras.STEREO_FRONT_LEFT.value]
            right_image_path = cameras[StereoCameras.STEREO_FRONT_RIGHT.value]
            
            # Find the ego motion features for one of the cameras (they should be very similar)
            # We'll use the left camera's timestamp for ego motion
            ego_features = None
            
            for pose in log_pose_history[log_id]:
                if pose['camera'] == StereoCameras.STEREO_FRONT_LEFT.value and pose['image_path'] == left_image_path:
                    timestamp = pose['timestamp']
                    if log_id in ego_motion_by_log and timestamp in ego_motion_by_log[log_id]:
                        ego_features = ego_motion_by_log[log_id][timestamp]
                        break
            
            if ego_features is None:
                continue  # Skip if we can't find ego motion features
            
            # Create the metadata entry with both image paths
            metadata_entry = {
                'log_id': log_id,
                'timestamp': int(sweep_timestamp),
                'left_image_path': left_image_path,
                'right_image_path': right_image_path,
                'translation': ego_features['translation'],
                'rotation': ego_features['rotation'],
                'velocity': ego_features['velocity'],
                'acceleration': ego_features['acceleration'],
                'angular_velocity': ego_features['angular_velocity']
            }
            
            metadata_entries.append(metadata_entry)
            
        except Exception as e:
            logger.error(f"Error creating metadata entry for {timestamp_key}: {str(e)}")
    
    return metadata_entries

def convert_argoverse2(argoverse_root, output_root, splits=None, img_size=None, num_workers=4, save_frequency=100, batch_size=500):
    """
    Convert Argoverse 2 dataset with proper ego motion extraction.
    
    Args:
        argoverse_root: Path to Argoverse 2 dataset root
        output_root: Path to output directory
        splits: Dictionary mapping 'train', 'val', 'test' to lists of log IDs
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers
        save_frequency: Number of entries to process before saving to parquet (to prevent data loss)
        batch_size: Number of log items to process in each batch
    """
    split_name = "val"
    start_time = datetime.now()
    logger.info(f"Converting Argoverse 2 dataset from {argoverse_root} to {output_root}")
    
    # Create output directory
    ensure_directory(output_root)
    
    # Define which cameras to use
    cam_names = (StereoCameras.STEREO_FRONT_LEFT, StereoCameras.STEREO_FRONT_RIGHT)
    
    # Store all frames with their metadata
    metadata_entries = []
    total_entries = 0
    # Store per-log pose history for calculating ego motion
    log_pose_history = defaultdict(list)
    # Store image paths by log_id and timestamp
    stereo_pairs = defaultdict(dict)

    try:
        # Create the dataset loader with our fixed version that handles iteration properly
        dataset = FixedSensorDataloader(
            Path(argoverse_root),
            with_annotations=True,
            with_cache=False,
            cam_names=cam_names,
        )
        
        # Check if the dataset has data before iterating
        if dataset.num_sweeps == 0:
            logger.error(f"No data found in the dataset at {argoverse_root}")
            logger.error("Please check that the dataset path is correct and contains valid Argoverse 2 data")
            return

        total_sweeps = dataset.num_sweeps
        logger.info(f"Found {total_sweeps} sweeps in the dataset")
        
        # Calculate how many batches to process
        num_batches = (total_sweeps + batch_size - 1) // batch_size
        processed_log_ids = set()
        
        # Process dataset in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_sweeps)
            
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (items {start_idx} to {end_idx-1})")
            
            # Get a batch of data
            batch_data = dataset.get_item_range(start_idx, end_idx)
            
            # Process this batch and collect poses
            batch_log_ids = process_batch(batch_data, log_pose_history, stereo_pairs, split_name, argoverse_root)
            processed_log_ids.update(batch_log_ids)
            
            # If we've processed enough logs in this batch, compute ego motion and save data
            if len(batch_log_ids) > 0:
                # Compute ego motion for the logs in this batch
                batch_ego_motion = {}
                for log_id in batch_log_ids:
                    try:
                        poses = log_pose_history[log_id]
                        # Skip if no poses for this log
                        if not poses:
                            continue
                            
                        # Convert rotation matrices to quaternions for easier computation
                        for pose in poses:
                            r = Rotation.from_matrix(pose['rotation'])
                            # Store as [x, y, z, w] format
                            pose['rotation'] = r.as_quat().tolist()  # [x, y, z, w]
                        
                        # Compute ego motion features
                        ego_motion_features = compute_ego_motion_features(poses)
                        batch_ego_motion[log_id] = ego_motion_features
                    except Exception as e:
                        logger.error(f"Error computing ego motion for log {log_id}: {str(e)}")
                        logger.debug(traceback.format_exc())
                
                # Generate metadata entries for this batch
                batch_entries = generate_metadata_entries(stereo_pairs, log_pose_history, batch_ego_motion, batch_log_ids)
                
                # Save this batch if we have entries
                if batch_entries:
                    save_metadata_to_parquet(batch_entries, output_root, split_name, append=(total_entries > 0))
                    total_entries += len(batch_entries)
                    logger.info(f"Saved {len(batch_entries)} entries from batch {batch_idx + 1}. Total entries so far: {total_entries}")
                
                # Clear memory for this batch of logs
                for log_id in batch_log_ids:
                    if log_id in batch_ego_motion:
                        del batch_ego_motion[log_id]
                    
            # Report progress
            logger.info(f"Completed {batch_idx + 1}/{num_batches} batches ({(batch_idx + 1) * 100 / num_batches:.1f}%)")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Report final stats
        logger.info("=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Dataset path: {argoverse_root}")
        logger.info(f"Total unique log IDs processed: {len(processed_log_ids)}")
        logger.info(f"Total stereo pairs extracted: {total_entries}")
        logger.info(f"Conversion completed in {datetime.now() - start_time}")
        logger.info("=" * 50)
        
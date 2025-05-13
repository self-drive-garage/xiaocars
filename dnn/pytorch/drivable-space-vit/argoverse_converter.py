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

def save_metadata(metadata_entries, output_root, split_name):
    """Helper function to save metadata to a file."""
    metadata_file = os.path.join(output_root, f"{split_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata_entries, f, indent=2)
    logger.info(f"Saved metadata for {split_name} split with {len(metadata_entries)} stereo pairs to {metadata_file}")
    return metadata_file

def convert_argoverse2(argoverse_root, output_root, splits=None, img_size=None, num_workers=4):
    """
    Convert Argoverse 2 dataset with proper ego motion extraction.
    
    Args:
        argoverse_root: Path to Argoverse 2 dataset root
        output_root: Path to output directory
        splits: Dictionary mapping 'train', 'val', 'test' to lists of log IDs
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers
    """
    split_name = "val"
    start_time = datetime.now()
    logger.info(f"Converting Argoverse 2 dataset from {argoverse_root} to {output_root}")
    
    # Create output directory
    ensure_directory(output_root)
    
    # Define which cameras to use
    cam_names = (StereoCameras.STEREO_FRONT_LEFT, StereoCameras.STEREO_FRONT_RIGHT)
    
    # Process each split
    # for split_name in ['train', 'val']:
    # Store all frames with their metadata
    metadata_entries = []
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

        # First pass: collect all poses with their timestamps and image paths
        # Use a try-except inside the loop to handle individual iteration errors
        processed_log_ids = set()

        # #raise exception(dataset.num_logs)
        for log_idx, datum in enumerate(track(dataset, f"Processing {split_name} data...")):
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
                logger.error(f"Error processing log at index {log_idx}: {str(e)}")
                logger.debug(traceback.format_exc())
                # Continue to next log even if this one fails
                #raise e

        logger.info(f"Collected pose data for {len(log_pose_history)} logs in {split_name} split")
        logger.info(f"Processed {len(processed_log_ids)} unique log IDs:")
        
        # Second pass: compute ego motion features for each log
        ego_motion_by_log = {}
        for log_id, poses in log_pose_history.items():
            try:
                # Convert rotation matrices to quaternions for easier computation
                for pose in poses:
                    r = Rotation.from_matrix(pose['rotation'])
                    # Store as [x, y, z, w] format
                    pose['rotation'] = r.as_quat().tolist()  # [x, y, z, w]
                
                # Compute ego motion features
                ego_motion_features = compute_ego_motion_features(poses)
                ego_motion_by_log[log_id] = ego_motion_features
            except Exception as e:
                logger.error(f"Error computing ego motion for log {log_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                # Continue to next log even if this one fails
                continue
        
        # Third pass: create metadata entries with stereo pairs
        for timestamp_key, cameras in stereo_pairs.items():
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
                
                # Continue to next entry even if this one fails
                #raise e

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        #raise e
    finally:
        # Always save whatever data we've collected, even if there was an error
        if metadata_entries:
            metadata_file = save_metadata(metadata_entries, output_root, split_name)
            logger.info(f"Successfully saved {len(metadata_entries)} entries to {metadata_file}")
        else:
            logger.error("No metadata entries were collected. Check logs for errors.")
    
    end_time = datetime.now()
    
    # Print final stats
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Dataset path: {argoverse_root}")
    logger.info(f"Total unique log IDs processed: {len(processed_log_ids)}")
    logger.info(f"Total stereo pairs extracted: {len(metadata_entries)}")
    logger.info(f"Conversion completed in {end_time - start_time}")
    logger.info("=" * 50)
        
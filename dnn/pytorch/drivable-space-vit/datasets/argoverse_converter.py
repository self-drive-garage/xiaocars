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
import tempfile
import glob
import uuid

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
            raise StopIteration
        
        # print(f"self._ptr: {self._ptr}")
        result = self.__getitem__(self._ptr)
        self._ptr += 1
        return result

def ensure_directory(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path



def calculate_angular_velocity(
    R1: np.ndarray, 
    R2: np.ndarray, 
    dt: float
) -> np.ndarray:
    """
    Calculate the angular velocity from two rotation matrices.
    
    Parameters:
    -----------
    R1 : np.ndarray of shape (3, 3)
        First rotation matrix at time t
    R2 : np.ndarray of shape (3, 3)
        Second rotation matrix at time t + dt
    dt : float
        Time difference between the two rotation matrices
        
    Returns:
    --------
    angular_velocity : np.ndarray of shape (3,) as python list
        The angular velocity vector in radians per second
    """
    # Calculate the relative rotation matrix
    R_rel = R2 @ R1.T  # R2 * inverse(R1)
    
    # Calculate the angle of rotation using trace
    # Clamp the value to handle numerical errors
    cos_theta = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Check if the rotation angle is close to zero
    if np.isclose(theta, 0):
        return np.zeros(3)
    
    # Extract the axis of rotation
    omega_hat = np.zeros(3)
    
    # Using the formula (R_rel - R_rel.T) / (2*sin(theta))
    if not np.isclose(np.sin(theta), 0):
        omega_hat = np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1]
        ]) / (2 * np.sin(theta))
    
    # Calculate the angular velocity
    angular_velocity = omega_hat * theta / dt
    
    return angular_velocity.tolist()


def compute_ego_motion_features(log_entries):
    """
    Compute rich ego motion features from pose sequence for a specific log ID.
    
    Args:
        log_entries: DataFrame containing all entries for a specific log ID
        
    Returns:
        Updated DataFrame with ego motion features
    """
    # Sort by timestamp
    log_entries = log_entries.sort_values(by='timestamp')
    
    # Convert string representations of lists back to actual lists if needed
    if isinstance(log_entries['translation'].iloc[0], str):
        log_entries['translation'] = log_entries['translation'].apply(eval)
    if isinstance(log_entries['rotation'].iloc[0], str):
        log_entries['rotation'] = log_entries['rotation'].apply(eval)
    
    # Process each row to calculate ego motion features
    for i in range(1, len(log_entries)):
        dt = (log_entries['timestamp'].iloc[i] - log_entries['timestamp'].iloc[i-1]) / 1e9  # seconds
        
        if dt > 1e-6:  # Avoid division by zero or very small dt
            try:
                # Linear velocity
                vel = [(log_entries['translation'].iloc[i][j] - log_entries['translation'].iloc[i-1][j]) / dt for j in range(3)]
                log_entries.at[log_entries.index[i], 'velocity'] = vel
                
                # Ensure rotation matrices are properly structured
                curr_rotation = log_entries['rotation'].iloc[i]
                prev_rotation = log_entries['rotation'].iloc[i-1]
   
                # Angular velocity
                angular_velocity = calculate_angular_velocity(  
                    np.vstack( prev_rotation), 
                    np.vstack(curr_rotation), 
                    dt
                )
                log_entries.at[log_entries.index[i], 'angular_velocity'] = angular_velocity
                
                # Calculate acceleration if possible
                if i > 1:
                    prev_vel = log_entries['velocity'].iloc[i-1]
                    if isinstance(prev_vel, list) and len(prev_vel) == 3:
                        log_entries.at[log_entries.index[i], 'acceleration'] = [
                            (vel[j] - prev_vel[j]) / dt for j in range(3)
                        ]
            except Exception as e:
                logger.warning(f">>>>>>>>>>>>> Error processing ego motion at index {i}: {str(e)}")
                # Use default values in case of error
                log_entries.at[log_entries.index[i], 'velocity'] = [0.0, 0.0, 0.0]
                log_entries.at[log_entries.index[i], 'angular_velocity'] = [0.0, 0.0, 0.0]
                log_entries.at[log_entries.index[i], 'acceleration'] = [0.0, 0.0, 0.0]
    
    return log_entries


def process_sweep(datum, argoverse_root, split_name, process_id):
    """Process a single sweep and return the stereo pair data"""
    try:
        timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
        synchronized_imagery = datum.synchronized_imagery
        log_id = datum.log_id
        
        if synchronized_imagery is None:
            return None
        
        left_stereo_camera = synchronized_imagery.get(StereoCameras.STEREO_FRONT_LEFT)
        right_stereo_camera = synchronized_imagery.get(StereoCameras.STEREO_FRONT_RIGHT)
        
        if left_stereo_camera is None or right_stereo_camera is None:
            return None

        base_path = f"{argoverse_root}/{split_name}/{log_id}/sensors/cameras/"
        city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[left_stereo_camera.timestamp_ns]

        # Create entry dictionary
        entry = {
            f'{StereoCameras.STEREO_FRONT_LEFT}': os.path.join(base_path, StereoCameras.STEREO_FRONT_LEFT, f"{left_stereo_camera.timestamp_ns}.jpg"),
            f'{StereoCameras.STEREO_FRONT_RIGHT}': os.path.join(base_path, StereoCameras.STEREO_FRONT_RIGHT, f"{right_stereo_camera.timestamp_ns}.jpg"),
            'timestamp': left_stereo_camera.timestamp_ns,
            'translation': city_SE3_ego_cam_t.translation.tolist(),
            'rotation': city_SE3_ego_cam_t.rotation.tolist(),
            'log_id': log_id,
            'velocity': [0.0, 0.0, 0.0],
            'acceleration': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }
        
        return entry
        
    except Exception as e:
        logger.error(f"Error in process_sweep: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


def worker_process(worker_id, indices, argoverse_root, output_root, split_name):
    """Worker process that handles a subset of dataset indices"""
    try:
        # Create dataset for this worker
        cam_names = (StereoCameras.STEREO_FRONT_LEFT, StereoCameras.STEREO_FRONT_RIGHT)
        dataset = FixedSensorDataloader(
            Path(argoverse_root),
            with_annotations=True,
            with_cache=False,
            cam_names=cam_names,
        )
        
        # Create a temp directory within output_root if it doesn't exist
        temp_dir = os.path.join(output_root, "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a unique temp file for this worker
        temp_file = os.path.join(temp_dir, f"temp_metadata_{worker_id}_{uuid.uuid4()}.parquet")
        
        entries = []
        for idx in tqdm(indices, desc=f"Worker {worker_id}"):
            try:
                if idx >= dataset.num_sweeps:
                    continue
                    
                datum = dataset[idx]
                entry = process_sweep(datum, argoverse_root, split_name, worker_id)
                
                if entry:
                    entries.append(entry)
                    
                    # Save in batches to avoid memory issues
                    if len(entries) >= 1000:
                        df = pd.DataFrame(entries)
                        # Use PyArrow to append to parquet file
                        if os.path.exists(temp_file):
                            # Read existing file schema
                            existing_table = pq.read_table(temp_file)
                            # Convert new data to table
                            new_table = pa.Table.from_pandas(df)
                            # Concatenate tables
                            combined_table = pa.concat_tables([existing_table, new_table])
                            # Write back
                            pq.write_table(combined_table, temp_file)
                        else:
                            df.to_parquet(temp_file)
                        entries = []
                        
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing index {idx}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Save any remaining entries
        if entries:
            df = pd.DataFrame(entries)
            if os.path.exists(temp_file):
                # Read existing file schema
                existing_table = pq.read_table(temp_file)
                # Convert new data to table
                new_table = pa.Table.from_pandas(df)
                # Concatenate tables
                combined_table = pa.concat_tables([existing_table, new_table])
                # Write back
                pq.write_table(combined_table, temp_file)
            else:
                df.to_parquet(temp_file)
        
        return temp_file
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


def combine_and_process_parquet_files(temp_files, output_root, split_name):
    """Combine all temporary parquet files and process ego motion by log ID"""
    try:
        logger.info(f"Combining {len(temp_files)} temporary files and processing ego motion")
        
        # Create a combined DataFrame from all temp files
        dfs = []
        for file in temp_files:
            if file and os.path.exists(file):
                df = pd.read_parquet(file)
                dfs.append(df)
        
        if not dfs:
            logger.error("No valid temporary files found")
            return
            
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataframe has {len(combined_df)} total entries")
        
        # Process ego motion for each log ID
        log_ids = combined_df['log_id'].unique()
        logger.info(f"Processing ego motion for {len(log_ids)} unique log IDs")
        
        final_dfs = []
        for log_id in tqdm(log_ids, desc="Processing ego motion by log"):
            log_df = combined_df[combined_df['log_id'] == log_id].copy()
            processed_df = compute_ego_motion_features(log_df)
            final_dfs.append(processed_df)
        
        # Combine all processed dataframes
        final_df = pd.concat(final_dfs, ignore_index=True)
        
        # Save to final parquet file
        final_file = os.path.join(output_root, f"{split_name}_metadata.parquet")
        final_df.to_parquet(final_file, index=False)
        
        logger.info(f"Saved final processed metadata to {final_file}")
        
        # Clean up temp files
        for file in temp_files:
            if file and os.path.exists(file):
                os.remove(file)
        
        # Try to clean up temp directory if it's empty
        temp_dir = os.path.join(output_root, "temp_files")
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
                logger.info(f"Removed empty temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temp directory: {str(e)}")
        
        return final_file
        
    except Exception as e:
        logger.error(f"Error combining and processing files: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


def convert_argoverse2(argoverse_root, output_root, splits=None, img_size=None, num_workers=100, save_frequency=100, batch_size=500):
    """
    Convert Argoverse 2 dataset with parallel processing and post-processing for ego motion.
    
    Args:
        argoverse_root: Path to Argoverse 2 dataset root
        output_root: Path to output directory
        splits: Dictionary mapping 'train', 'val', 'test' to lists of log IDs
        img_size: Optional tuple (width, height) to resize images
        num_workers: Number of parallel workers
        save_frequency: Number of entries to process before saving to parquet (to prevent data loss)
        batch_size: Number of log items to process in each batch
    """
    start_time = datetime.now()
    logger.info(f"Converting Argoverse 2 dataset from {argoverse_root} to {output_root}")
    
    # Create output directory and temp directory
    ensure_directory(output_root)
    ensure_directory(os.path.join(output_root, "temp_files"))
    
    # Define which cameras to use
    cam_names = (StereoCameras.STEREO_FRONT_LEFT, StereoCameras.STEREO_FRONT_RIGHT)
    split_name = "train"  # Using val split by default
    
    try:
        # Create a temporary dataset loader to get dataset size
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

        total_items = dataset.num_sweeps
        logger.info(f"Dataset contains {total_items} total items")
        
        # Determine number of workers (don't use more workers than CPUs)
        actual_workers = min(num_workers, mp.cpu_count())
        logger.info(f"Using {actual_workers} worker processes")
        
        # Divide work among workers
        items_per_worker = total_items // actual_workers
        remainder = total_items % actual_workers
        
        # Create index ranges for each worker
        worker_indices = []
        start_idx = 0
        for i in range(actual_workers):
            # Add one extra item to some workers if there's a remainder
            extra = 1 if i < remainder else 0
            end_idx = start_idx + items_per_worker + extra
            worker_indices.append(list(range(start_idx, end_idx)))
            start_idx = end_idx
        
        # Launch parallel workers
        with mp.Pool(processes=actual_workers) as pool:
            worker_func = partial(
                worker_process,
                argoverse_root=argoverse_root,
                output_root=output_root,
                split_name=split_name
            )
            
            temp_files = pool.starmap(worker_func, enumerate(worker_indices))
        
        # Combine all temp files and process ego motion
        final_file = combine_and_process_parquet_files(temp_files, output_root, split_name)
        
        if final_file:
            logger.info(f"Successfully created final metadata file: {final_file}")
        else:
            logger.error("Failed to create final metadata file")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Report final stats
        logger.info("=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Dataset path: {argoverse_root}")
        logger.info(f"Output path: {output_root}")
        logger.info(f"Conversion completed in {datetime.now() - start_time}")
        logger.info("=" * 50)

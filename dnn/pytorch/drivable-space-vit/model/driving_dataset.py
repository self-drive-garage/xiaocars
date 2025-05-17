import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import random
from pathlib import Path
import logging
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def get_default_config():
    """Return default configuration parameters for DrivingDataset"""
    return {
        'model': {
            'img_size': 224,
        },
        'dataset': {
            'seq_len': 5,              # Number of frames used by the model for one sample
            'batch_size': 16,
            'num_workers': 8,
            'random_sequence': True,
            'cache_images': False,
            'temporal': {
                'stride': 1,           # Number of frames to skip between sequences
                'overlap': 0,          # Number of overlapping frames between sequences
                'max_gap': 0.1,        # Maximum allowed time gap between consecutive frames (in seconds)
                'log_handling': {
                    'min_log_frames': 10,  # Minimum frames required in a log
                    'max_log_frames': 1000  # Maximum frames to use from a log
                },
                'validation': {
                    'stride': 5,       # Larger stride for validation
                    'max_gap': 0.05    # Stricter gap requirement for validation
                }
            }
        }
    }

class DrivingDataset(Dataset):
    """Dataset for stereo driving data with ego motion"""
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_len: int = None,
        img_size: int = None,
        transform=None,
        random_sequence=None,
        cache_images=None,
        config=None,
        debug: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Get default configuration
        default_config = get_default_config()
        
        # Use provided config if available
        dataset_config = {}
        model_config = {}
        if config is not None:
            if 'dataset' in config:
                dataset_config = config['dataset']
            if 'model' in config:
                model_config = config['model']
        
        # Use provided parameters if given, otherwise use config, fallback to defaults
        self.seq_len = seq_len if seq_len is not None else dataset_config.get('seq_len', default_config['dataset']['seq_len'])
        self.img_size = img_size if img_size is not None else model_config.get('img_size', default_config['model']['img_size'])
        self.random_sequence = random_sequence if random_sequence is not None else dataset_config.get('random_sequence', default_config['dataset']['random_sequence'])
        self.cache_images = cache_images if cache_images is not None else dataset_config.get('cache_images', default_config['dataset']['cache_images'])
        
        # Get temporal parameters
        temporal_config = dataset_config.get('temporal', default_config['dataset']['temporal'])
        self.temporal_stride = temporal_config.get('stride', 1)
        self.max_gap = temporal_config.get('max_gap', 0.1)
        self.min_log_frames = temporal_config.get('log_handling', {}).get('min_log_frames', 10)
        self.max_log_frames = temporal_config.get('log_handling', {}).get('max_log_frames', 1000)
        
        # Use validation-specific parameters if in validation mode
        if split == 'val' or split == 'test':
            validation_config = temporal_config.get('validation', {})
            self.temporal_stride = validation_config.get('stride', self.temporal_stride)
            self.max_gap = validation_config.get('max_gap', self.max_gap)
        
        # Only randomize for training
        self.is_train = split == 'train'
        if self.random_sequence is True:
            self.random_sequence = self.is_train
        
        self.image_cache = {}  # Dictionary to store loaded images
        
        # Default transform if none provided
        if transform is None:
            if self.is_train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform
        
        # Load metadata from parquet file
        self.metadata_file = self.data_dir / f"{split}_metadata.parquet"
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        # Load parquet file
        self.metadata = pd.read_parquet(self.metadata_file)
        
        # Debug mode: use only 5 log IDs
        if debug:
            unique_log_ids = self.metadata['log_id'].unique()
            selected_log_ids = unique_log_ids[:5]  # Take first 5 log IDs
            self.metadata = self.metadata[self.metadata['log_id'].isin(selected_log_ids)]
            logger.info(f"Debug mode: Using {len(selected_log_ids)} log IDs: {selected_log_ids}")
        
        # Group by log_id first
        self.log_sequences = {}
        
        # Sort by log_id and timestamp within each log
        self.metadata = self.metadata.sort_values(['log_id', 'timestamp'])
        
        # Group into sequences by log_id
        for log_id, group in self.metadata.groupby('log_id'):
            # Sort by timestamp within each log
            group = group.sort_values('timestamp')
            
            # Validate temporal continuity within log
            valid_frames = []
            for i in range(len(group)):
                if i > 0:
                    time_diff = (group.iloc[i]['timestamp'] - group.iloc[i-1]['timestamp']) / 1e9  # Convert ns to seconds
                    if time_diff <= 0:
                        logger.warning(f"Non-monotonic timestamps in log {log_id} at index {i}")
                        continue
                    if time_diff > self.max_gap:
                        logger.warning(f"Large time gap in log {log_id} at index {i}: {time_diff:.3f}s (expected ~0.1s for 10Hz)")
                        continue
                
                valid_frames.append({
                    'timestamp': group.iloc[i]['timestamp'],
                    'left_image_path': group.iloc[i]["stereo_front_left"],
                    'right_image_path': group.iloc[i]["stereo_front_right"],
                    'translation': np.array(group.iloc[i]['translation'].tolist(), dtype=np.float32),  # Convert to list first
                    'rotation': np.array(group.iloc[i]['rotation'].tolist(), dtype=np.float32),
                    'velocity': np.array(group.iloc[i]['velocity'].tolist(), dtype=np.float32),
                    'acceleration': np.array(group.iloc[i]['acceleration'].tolist(), dtype=np.float32),
                    'angular_velocity': np.array(group.iloc[i]['angular_velocity'].tolist(), dtype=np.float32)
                })
            
            # Only keep logs with enough frames for at least one sequence
            if len(valid_frames) >= self.seq_len:
                # Limit the number of frames per log if specified
                if self.max_log_frames > 0:
                    valid_frames = valid_frames[:self.max_log_frames]
                self.log_sequences[log_id] = valid_frames
            
            # In __init__, after loading the first frame of each log:
            if debug and len(valid_frames) > 0:
                first_frame = valid_frames[0]
                logger.info(f"Debug - Motion value shapes for log {log_id}:")
                logger.info(f"  translation: {first_frame['translation'].shape}")
                logger.info(f"  rotation: {first_frame['rotation'].shape}")
                logger.info(f"  velocity: {first_frame['velocity'].shape}")
                logger.info(f"  acceleration: {first_frame['acceleration'].shape}")
                logger.info(f"  angular_velocity: {first_frame['angular_velocity'].shape}")
        
        # Create flat list of valid sequences for indexing
        self.sequences = []
        for log_id, frames in self.log_sequences.items():
            # Create sequences of seq_len frames
            for i in range(0, len(frames) - self.seq_len + 1, self.temporal_stride):
                self.sequences.append(frames[i:i + self.seq_len])
        
        logger.info(f"Loaded {len(self.log_sequences)} logs")
        logger.info(f"Created {len(self.sequences)} valid sequences")
        logger.info(f"Average sequence length: {np.mean([len(s) for s in self.sequences]):.2f} frames")
        
        # In __init__ method, after loading all log sequences
        ego_motion_data = []
        for frames in self.log_sequences.values():
            for frame in frames:
                ego_motion_values = np.concatenate([
                    frame['translation'],
                    np.arctan2([frame['rotation'][2, 1], -frame['rotation'][2, 0], frame['rotation'][1, 0]], 
                              [frame['rotation'][2, 2], np.sqrt(frame['rotation'][2, 1]**2 + frame['rotation'][2, 2]**2), frame['rotation'][0, 0]]),
                    frame['velocity'],
                    frame['acceleration'],
                    frame['angular_velocity']
                ])
                ego_motion_data.append(ego_motion_values)
        
        # Calculate mean and std for normalization
        self.ego_motion_mean = torch.tensor(np.mean(ego_motion_data, axis=0), dtype=torch.float32)
        self.ego_motion_std = torch.tensor(np.std(ego_motion_data, axis=0), dtype=torch.float32)
        # Avoid division by zero
        self.ego_motion_std = torch.where(self.ego_motion_std == 0, torch.ones_like(self.ego_motion_std), self.ego_motion_std)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get the sequence
        sequence = self.sequences[idx]
        
        # Load images for the sequence
        left_images = []
        right_images = []
        ego_motions = []
        timestamps = []
        
        for frame in sequence:
            # Load images
            left_img = self.load_image(frame['left_image_path'])
            right_img = self.load_image(frame['right_image_path'])
            
            # Extract ego motion features
            # Convert 3x3 rotation matrix to Euler angles
            r = frame['rotation']  # 3x3 rotation matrix
            # Extract Euler angles from rotation matrix (roll, pitch, yaw)
            roll = np.arctan2(r[2, 1], r[2, 2])
            pitch = np.arctan2(-r[2, 0], np.sqrt(r[2, 1]**2 + r[2, 2]**2))
            yaw = np.arctan2(r[1, 0], r[0, 0])
            rotation_euler = np.array([roll, pitch, yaw], dtype=np.float32)
            
            ego_motion_values = np.concatenate([
                frame['translation'],  # Already numpy array (3,)
                rotation_euler,        # Euler angles (3,)
                frame['velocity'],     # Already numpy array (3,)
                frame['acceleration'], # Already numpy array (3,)
                frame['angular_velocity']  # Already numpy array (3,)
            ])
            
            left_images.append(left_img)
            right_images.append(right_img)
            ego_motions.append(torch.from_numpy(ego_motion_values))
            timestamps.append(torch.tensor(frame['timestamp'], dtype=torch.float64))
        
        # Stack sequences
        left_images = torch.stack(left_images)  # [seq_len, C, H, W]
        right_images = torch.stack(right_images)  # [seq_len, C, H, W]
        ego_motions = torch.stack(ego_motions)  # [seq_len, 15]
        timestamps = torch.stack(timestamps)  # [seq_len]
        
        # Add future_features for future prediction loss
        # For simplicity, use the last timestep's ego_motion as future_features
        # In a real implementation, this would be the ego_motion of future frames
        # Example normalization for future_features
        future_features = (ego_motions[-1] - self.ego_motion_mean.to(ego_motions.device)) / self.ego_motion_std.to(ego_motions.device)
        
        return {
            'left_images': left_images,
            'right_images': right_images,
            'ego_motion': ego_motions,
            'timestamp': timestamps,
            'future_features': future_features,  # Added for future prediction loss
        }

    def load_image(self, image_path):
        """Load and transform an image from the given path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Transformed image tensor of shape [C, H, W]
        """
        # Convert to Path object if string
        image_path = Path(image_path)
        
        # Check if image is in cache
        if self.cache_images and str(image_path) in self.image_cache:
            return self.image_cache[str(image_path)]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Cache if enabled
        if self.cache_images:
            self.image_cache[str(image_path)] = image
        
        return image

    def clear_cache(self):
        """Clear the image cache if caching is enabled."""
        if self.cache_images:
            self.image_cache.clear()
            logger.info("Image cache cleared")

    def get_cache_size(self):
        """Get the current size of the image cache in MB."""
        if not self.cache_images:
            return 0
        
        total_size = 0
        for tensor in self.image_cache.values():
            total_size += tensor.element_size() * tensor.nelement()
        
        return total_size / (1024 * 1024)  # Convert to MB

def create_dataloader(dataset, batch_size=None, num_workers=None, shuffle=True, sampler=None, config=None, debug=False):
    """Create data loader with safer defaults for distributed training"""
    # Get default configuration
    default_config = get_default_config()
    
    # Use provided config if available
    dataset_config = {}
    if config is not None and 'dataset' in config:
        dataset_config = config['dataset']
    
    # Set safer defaults for distributed training
    is_distributed = sampler is not None  # If sampler provided, likely distributed
    
    # In debug mode or distributed mode, use safer settings
    if debug or is_distributed:
        # Start with 0 workers for debugging distributed training
        safe_workers = 0
        # Use smaller batch size for debugging
        safe_batch_size = 1 if debug else 2
    else:
        safe_workers = 2  # Default to 2 workers for non-distributed
        safe_batch_size = dataset_config.get('batch_size', default_config['dataset']['batch_size'])
    
    # Use provided values or safe defaults
    batch_size = batch_size if batch_size is not None else safe_batch_size
    num_workers = num_workers if num_workers is not None else safe_workers
    
    # Print dataloader configuration for debugging
    rank = getattr(sampler, 'rank', None) if sampler else 'N/A'
    print(f"Rank {rank}: Creating DataLoader with batch_size={batch_size}, workers={num_workers}, shuffle={shuffle and sampler is None}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Use 0 workers for debugging
        shuffle=shuffle if sampler is None else False,  # Don't shuffle when using sampler
        sampler=sampler,
        pin_memory=False,  # Disable pin_memory for debugging
        drop_last=True,  # Keep drop_last=True for consistent batch sizes
        persistent_workers=False,  # Disable persistent workers
        timeout=60  # Add timeout to detect worker hangs
    )
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
                'max_gap': 0.3,        # Maximum allowed time gap between consecutive frames (in seconds)
                'log_handling': {
                    'min_log_frames': 10,  # Minimum frames required in a log
                    'max_log_frames': 1000  # Maximum frames to use from a log
                },
                'validation': {
                    'stride': 5,       # Larger stride for validation
                    'max_gap': 0.3    # Stricter gap requirement for validation
                }
            }
        }
    }

class DrivingDataset(Dataset):
    """Dataset for multi-view driving data with ego motion"""
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
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.rank = rank
        self.world_size = world_size
        
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
        self.max_gap = temporal_config.get('max_gap', 0.3)
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
                    # Adjusted normalization values for driving data
                    # Values closer to actual driving image statistics (typically brighter than ImageNet)
                    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.25, 0.25, 0.25]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    # Adjusted normalization values for driving data
                    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.25, 0.25, 0.25]),
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
        
        # Shard data for distributed training - Only process a subset of logs assigned to this rank
        unique_log_ids = self.metadata['log_id'].unique()
        
        # Deterministic allocation of logs to different ranks
        if self.world_size > 1:
            # Get log IDs assigned to this rank
            assigned_log_ids = []
            for i, log_id in enumerate(unique_log_ids):
                if i % self.world_size == self.rank:
                    assigned_log_ids.append(log_id)
            
            # Only keep metadata for assigned logs
            self.metadata = self.metadata[self.metadata['log_id'].isin(assigned_log_ids)]
            logger.info(f"Rank {self.rank}/{self.world_size}: Assigned {len(assigned_log_ids)} logs out of {len(unique_log_ids)}")
        
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
                        # logger.warning(f"Large time gap in log {log_id} at index {i}: {time_diff:.3f}s (expected ~0.1s for 10Hz)")
                        continue
                
                valid_frames.append({
                    'timestamp': group.iloc[i]['timestamp'],
                    'left_image_path': group.iloc[i]["ring_front_left"],  # Updated camera keys
                    'center_image_path': group.iloc[i]["ring_front_center"],  # Updated camera keys
                    'right_image_path': group.iloc[i]["ring_front_right"],  # Updated camera keys
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
                logger.info(f"  velocity: {first_frame['velocity'].shape}")
                logger.info(f"  acceleration: {first_frame['acceleration'].shape}")
                logger.info(f"  angular_velocity: {first_frame['angular_velocity'].shape}")
        
        # Create flat list of valid sequences for indexing
        self.sequences = []
        for log_id, frames in self.log_sequences.items():
            # Create sequences of seq_len frames
            for i in range(0, len(frames) - self.seq_len + 1, self.temporal_stride):
                self.sequences.append(frames[i:i + self.seq_len])
        
        logger.info(f"Rank {self.rank}/{self.world_size}: Loaded {len(self.log_sequences)} logs")
        logger.info(f"Rank {self.rank}/{self.world_size}: Created {len(self.sequences)} valid sequences")
        logger.info(f"Rank {self.rank}/{self.world_size}: Average sequence length: {np.mean([len(s) for s in self.sequences]):.2f} frames")
        
        # In __init__ method, after loading all log sequences
        ego_motion_data = []
        for frames in self.log_sequences.values():
            for frame in frames:
                # Only use velocity, acceleration, and angular velocity (removing translation and rotation)
                ego_motion_values = np.concatenate([
                    frame['velocity'],
                    frame['acceleration'],
                    frame['angular_velocity']
                ])
                ego_motion_data.append(ego_motion_values)
        
        # Calculate mean and std for normalization
        if len(ego_motion_data) > 0:
            self.ego_motion_mean = torch.tensor(np.mean(ego_motion_data, axis=0), dtype=torch.float32)
            self.ego_motion_std = torch.tensor(np.std(ego_motion_data, axis=0), dtype=torch.float32)
            # Avoid division by zero
            self.ego_motion_std = torch.where(self.ego_motion_std == 0, torch.ones_like(self.ego_motion_std), self.ego_motion_std)
        else:
            # Fallback if no data (shouldn't happen normally)
            self.ego_motion_mean = torch.zeros(9, dtype=torch.float32)
            self.ego_motion_std = torch.ones(9, dtype=torch.float32)
            logger.warning(f">>>>>>>> Rank {self.rank}: No ego motion data available, using default normalization values")
            
        # Synchronize ego_motion_mean and ego_motion_std across ranks if needed
        if self.world_size > 1 and torch.distributed.is_initialized():
            # Create local tensors with correct dimensions
            local_mean = self.ego_motion_mean.clone().to('cuda')
            local_std = self.ego_motion_std.clone().to('cuda')
            local_count = torch.tensor([len(ego_motion_data)], dtype=torch.float32).to('cuda')
            
            # All-reduce to get global sum and count
            torch.distributed.all_reduce(local_mean, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_std, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)
            
            # Compute global mean and std
            if local_count.item() > 0:
                self.ego_motion_mean = (local_mean / local_count.item()).cpu()
                self.ego_motion_std = (local_std / local_count.item()).cpu()
                self.ego_motion_std = torch.where(self.ego_motion_std == 0, torch.ones_like(self.ego_motion_std), self.ego_motion_std)
        
        # self.sequences = self.sequences[:150]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get the sequence
        sequence = self.sequences[idx]
        
        # Load images for the sequence
        left_images = []
        center_images = []  # Added center images
        right_images = []
        ego_motions = []
        timestamps = []
        
        for frame in sequence:
            # Load images
            left_img = self.load_image(frame['left_image_path'])
            center_img = self.load_image(frame['center_image_path'])  # Load center image
            right_img = self.load_image(frame['right_image_path'])
            
            # Extract ego motion features - only using velocity, acceleration, and angular velocity
            ego_motion_values = np.concatenate([
                frame['velocity'],     # Already numpy array (3,)
                frame['acceleration'], # Already numpy array (3,)
                frame['angular_velocity']  # Already numpy array (3,)
            ])
            
            left_images.append(left_img)
            center_images.append(center_img)  # Add center image
            right_images.append(right_img)
            ego_motions.append(torch.from_numpy(ego_motion_values))
            timestamps.append(torch.tensor(frame['timestamp'], dtype=torch.float64))
        
        # Stack sequences
        left_images = torch.stack(left_images)  # [seq_len, C, H, W]
        center_images = torch.stack(center_images)  # [seq_len, C, H, W]
        right_images = torch.stack(right_images)  # [seq_len, C, H, W]
        ego_motions = torch.stack(ego_motions)  # [seq_len, 9] - now 9 features instead of 15
        timestamps = torch.stack(timestamps)  # [seq_len]
        
        # Add future_features for future prediction loss
        # For simplicity, use the last timestep's ego_motion as future_features
        # Example normalization for future_features
        future_features = (ego_motions[-1] - self.ego_motion_mean.to(ego_motions.device)) / self.ego_motion_std.to(ego_motions.device)
        
        return {
            'left_images': left_images,
            'center_images': center_images,  # Added center images 
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
            # Open image file and explicitly convert to RGB
            # Some image formats might be stored in BGR format
            image = Image.open(image_path).convert('RGB')
            
            # Debug: print unique image path for first few images to trace color issues
            # if random.random() < 0.001:  # Only log ~0.1% of images to avoid spam
            #     logger.info(f"Loaded image {image_path}, mode: {image.mode}, size: {image.size}")
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

def create_dataloader(dataset, batch_size=None, num_workers=None, is_distributed=False, rank=0, world_size=1, is_train=False):
    """Create data loader with distributed sampling support
    
    Args:
        dataset: The dataset to load data from
        batch_size: Batch size per GPU
        num_workers: Number of worker processes for loading
        is_distributed: Whether running in distributed mode
        rank: Process rank in distributed training
        world_size: Total number of processes in distributed training
        is_train: Whether this is a training dataset
    """
    from torch.utils.data.distributed import DistributedSampler
    
    # Removed the hardcoded minimum batch size to respect config
    # Let's respect the batch size from the config file
    
    # Create sampler for distributed training
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,  # Shuffle only for training
            drop_last=True
        )
        logger.info(f"Rank {rank}/{world_size}: Created DistributedSampler with {len(dataset)} samples")
        shuffle = False  # When using DistributedSampler, DataLoader shuffle must be False
    else:
        sampler = None
        shuffle = is_train  # Shuffle only for training if not distributed
        logger.info(f"Created regular DataLoader with {len(dataset)} samples")
    
    # Log distribution info
    if is_distributed:
        logger.info(f"Rank {rank}/{world_size}: Batch size per GPU: {batch_size}, Total batch size: {batch_size * world_size}")
    else:
        logger.info(f"Batch size: {batch_size}")
    
    # Create and return DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,  # Only shuffle if no sampler
        drop_last=True,
        persistent_workers=(num_workers > 0),
        timeout=60
    )
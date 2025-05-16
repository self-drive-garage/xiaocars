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
                    'translation': group.iloc[i]['translation'],
                    'rotation': group.iloc[i]['rotation'],
                    'velocity': group.iloc[i]['velocity'],
                    'acceleration': group.iloc[i]['acceleration'],
                    'angular_velocity': group.iloc[i]['angular_velocity']
                })
            
            # Only keep logs with enough frames for at least one sequence
            if len(valid_frames) >= self.seq_len:
                # Limit the number of frames per log if specified
                if self.max_log_frames > 0:
                    valid_frames = valid_frames[:self.max_log_frames]
                self.log_sequences[log_id] = valid_frames
        
        # Create flat list of valid sequences for indexing
        self.sequences = []
        for log_id, frames in self.log_sequences.items():
            # Create sequences of seq_len frames
            for i in range(0, len(frames) - self.seq_len + 1, self.temporal_stride):
                self.sequences.append(frames[i:i + self.seq_len])
        
        logger.info(f"Loaded {len(self.log_sequences)} logs")
        logger.info(f"Created {len(self.sequences)} valid sequences")
        logger.info(f"Average sequence length: {np.mean([len(s) for s in self.sequences]):.2f} frames")
    
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
            ego_motion_values = (
                frame['translation'] +  # 3 values
                frame['rotation'] +     # 3 values
                frame['velocity'] +     # 3 values
                frame['acceleration'] + # 3 values
                frame['angular_velocity']  # 3 values
            )
            
            left_images.append(left_img)
            right_images.append(right_img)
            ego_motions.append(torch.tensor(ego_motion_values, dtype=torch.float32))
            timestamps.append(torch.tensor(frame['timestamp'], dtype=torch.float64))
        
        # Stack sequences
        left_images = torch.stack(left_images)  # [seq_len, C, H, W]
        right_images = torch.stack(right_images)  # [seq_len, C, H, W]
        ego_motions = torch.stack(ego_motions)  # [seq_len, 15]
        timestamps = torch.stack(timestamps)  # [seq_len]
        
        return {
            'left_images': left_images,
            'right_images': right_images,
            'ego_motion': ego_motions,
            'timestamp': timestamps,
        }


def create_dataloader(dataset, batch_size=None, num_workers=None, shuffle=True, sampler=None, config=None):
    """Create data loader for the dataset"""
    # Get default configuration
    default_config = get_default_config()
    
    # Use provided config if available
    dataset_config = {}
    if config is not None and 'dataset' in config:
        dataset_config = config['dataset']
        
    # Use provided parameters if given, otherwise use config, fallback to defaults
    batch_size = batch_size if batch_size is not None else dataset_config.get('batch_size', default_config['dataset']['batch_size'])
    num_workers = num_workers if num_workers is not None else dataset_config.get('num_workers', default_config['dataset']['num_workers'])
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch
    )

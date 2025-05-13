import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import random
from pathlib import Path
import logging
from PIL import Image

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
            'seq_len': 5,
            'batch_size': 16,
            'num_workers': 8,
            'random_sequence': True,
            'cache_images': False,
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
        if self.random_sequence is True:  # Only randomize for training
            self.random_sequence = self.random_sequence and split == 'train'
        self.is_train = split == 'train'
        self.cache_images = cache_images if cache_images is not None else dataset_config.get('cache_images', default_config['dataset']['cache_images'])
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
        
        # Load metadata
        self.metadata_file = self.data_dir / f"{split}_metadata.json"
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter sequences to ensure they have at least seq_len frames
        self.sequences = [seq for seq in self.metadata['sequences'] if len(seq['frames']) >= self.seq_len]
        logger.info(f"Loaded {len(self.sequences)} sequences for {split}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Randomly select a starting frame, ensuring we have enough frames for the sequence
        if self.random_sequence:
            start_idx = random.randint(0, len(sequence['frames']) - self.seq_len)
        else:
            # Deterministic selection (e.g., always first frames or middle frames)
            start_idx = 0  # Or other deterministic strategy
        frames = sequence['frames'][start_idx:start_idx + self.seq_len]
        
        # Load images and ego motion data
        left_images = []
        right_images = []
        ego_motion = []
        timestamps = []
        
        for frame in frames:
            # Extract timestamp
            timestamps.append(frame.get('timestamp', 0))
            
            # Prepare paths
            left_path = frame['left_image_path']
            right_path = frame['right_image_path']
            
            # Handle absolute vs relative paths
            if left_path.startswith('/'):
                left_img_path = Path(left_path)
            else:
                left_img_path = self.data_dir / left_path
                
            if right_path.startswith('/'):
                right_img_path = Path(right_path)
            else:
                right_img_path = self.data_dir / right_path
            
            # Load images with caching if enabled
            try:
                if self.cache_images and str(left_img_path) in self.image_cache:
                    left_img = self.image_cache[str(left_img_path)]
                else:
                    left_img = Image.open(left_img_path).convert('RGB')
                    if self.cache_images:
                        self.image_cache[str(left_img_path)] = left_img
                
                if self.cache_images and str(right_img_path) in self.image_cache:
                    right_img = self.image_cache[str(right_img_path)]
                else:
                    right_img = Image.open(right_img_path).convert('RGB')
                    if self.cache_images:
                        self.image_cache[str(right_img_path)] = right_img
            except (FileNotFoundError, IOError) as e:
                logger.error(f"Error loading image: {e}")
                raise RuntimeError(f"Failed to load images at {left_img_path} or {right_img_path}")
            
            # Apply transformations
            if self.transform:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
            
            left_images.append(left_img)
            right_images.append(right_img)
            
            # Define which fields to extract and provide defaults
            ego_motion_fields = {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'acceleration': {'x': 0.0, 'y': 0.0},
                'velocity': {'x': 0.0, 'y': 0.0},
                'angular_velocity': {'roll': 0.0, 'yaw': 0.0}
            }

            # Extract available data with defaults for missing fields
            ego_motion_values = []
            for field, default in ego_motion_fields.items():
                if isinstance(default, dict):
                    for subfield, subdefault in default.items():
                        value = frame.get(field, {}).get(subfield, subdefault)
                        ego_motion_values.append(value)
                else:
                    ego_motion_values.append(frame.get(field, default))

            frame_ego_motion = torch.tensor(ego_motion_values, dtype=torch.float32)
            ego_motion.append(frame_ego_motion)
        
        # Stack data
        left_images = torch.stack(left_images)   # (seq_len, C, H, W)
        right_images = torch.stack(right_images) # (seq_len, C, H, W)
        ego_motion = torch.stack(ego_motion)     # (seq_len, ego_motion_dim=12)
        timestamps = torch.tensor(timestamps, dtype=torch.float64)  # (seq_len,)
        
        return {
            'left_images': left_images,
            'right_images': right_images,
            'ego_motion': ego_motion,
            'sequence_id': sequence['id'],
            'timestamps': timestamps,
        }


def create_dataloader(dataset, batch_size=None, num_workers=None, shuffle=True, config=None):
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
        pin_memory=True,
        drop_last=True,
    )

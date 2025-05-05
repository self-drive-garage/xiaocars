import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
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

# Constants
IMG_SIZE = 224  # Input image size for ViT
SEQ_LEN = 5  # Number of frames in sequence
BATCH_SIZE = 16  # Batch size for training
NUM_WORKERS = 8  # Number of worker processes for data loading

class DrivingDataset(Dataset):
    """Dataset for stereo driving data with ego motion"""
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_len: int = SEQ_LEN,
        img_size: int = IMG_SIZE,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.img_size = img_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
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
        self.sequences = [seq for seq in self.metadata['sequences'] if len(seq['frames']) >= seq_len]
        logger.info(f"Loaded {len(self.sequences)} sequences for {split}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Randomly select a starting frame, ensuring we have enough frames for the sequence
        start_idx = random.randint(0, len(sequence['frames']) - self.seq_len)
        frames = sequence['frames'][start_idx:start_idx + self.seq_len]
        
        # Load images and ego motion data
        left_images = []
        right_images = []
        ego_motion = []
        
        for frame in frames:
            # Load left and right camera images
            left_img_path = self.data_dir / frame['left_image_path']
            right_img_path = self.data_dir / frame['right_image_path']
            
            left_img = Image.open(left_img_path).convert('RGB')
            right_img = Image.open(right_img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
            
            left_images.append(left_img)
            right_images.append(right_img)
            
            # Extract ego motion data (speed, acceleration, steering, etc.)
            # Format might vary based on dataset, adjust accordingly
            frame_ego_motion = torch.tensor([
                frame['speed'],
                frame['acceleration']['x'],
                frame['acceleration']['y'],
                frame['acceleration']['z'],
                frame['steering_angle'],
                frame['angular_velocity']['z']  # Yaw rate
            ], dtype=torch.float32)
            
            ego_motion.append(frame_ego_motion)
        
        # Stack data
        left_images = torch.stack(left_images)   # (seq_len, C, H, W)
        right_images = torch.stack(right_images) # (seq_len, C, H, W)
        ego_motion = torch.stack(ego_motion)     # (seq_len, ego_motion_dim)
        
        return {
            'left_images': left_images,
            'right_images': right_images,
            'ego_motion': ego_motion,
            'sequence_id': sequence['id'],
        }


def create_dataloader(dataset, batch_size, num_workers, shuffle=True):
    """Create data loader for the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )

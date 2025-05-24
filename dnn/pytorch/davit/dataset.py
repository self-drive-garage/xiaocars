"""
PyTorch Dataset loader for Cityscapes data stored in Parquet format.

Features:
- Efficient loading from Parquet files
- Data augmentation for training
- Memory-efficient batch loading
- Support for distributed training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class CityscapesParquetDataset(Dataset):
    """
    PyTorch Dataset for loading Cityscapes data from Parquet files.
    
    Args:
        parquet_dir: Directory containing Parquet files
        split: Dataset split ('train', 'val', 'test')
        transform: Albumentations transform pipeline
        cache_size: Number of Parquet files to cache in memory
    """
    
    def __init__(
        self,
        parquet_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        cache_size: int = 2
    ):
        self.parquet_dir = Path(parquet_dir)
        self.split = split
        self.transform = transform
        self.cache_size = cache_size
        
        # Load metadata
        with open(self.parquet_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load index
        self.index_df = pd.read_parquet(self.parquet_dir / 'index.parquet')
        self.index_df = self.index_df[self.index_df['split'] == split].reset_index(drop=True)
        
        if len(self.index_df) == 0:
            raise ValueError(f"No data found for split '{split}'")
        
        # Calculate cumulative indices for efficient indexing
        self.cumulative_sizes = [0]
        for _, row in self.index_df.iterrows():
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + row['num_samples'])
        
        self.total_samples = self.cumulative_sizes[-1]
        
        # Cache for loaded Parquet files
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_order: List[str] = []
        
    def __len__(self) -> int:
        return self.total_samples
    
    def _get_file_and_index(self, idx: int) -> Tuple[str, int]:
        """
        Convert global index to file and local index.
        
        Args:
            idx: Global index
            
        Returns:
            Tuple of (filename, local_index)
        """
        # Binary search for the correct file
        file_idx = 0
        for i in range(len(self.cumulative_sizes) - 1):
            if self.cumulative_sizes[i] <= idx < self.cumulative_sizes[i + 1]:
                file_idx = i
                break
        
        local_idx = idx - self.cumulative_sizes[file_idx]
        filename = self.index_df.iloc[file_idx]['filename']
        
        return filename, local_idx
    
    def _load_parquet_file(self, filename: str) -> pd.DataFrame:
        """
        Load a Parquet file with caching.
        
        Args:
            filename: Name of the Parquet file
            
        Returns:
            DataFrame containing the data
        """
        if filename in self.cache:
            # Move to end of cache order (LRU)
            self.cache_order.remove(filename)
            self.cache_order.append(filename)
            return self.cache[filename]
        
        # Load new file
        df = pd.read_parquet(self.parquet_dir / filename)
        
        # Add to cache
        self.cache[filename] = df
        self.cache_order.append(filename)
        
        # Remove oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        return df
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image' and 'mask' tensors
        """
        # Get file and local index
        filename, local_idx = self._get_file_and_index(idx)
        
        # Load data
        df = self._load_parquet_file(filename)
        row = df.iloc[local_idx]
        
        # Decode image
        img_bytes = row['image']
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Decode mask
        mask_bytes = row['mask']
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()  # Ensure mask is long tensor
        else:
            # Default conversion to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'city': row['city'],
            'filename': row['filename']
        }


def get_training_augmentation(img_size: Tuple[int, int] = (1024, 512)) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Args:
        img_size: Target image size (width, height)
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # First resize to target size
        A.Resize(height=img_size[1], width=img_size[0]),
        
        # Spatial augmentations
        A.RandomResizedCrop(
            size=(img_size[1], img_size[0]),  # (height, width)
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=1.0
            ),
        ], p=0.8),
        
        # Weather augmentations (for robustness)
        A.OneOf([
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=5,
                brightness_coefficient=0.7,
                p=1.0
            ),
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.1,
                p=1.0
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=3,
                num_flare_circles_upper=7,
                src_radius=100,
                src_color=(255, 255, 255),
                p=1.0
            ),
        ], p=0.1),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_validation_augmentation(img_size: Tuple[int, int] = (1024, 512)) -> A.Compose:
    """
    Get augmentation pipeline for validation.
    
    Args:
        img_size: Target image size (width, height)
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def create_dataloaders(
    parquet_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (1024, 512),
    pin_memory: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        parquet_dir: Directory containing Parquet files
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Target image size (width, height)
        pin_memory: Whether to pin memory for CUDA
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CityscapesParquetDataset(
        parquet_dir=parquet_dir,
        split='train',
        transform=get_training_augmentation(img_size)
    )
    
    val_dataset = CityscapesParquetDataset(
        parquet_dir=parquet_dir,
        split='val',
        transform=get_validation_augmentation(img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for handling metadata.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    cities = [item['city'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'images': images,
        'masks': masks,
        'cities': cities,
        'filenames': filenames
    }


if __name__ == "__main__":
    # Test the dataset
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = CityscapesParquetDataset(
        parquet_dir='./cityscapes_parquet',
        split='train',
        transform=get_training_augmentation()
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load and visualize a sample
    sample = dataset[0]
    
    # Denormalize image for visualization
    img = sample['image'].numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std[:, None, None]) + mean[:, None, None]
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    
    mask = sample['mask'].numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title('Drivable Space Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_data.png')
    print("Sample visualization saved to sample_data.png")
"""
Convert Cityscapes dataset to Parquet format for easy sharing and loading.

This script processes the Cityscapes dataset and creates Parquet files containing:
- Image data (as byte arrays)
- Segmentation masks 
- Metadata (city, split, filename)

Usage:
    python cityscapes_to_parquet.py --cityscapes_root /path/to/cityscapes --output_dir ./cityscapes_parquet
"""

import os
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import json


# Cityscapes class mappings for drivable space
DRIVABLE_CLASSES = {
    0: 0,   # unlabeled -> non-drivable
    1: 0,   # ego vehicle -> non-drivable  
    2: 0,   # rectification border -> non-drivable
    3: 0,   # out of roi -> non-drivable
    4: 0,   # static -> non-drivable
    5: 0,   # dynamic -> non-drivable
    6: 0,   # ground -> non-drivable
    7: 1,   # road -> drivable
    8: 0,   # sidewalk -> non-drivable
    9: 1,   # parking -> drivable
    10: 0,  # rail track -> non-drivable
    11: 0,  # building -> non-drivable
    12: 0,  # wall -> non-drivable
    13: 0,  # fence -> non-drivable
    14: 0,  # guard rail -> non-drivable
    15: 0,  # bridge -> non-drivable
    16: 0,  # tunnel -> non-drivable
    17: 0,  # pole -> non-drivable
    18: 0,  # polegroup -> non-drivable
    19: 0,  # traffic light -> non-drivable
    20: 0,  # traffic sign -> non-drivable
    21: 0,  # vegetation -> non-drivable
    22: 0,  # terrain -> non-drivable
    23: 0,  # sky -> non-drivable
    24: 0,  # person -> non-drivable
    25: 0,  # rider -> non-drivable
    26: 0,  # car -> non-drivable
    27: 0,  # truck -> non-drivable
    28: 0,  # bus -> non-drivable
    29: 0,  # caravan -> non-drivable
    30: 0,  # trailer -> non-drivable
    31: 0,  # train -> non-drivable
    32: 0,  # motorcycle -> non-drivable
    33: 0,  # bicycle -> non-drivable
    -1: 2,  # license plate (255 in gt) -> uncertain
}


def process_cityscapes_split(
    cityscapes_root: Path,
    split: str,
    target_size: Tuple[int, int] = (1024, 512)
) -> pd.DataFrame:
    """
    Process a single split (train/val/test) of Cityscapes dataset.
    
    Args:
        cityscapes_root: Root directory of Cityscapes dataset
        split: Dataset split ('train', 'val', or 'test')
        target_size: Target image size (width, height)
    
    Returns:
        DataFrame with processed data
    """
    data_records = []
    
    # Paths
    img_dir = cityscapes_root / "leftImg8bit" / split
    gt_dir = cityscapes_root / "gtFine" / split
    
    # Get all cities in this split
    cities = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    
    for city in tqdm(cities, desc=f"Processing {split} split"):
        city_name = city.name
        
        # Get all images in this city
        img_files = sorted(city.glob("*_leftImg8bit.png"))
        
        for img_path in img_files:
            # Construct corresponding label path
            base_name = img_path.name.replace("_leftImg8bit.png", "")
            label_path = gt_dir / city_name / f"{base_name}_gtFine_labelIds.png"
            
            if not label_path.exists():
                print(f"Warning: Label not found for {img_path}")
                continue
            
            # Read and resize image
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, target_size)
            
            # Read and process label
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to drivable space mask
            drivable_mask = np.zeros_like(label, dtype=np.uint8)
            for cityscapes_id, drivable_id in DRIVABLE_CLASSES.items():
                if cityscapes_id == -1:  # Handle license plate (255)
                    drivable_mask[label == 255] = drivable_id
                else:
                    drivable_mask[label == cityscapes_id] = drivable_id
            
            # Encode image and mask as bytes
            _, img_encoded = cv2.imencode('.png', img)
            _, mask_encoded = cv2.imencode('.png', drivable_mask)
            
            
            # Create record
            record = {
                'city': city_name,
                'filename': img_path.name,
                'split': split,
                'image': img_encoded.tobytes(),
                'mask': mask_encoded.tobytes(),
                'width': target_size[0],
                'height': target_size[1]
            }
            
            data_records.append(record)
    
    return pd.DataFrame(data_records)


def create_parquet_dataset(
    cityscapes_root: str,
    output_dir: str,
    splits: List[str] = ['train', 'val'],
    target_size: Tuple[int, int] = (1024, 512),
    chunk_size: int = 500
):
    """
    Convert entire Cityscapes dataset to Parquet format.
    
    Args:
        cityscapes_root: Root directory of Cityscapes dataset
        output_dir: Output directory for Parquet files
        splits: List of splits to process
        target_size: Target image size (width, height)
        chunk_size: Number of samples per Parquet file
    """
    cityscapes_root = Path(cityscapes_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'target_size': target_size,
        'num_classes': 3,
        'class_names': ['non-drivable', 'drivable', 'uncertain'],
        'splits': splits
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        df = process_cityscapes_split(cityscapes_root, split, target_size)
        
        if len(df) == 0:
            print(f"Warning: No data found for {split} split")
            continue
        
        # Save to multiple Parquet files for easier handling
        num_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Define schema for better compression
            schema = pa.schema([
                ('city', pa.string()),
                ('filename', pa.string()),
                ('split', pa.string()),
                ('image', pa.binary()),
                ('mask', pa.binary()),
                ('width', pa.int32()),
                ('height', pa.int32())
            ])
            
            table = pa.Table.from_pandas(chunk_df, schema=schema)
            
            output_path = output_dir / f'cityscapes_{split}_{i:04d}.parquet'
            pq.write_table(
                table, 
                output_path,
                compression='snappy',
                use_dictionary=True
            )
            
            print(f"Saved {output_path} ({len(chunk_df)} samples)")
    
    # Create index file for easy loading
    index_data = []
    for parquet_file in sorted(output_dir.glob('*.parquet')):
        if parquet_file.name == 'index.parquet':
            continue
            
        # Extract split and chunk info from filename
        parts = parquet_file.stem.split('_')
        split = parts[1]
        chunk_id = int(parts[2])
        
        # Get number of rows
        table = pq.read_table(parquet_file)
        num_rows = len(table)
        
        index_data.append({
            'filename': parquet_file.name,
            'split': split,
            'chunk_id': chunk_id,
            'num_samples': num_rows
        })
    
    index_df = pd.DataFrame(index_data)
    index_df.to_parquet(output_dir / 'index.parquet')
    
    print(f"\nDataset conversion complete!")
    print(f"Total files created: {len(index_data) + 2}")  # +2 for metadata.json and index.parquet
    print(f"Output directory: {output_dir}")


def verify_parquet_dataset(output_dir: str):
    """
    Verify the created Parquet dataset by loading and displaying statistics.
    
    Args:
        output_dir: Directory containing Parquet files
    """
    output_dir = Path(output_dir)
    
    # Load index
    index_df = pd.read_parquet(output_dir / 'index.parquet')
    
    print("\nDataset Statistics:")
    print("-" * 40)
    
    for split in index_df['split'].unique():
        split_df = index_df[index_df['split'] == split]
        total_samples = split_df['num_samples'].sum()
        print(f"{split:8s}: {total_samples:6d} samples in {len(split_df)} files")
    
    # Load and display a sample
    print("\nLoading sample data...")
    first_file = output_dir / index_df.iloc[0]['filename']
    sample_df = pd.read_parquet(first_file)
    
    print(f"\nSample from {first_file.name}:")
    print(f"  Columns: {list(sample_df.columns)}")
    print(f"  Shape: {sample_df.shape}")
    
    # Decode and display first image
    img_bytes = sample_df.iloc[0]['image']
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    mask_bytes = sample_df.iloc[0]['mask']
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    
    print(f"  Image shape: {img.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask unique values: {np.unique(mask)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Cityscapes to Parquet format")
    parser.add_argument(
        "--cityscapes_root", 
        type=str, 
        required=True,
        help="Root directory of Cityscapes dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./cityscapes_parquet",
        help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        nargs='+', 
        default=['train', 'val'],
        help="Splits to process"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=1024,
        help="Target image width"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=512,
        help="Target image height"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=500,
        help="Number of samples per Parquet file"
    )
    parser.add_argument(
        "--verify", 
        action='store_true',
        help="Verify dataset after creation"
    )
    
    args = parser.parse_args()
    
    # Convert dataset
    create_parquet_dataset(
        cityscapes_root=args.cityscapes_root,
        output_dir=args.output_dir,
        splits=args.splits,
        target_size=(args.width, args.height),
        chunk_size=args.chunk_size
    )
    
    # Verify if requested
    if args.verify:
        verify_parquet_dataset(args.output_dir)
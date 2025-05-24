# Drivable Space Transformer Training Guide

This repository contains a minimal transformer-based approach for drivable space detection using the Cityscapes dataset.

## Project Structure

```
.
├── cityscapes_to_parquet.py   # Convert Cityscapes to Parquet format
├── dataset.py                  # PyTorch dataset loader
├── model.py                    # Transformer model architecture
├── train.py                    # Training script with Hydra
├── configs/
│   └── config.yaml            # Hydra configuration
└── README.md                  # This file
```

## Setup

### 1. Install Requirements

```bash
pip install torch torchvision
pip install hydra-core omegaconf
pip install opencv-python pillow
pip install pandas pyarrow
pip install albumentations
pip install matplotlib tqdm
pip install wandb  # Optional, for experiment tracking
```

### 2. Download Cityscapes Dataset

1. Register and download from [Cityscapes website](https://www.cityscapes-dataset.com/)
2. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`
3. Extract to a directory (e.g., `/path/to/cityscapes`)

Expected structure:
```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Usage

### Step 1: Convert Cityscapes to Parquet

Convert the dataset to Parquet format for efficient loading:

```bash
python cityscapes_to_parquet.py \
    --cityscapes_root /path/to/cityscapes \
    --output_dir ./cityscapes_parquet \
    --splits train val \
    --width 1024 \
    --height 512 \
    --chunk_size 500 \
    --verify
```

This will create:
- Multiple Parquet files for each split
- `metadata.json` with dataset information
- `index.parquet` for efficient file lookup

### Step 2: Train the Model

Basic training:
```bash
python train.py
```

With custom configuration:
```bash
python train.py \
    training.batch_size=16 \
    model.depth=8 \
    optimizer.lr=5e-5
```

Resume training:
```bash
python train.py \
    training.resume=true \
    training.resume_checkpoint="checkpoint_epoch_0050.pth"
```

### Step 3: Monitor Training

The training script will:
- Save checkpoints every 5 epochs to `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`
- Generate visualizations every 5 epochs to `outputs/YYYY-MM-DD/HH-MM-SS/visualizations/`
- Save the best model based on validation mIoU

## Configuration Options

Key configuration parameters in `configs/config.yaml`:

### Model Architecture
- `model.embed_dim`: Embedding dimension (default: 384)
- `model.depth`: Number of transformer blocks (default: 6)
- `model.num_heads`: Number of attention heads (default: 6)
- `model.patch_size`: Size of image patches (default: 16)

### Training
- `training.batch_size`: Batch size (default: 8)
- `training.num_epochs`: Number of epochs (default: 100)
- `training.mixed_precision`: Use mixed precision training (default: true)
- `optimizer.lr`: Learning rate (default: 1e-4)

### Data
- `data.parquet_dir`: Path to Parquet dataset
- `data.num_workers`: Number of data loading workers

## Model Architecture

The transformer model consists of:
1. **Patch Embedding**: Divides 512×1024 images into 16×16 patches
2. **Transformer Blocks**: 6 blocks with multi-head self-attention
3. **Simple Decoder**: Upsampling with transposed convolutions
4. **Output**: 3-class segmentation (non-drivable, drivable, uncertain)

Total parameters: ~44M (efficient for 32GB GPU)

## Expected Performance

- Training time: ~8-12 hours on single 32GB GPU
- Memory usage: ~20GB with batch size 8
- Expected mIoU: 92-95% on drivable space after 100 epochs

## Visualization

The training script generates visualizations showing:
- Original image
- Ground truth segmentation
- Model predictions

Color coding:
- Black: Non-drivable areas
- Green: Drivable areas
- Yellow: Uncertain areas

## Tips for Better Results

1. **Data Augmentation**: The dataset includes various augmentations (flip, brightness, weather effects)
2. **Learning Rate**: Use cosine annealing for stable training
3. **Mixed Precision**: Enabled by default for faster training
4. **Multi-GPU**: Set `training.multi_gpu=true` if available

## Troubleshooting

### Out of Memory
- Reduce `training.batch_size`
- Reduce `model.embed_dim` or `model.depth`
- Ensure `training.mixed_precision=true`

### Slow Data Loading
- Increase `data.num_workers`
- Ensure Parquet files are on fast storage (SSD)

### Poor Validation Performance
- Train for more epochs
- Adjust learning rate
- Try different augmentations

## Next Steps

After training:
1. Export model to TorchScript/ONNX for C++ deployment
2. Optimize with TensorRT for faster inference
3. Fine-tune on your specific campus/parking lot data

## Citation

If you use this code, please cite:
- Cityscapes dataset papers
- Vision Transformer (ViT) paper
- Any other relevant papers
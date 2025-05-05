# Drivable Space Vision Transformer

A stereo vision transformer model for drivable space segmentation using temporal information and ego motion.

## Architecture

The model consists of several key components:

1. **StereoTransformer**: The main model which combines:
   - Vision transformer (ViT) for spatial encoding of left and right images
   - Cross-view transformer for stereo fusion
   - Temporal transformer for sequence modeling
   - Ego motion integration for vehicle dynamics

2. **Self-supervised learning** objectives:
   - Image reconstruction
   - View consistency (stereo correspondence)
   - Future feature prediction

## Directory Structure

```
drivable-space-vit/
│
├── model/
│   ├── patch_embed.py                   # Patch embedding for ViT
│   ├── transformer_encoder_layer.py     # Basic transformer layer
│   ├── cross_view_transformer_layer.py  # Cross-view attention layer
│   ├── temporal_transfomer_layer.py     # Temporal transformer layer
│   ├── ego_motion_encoder.py            # Encoder for ego motion data
│   ├── drivable_space_decoder.py        # Decoder for segmentation
│   ├── stereo_transformer.py            # Main model
│   ├── self_supervised_loss.py          # Loss functions
│   ├── driving_dataset.py               # Dataset and dataloader
│   ├── cosine_scheduler_with_warmup.py  # LR scheduler
│   └── model.py                         # Main entry point for model usage
│
├── train_example.py                     # Example training script
├── inference_example.py                 # Example inference script
└── README.md                            # This file
```

## Model Usage

### Training

Use the `train_example.py` script to train the model:

```bash
python train_example.py \
    --data_dir /path/to/dataset \
    --output_dir ./outputs \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --warmup_epochs 10 \
    --img_size 224 \
    --patch_size 16 \
    --embed_dim 768 \
    --depth 12 \
    --num_heads 12
```

### Inference

Use the `inference_example.py` script for prediction:

```bash
python inference_example.py \
    --checkpoint ./outputs/best_model.pth \
    --left_images /path/to/left_img1.jpg /path/to/left_img2.jpg /path/to/left_img3.jpg \
    --right_images /path/to/right_img1.jpg /path/to/right_img2.jpg /path/to/right_img3.jpg \
    --output_dir ./results \
    --visualization
```

### Python API

Use the model in your own code:

```python
import torch
from model.model import create_model, load_model_from_checkpoint

# Create a new model
model = create_model(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Or load from checkpoint
model, checkpoint = load_model_from_checkpoint('path/to/checkpoint.pth')

# Prepare input
batch = {
    'left_images': left_images,   # Shape: (batch_size, seq_len, channels, height, width)
    'right_images': right_images, # Shape: (batch_size, seq_len, channels, height, width)
    'ego_motion': ego_motion      # Optional, shape: (batch_size, seq_len, ego_motion_dim)
}

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(batch, task='drivable_space')

# Get drivable space prediction
drivable_space = outputs['drivable_space']  # Shape: (B, 1, H, W)
```

## Dataset Format

The model expects a dataset with the following structure:

```
dataset/
├── train_metadata.json
├── val_metadata.json
├── test_metadata.json
└── images/
    ├── seq1/
    │   ├── left_001.jpg
    │   ├── right_001.jpg
    │   ├── left_002.jpg
    │   └── ...
    └── ...
```

The metadata JSON files should have the following format:

```json
{
  "sequences": [
    {
      "id": "seq1",
      "frames": [
        {
          "left_image_path": "images/seq1/left_001.jpg",
          "right_image_path": "images/seq1/right_001.jpg",
          "speed": 15.5,
          "acceleration": {"x": 0.1, "y": 0.0, "z": 0.05},
          "steering_angle": 0.1,
          "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.02}
        },
        ...
      ]
    },
    ...
  ]
}
```

## Requirements

- PyTorch 1.8+
- torchvision
- numpy
- PIL
- matplotlib (for visualization)
- tqdm
- tensorboard (for training visualization) 
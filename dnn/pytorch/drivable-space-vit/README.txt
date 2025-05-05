# Transformer-Based Self-Supervised Drivable Space Detection

This repository contains a PyTorch implementation of a transformer-based self-supervised model for drivable space detection using stereo cameras and ego motion data. The model is designed to be trained on large driving datasets like nuScenes and Argoverse v2, with the flexibility for future fine-tuning on parking lot scenarios.

## Overview

The model architecture leverages a transformer-based approach with the following key components:

1. **Stereo Vision Processing**: Separate but weight-shared patch embedding and spatial transformer blocks process left and right camera views.
2. **Cross-View Fusion**: Specialized transformer layers that enable information exchange between the two camera views.
3. **Temporal Modeling**: Temporal transformer layers that capture motion and dynamics over sequence frames.
4. **Ego Motion Integration**: A dedicated encoder that processes vehicle motion data and integrates it with visual features.
5. **Self-Supervised Learning Objectives**: Multiple pretext tasks including image reconstruction, view consistency, and future feature prediction.

## Requirements

- Python 3.8+
- PyTorch 2.5+
- CUDA-compatible GPU with at least 16GB memory (32GB recommended)
- Other dependencies (install via pip):
  ```
  torch>=2.5.0
  torchvision>=0.16.0
  numpy>=1.20.0
  matplotlib>=3.5.0
  Pillow>=9.0.0
  einops>=0.4.0
  ```

## Hardware Requirements

The model has been designed to run on a single NVIDIA RTX 5000 Ada Generation GPU with 32GB of memory. With this setup:
- Initial training iterations are feasible even with a batch size of 16
- Memory-efficient implementation allows for full model training
- Training time may extend to several days for convergence

## Dataset Preparation

The model expects data organized in the following structure:

```
dataset_root/
├── train_metadata.json
├── val_metadata.json
├── test_metadata.json
├── images/
│   ├── seq_0001/
│   │   ├── left_000001.jpg
│   │   ├── right_000001.jpg
│   │   ├── left_000002.jpg
│   │   └── ...
│   ├── seq_0002/
│   └── ...
└── ego_motion/
    ├── seq_0001.json
    ├── seq_0002.json
    └── ...
```

The metadata JSON files should have the following structure:

```json
{
  "sequences": [
    {
      "id": "seq_0001",
      "frames": [
        {
          "left_image_path": "images/seq_0001/left_000001.jpg",
          "right_image_path": "images/seq_0001/right_000001.jpg",
          "speed": 5.2,
          "acceleration": {"x": 0.1, "y": 0.0, "z": 0.05},
          "steering_angle": 0.02,
          "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.01}
        },
        ...
      ]
    },
    ...
  ]
}
```

### Data Conversion Scripts

For nuScenes dataset:
```python
# Example code to convert nuScenes to the required format
import nuscenes
from nuscenes.nuscenes import NuScenes
# ... (conversion code)
```

For Argoverse v2 dataset:
```python
# Example code to convert Argoverse to the required format
from av2.datasets.sensor.sensor_dataloader import SensorDataLoader
# ... (conversion code)
```

## Usage

### Training

To train the model:

```bash
python main.py --data_dir /path/to/dataset --batch_size 16 --epochs 100
```

Key parameters to adjust:
- `--batch_size`: Adjust based on available GPU memory
- `--seq_len`: Number of frames in each sequence (default: 5)
- `--img_size`: Input image size (default: 224)
- `--embed_dim`: Embedding dimension (default: 768)

### Checkpointing and Resuming

The training script automatically saves checkpoints to the `checkpoints` directory. To resume training from a checkpoint:

```bash
python main.py --data_dir /path/to/dataset --resume checkpoints/latest.pt
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data_dir /path/to/test_data
```

### Inference

For inference on new data:

```python
import torch
from model import load_model_for_inference, predict_drivable_space

# Load model
model = load_model_for_inference('checkpoints/best.pt')

# Prepare inputs (sequence of stereo images)
left_images = torch.randn(1, 5, 3, 224, 224)  # (B, T, C, H, W)
right_images = torch.randn(1, 5, 3, 224, 224)
ego_motion = torch.randn(1, 5, 6)  # Optional

# Predict drivable space
drivable_space = predict_drivable_space(model, left_images, right_images, ego_motion)
```

## Model Adaptation

### Transfer to Parking Lots

For future adaptation to parking lot scenarios, consider:

1. **Data Collection**: Collect a small dataset of stereo sequences in parking lots with the same camera setup used for pre-training.
2. **Domain Adaptation**: Implement domain adaptation losses to bridge the gap between road and parking lot domains.
3. **Fine-tuning**: Fine-tune the pre-trained model with a combination of supervised and self-supervised losses.

## Memory Optimization Tips

If you encounter memory issues:

1. Reduce batch size
2. Use gradient accumulation (update weights after multiple forward/backward passes)
3. Use mixed precision training (FP16)
4. Reduce sequence length
5. Reduce model size (embed_dim, num_layers, num_heads)

## Citation

If you use this code in your research, please cite:

```
@misc{transformer_drivable_space,
  author = {Your Name},
  title = {Transformer-Based Self-Supervised Drivable Space Detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/repo}}
}
```

## License

MIT License
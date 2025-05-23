# DeepSpeed Training for Drivable Space Vision Transformer

This document describes how to use the DeepSpeed-based training system that replaces the original PyTorch FSDP implementation.

## Overview

The DeepSpeed implementation provides the following advantages:
- **ZeRO-3 Parameter Sharding**: Enables training large models that don't fit on a single GPU
- **Mixed Precision Training**: FP16 support for memory efficiency and faster training
- **Optimized Communication**: Better gradient synchronization across 16 GPUs
- **Memory Optimization**: Advanced memory management with activation checkpointing
- **Better Stability**: More robust distributed training compared to FSDP

## Key Files

### New DeepSpeed Files
- `ds_train.py` - Main DeepSpeed training script
- `config/config_deepspeed.yaml` - DeepSpeed configuration
- `model/ds_modular_model.py` - DeepSpeed-compatible model creation utilities
- `model/ds_modular_vision_transformer.py` - DeepSpeed-optimized vision transformer
- `model/ds_transformer_encoder_layer.py` - DeepSpeed transformer encoder layer
- `model/ds_multihead_attention.py` - DeepSpeed-compatible attention implementation
- `launch_deepspeed.sh` - Launch script for distributed training

### Configuration
The DeepSpeed configuration (`config/config_deepspeed.yaml`) includes:
- **ZeRO-3 Settings**: Parameter sharding across 16 GPUs
- **FP16 Mixed Precision**: Memory-efficient training
- **Gradient Accumulation**: 8 steps for effective large batch training
- **Activation Checkpointing**: Memory optimization for large models

## Installation Requirements

```bash
# Install DeepSpeed
pip install deepspeed

# Verify installation
deepspeed --help
```

## Usage

### Quick Start
```bash
# Launch training on 16 GPUs
./launch_deepspeed.sh
```

### Manual Launch
```bash
# Using DeepSpeed launcher
deepspeed --num_gpus=16 ds_train.py --config-path=config --config-name=config_deepspeed

# Or using torchrun (alternative)
torchrun --nproc_per_node=16 ds_train.py --config-path=config --config-name=config_deepspeed
```

### Custom Configuration
```bash
# Use different configuration
CONFIG_NAME=my_custom_config ./launch_deepspeed.sh

# Use different number of GPUs
NUM_GPUS=8 ./launch_deepspeed.sh

# Use different master port
MASTER_PORT=29501 ./launch_deepspeed.sh
```

## Configuration Details

### Model Configuration
```yaml
model:
  img_size: 256
  patch_size: 16
  embed_dim: 768
  spatial_layers: 4
  cross_view_layers: 4
  temporal_layers: 4
  num_heads: 12
```

### DeepSpeed ZeRO-3 Configuration
```yaml
training:
  zero_optimization:
    stage: 3  # Parameter sharding
    cpu_offload: false  # Keep on GPU
    overlap_comm: true
    stage3_max_live_parameters: 1e9
    stage3_prefetch_bucket_size: 5e7
```

### Mixed Precision (FP16)
```yaml
training:
  fp16:
    enabled: true
    loss_scale: 0  # Dynamic loss scaling
    initial_scale_power: 16
```

### Memory Optimization
```yaml
training:
  activation_checkpointing:
    partition_activations: true
    cpu_checkpointing: false
    contiguous_memory_optimization: true
```

## Architecture Differences

### DeepSpeed vs FSDP
| Feature | FSDP | DeepSpeed |
|---------|------|-----------|
| Parameter Sharding | Manual wrapping policy | Automatic ZeRO-3 |
| Mixed Precision | Manual configuration | Integrated FP16 engine |
| Gradient Accumulation | Manual implementation | Built-in support |
| Memory Management | Limited options | Advanced optimizations |
| Checkpointing | Manual state dict handling | Integrated checkpoint system |

### Model Architecture
The DeepSpeed implementation uses:
- `DeepSpeedCompatibleMultiheadAttention` for optimized attention
- `DeepSpeedTransformerEncoderLayer` for transformer blocks
- `DeepSpeedModularVisionTransformer` for the complete model
- Automatic parameter sharding without manual wrapping policies

## Memory Usage

### Expected Memory Usage (per GPU with 16GB)
- **Model Parameters**: ~2-3GB (with ZeRO-3 sharding)
- **Activations**: ~4-6GB (with checkpointing)
- **Gradients**: ~1-2GB (with ZeRO-3)
- **Optimizer States**: ~1-2GB (with ZeRO-3)
- **Buffer/Overhead**: ~2-3GB
- **Total**: ~10-16GB per GPU

### Batch Size Recommendations
- **Per GPU Batch Size**: 2 (fits in 16GB)
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 2 × 8 × 16 = 256

## Monitoring

### Tensorboard Logs
```bash
tensorboard --logdir outputs/deepspeed/tensorboard
```

### Memory Monitoring
The training script logs memory usage to tensorboard:
- `train/memory_allocated_gb`
- `train/memory_reserved_gb`

### Performance Metrics
- Training loss per batch and epoch
- Validation loss
- Learning rate schedule
- GPU memory utilization

## Checkpointing

### Automatic Checkpointing
- Checkpoints saved every 5 epochs (configurable)
- Best model saved based on validation loss
- Includes model, optimizer, and scheduler states

### Checkpoint Structure
```
outputs/deepspeed/
├── checkpoint_epoch_N/
│   ├── mp_rank_00_model_states.pt
│   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── zero_pp_rank_0_mp_rank_00_model_states.pt
├── best_model/
└── tensorboard/
```

### Resuming Training
```yaml
training:
  resume: "outputs/deepspeed/checkpoint_epoch_100"
```

## Troubleshooting

### Common Issues

#### Out of Memory
- Reduce batch size: `dataset.batch_size: 1`
- Increase gradient accumulation: `training.gradient_accumulation: 16`
- Enable CPU offloading: `training.zero_optimization.cpu_offload: true`

#### Communication Timeouts
- Increase timeout: `NCCL_TIMEOUT=3600`
- Check network interfaces: `NCCL_SOCKET_IFNAME`
- Disable problematic transports: `NCCL_IB_DISABLE=1`

#### Gradient Explosion
- Reduce learning rate: `training.lr: 4e-4`
- Adjust gradient clipping: `training.gradient_clipping: 0.5`
- Check loss scaling: `training.fp16.initial_scale_power: 12`

### Debug Mode
Enable detailed logging:
```yaml
training:
  debug: true
```

### Performance Tuning
- Adjust `stage3_prefetch_bucket_size` for communication optimization
- Tune `stage3_max_live_parameters` for memory vs communication tradeoff
- Experiment with `overlap_comm` settings

## Validation

### Quick Test
```bash
# Test DeepSpeed installation
python -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"

# Test CUDA setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# Test model creation
python -c "from model.ds_modular_model import create_modular_model; print('Model creation: OK')"
```

### Single GPU Test
```bash
# Test on single GPU first
deepspeed --num_gpus=1 ds_train.py --config-path=config --config-name=config_deepspeed
```

## Performance Comparison

### Expected Improvements over FSDP
- **Memory Efficiency**: 20-30% reduction in peak memory usage
- **Training Speed**: 10-15% faster due to optimized communication
- **Stability**: Fewer hang/timeout issues
- **Scalability**: Better performance with 16+ GPUs

## Migration Notes

### From FSDP to DeepSpeed
1. **Configuration**: Update from `config_pytorch_fsdp.yaml` to `config_deepspeed.yaml`
2. **Training Script**: Use `ds_train.py` instead of `train.py`
3. **Launch**: Use `launch_deepspeed.sh` instead of torchrun
4. **Checkpoints**: DeepSpeed checkpoints are not compatible with FSDP checkpoints

### Code Changes
- Model creation uses `create_deepspeed_model_and_engine()`
- Training loop uses `model_engine.backward()` and `model_engine.step()`
- Checkpointing uses DeepSpeed's built-in save/load functions

This DeepSpeed implementation should provide more reliable and efficient training for your large vision transformer model across 16 GPUs with 16GB each. 
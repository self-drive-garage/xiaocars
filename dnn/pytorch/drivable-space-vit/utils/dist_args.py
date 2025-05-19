import argparse
import deepspeed

def parse_args():
    parser = argparse.ArgumentParser(description='Train Drivable Space ViT model using hybrid parallelism')

    # Add DeepSpeed's local_rank argument that gets passed automatically
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank passed from distributed launcher')
    
    # DeepSpeed parallel strategies
    parser.add_argument('--dp_size', type=int, default=4,
                        help='Data Parallelism size (number of data-parallel replicas)')
    parser.add_argument('--pp_size', type=int, default=2,
                        help='Pipeline Parallelism size (number of pipeline stages)')
    parser.add_argument('--tp_size', type=int, default=2,
                        help='Tensor Parallelism size (number of tensor-parallel slices)')
    parser.add_argument('--zero_stage', type=int, default=2,
                        help='ZeRO optimization stage (0, 1, 2, or 3)')
    
    # Config file argument
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='datasets/argoversev2',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to save checkpoints and logs')
    
    # Model arguments (can override config)
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for ViT')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=None,
                        help='MLP expansion ratio')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout probability')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length')
    
    # Training arguments (can override config)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU for training')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='Minimum learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=None,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', default=None,
                        help='Use mixed precision training')
    
    # Loss weights (can override config)
    parser.add_argument('--reconstruction_weight', type=float, default=None,
                        help='Weight for reconstruction loss')
    parser.add_argument('--consistency_weight', type=float, default=None,
                        help='Weight for consistency loss')
    parser.add_argument('--future_weight', type=float, default=None,
                        help='Weight for future prediction loss')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging and saving (can override config)
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Logging interval in steps')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Checkpoint saving interval in epochs')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Evaluation interval in epochs')
    
    # Visualization (can override config)
    parser.add_argument('--visualize_every', type=int, default=None,
                        help='Visualize predictions every N epochs')
    parser.add_argument('--num_viz_samples', type=int, default=None,
                        help='Number of samples to visualize')
    
    # # DeepSpeed config
    # parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
    #                     help='DeepSpeed configuration file')
    
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


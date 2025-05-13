#!/usr/bin/env python
"""
Script to convert older model checkpoints to work with the new YAML-based configuration.

Usage:
    python convert_checkpoint.py --input <old_checkpoint.pth> --output <new_checkpoint.pth> [--config config.yaml]
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert old model checkpoints to new YAML-based configuration')
    parser.add_argument('--input', type=str, required=True, help='Path to input checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to output checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def convert_checkpoint(input_path, output_path, config_path):
    """Convert a checkpoint from old format to new format"""
    logger.info(f"Loading checkpoint from {input_path}")
    
    # Check if input file exists
    if not Path(input_path).exists():
        logger.error(f"Input checkpoint file {input_path} does not exist")
        sys.exit(1)
        
    # Load the checkpoint
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)
    
    # Check if the checkpoint has model_config
    if 'model_config' not in checkpoint:
        logger.info("No model_config found in checkpoint. Creating one from hardcoded parameters.")
        # Create model_config from hardcoded parameters
        # These are the old default values
        checkpoint['model_config'] = {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4,
            'dropout': 0.1,
            'attn_dropout': 0.1,
            'ego_motion_dim': 12,
        }
    
    # Update model_config with values from config file
    model_config = checkpoint['model_config']
    logger.info(f"Original model_config: {model_config}")
    
    # Map between old and new parameter names if needed
    param_mapping = {
        'in_chans': 'num_channels',
        'depth': 'num_layers',
    }
    
    # Update model_config with new parameter names
    for old_name, new_name in param_mapping.items():
        if old_name in model_config and new_name not in model_config:
            model_config[new_name] = model_config[old_name]
    
    logger.info(f"Updated model_config: {model_config}")
    
    # Save the updated checkpoint
    logger.info(f"Saving updated checkpoint to {output_path}")
    try:
        torch.save(checkpoint, output_path)
        logger.info("Checkpoint converted successfully!")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    convert_checkpoint(args.input, args.output, args.config)

if __name__ == "__main__":
    main() 
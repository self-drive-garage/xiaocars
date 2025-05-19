import torch
import torch.nn as nn
import logging
from deepspeed.pipe import PipelineModule, LayerSpec
from model.pipeline_processors import ComprehensiveInputProcessor, OutputProcessor

logger = logging.getLogger(__name__)


def create_pipeline_model(model, loss_fn, num_stages):
    """
    Create a pipeline-parallel model from a base model.
    
    Args:
        model: The base model to convert to pipeline parallel
        loss_fn: The original loss function
        num_stages: Number of pipeline stages to use
        
    Returns:
        PipelineModule configured for distributed execution
    """
    # Create the comprehensive input processor
    input_processor = ComprehensiveInputProcessor()
    
    # Create layers_spec as before, but with the simplified input processor
    layers_spec = []
    
    # First stage: Input processing with the comprehensive processor
    input_layer = nn.ModuleList([
        input_processor,  # This now handles both extraction and data type conversion
        model.patch_embed,
        model.pos_drop
    ])
    layers_spec.append(LayerSpec(nn.Sequential, *input_layer))
    
    # Stage 2: Ego motion encoder
    ego_motion_layer = nn.ModuleList([
        model.ego_motion_encoder,
        model.motion_attention
    ])
    layers_spec.append(LayerSpec(nn.Sequential, *ego_motion_layer))
    
    # Calculate layers per transformer stage
    # We have effectively 6 total stages now:
    # 1. Input processing
    # 2. Ego motion processing
    # 3. Spatial transformer (Per-view processing)
    # 4. Cross-view transformer (View fusion)
    # 5. Temporal transformer (Temporal processing)
    # 6. Output (decoders and prediction heads)
    
    # Need at least 6 stages for complete pipeline
    num_transformer_stages = max(3, num_stages - 3)
    
    # Divide transformer stages evenly if we have at least 3 transformer stages
    if num_transformer_stages >= 3:
        # Distribute layers across spatial, cross-view, and temporal
        spatial_layers = len(model.spatial_transformer_layers)
        cross_view_layers = len(model.cross_view_transformer_layers)
        temporal_layers = len(model.temporal_transformer_layers)
        
        # Calculate how many stages to allocate for each transformer type
        total_layers = spatial_layers + cross_view_layers + temporal_layers
        spatial_stages = max(1, int(num_transformer_stages * (spatial_layers / total_layers)))
        cross_view_stages = max(1, int(num_transformer_stages * (cross_view_layers / total_layers)))
        temporal_stages = max(1, num_transformer_stages - spatial_stages - cross_view_stages)
        
        # Adjust if we over-allocated
        if spatial_stages + cross_view_stages + temporal_stages > num_transformer_stages:
            # Reduce the largest allocation
            if spatial_stages >= cross_view_stages and spatial_stages >= temporal_stages:
                spatial_stages -= 1
            elif cross_view_stages >= spatial_stages and cross_view_stages >= temporal_stages:
                cross_view_stages -= 1
            else:
                temporal_stages -= 1
        
        # Calculate layers per stage for each transformer type
        spatial_layers_per_stage = max(1, spatial_layers // spatial_stages)
        cross_view_layers_per_stage = max(1, cross_view_layers // cross_view_stages)
        temporal_layers_per_stage = max(1, temporal_layers // temporal_stages)
    else:
        # If we have fewer than 3 transformer stages, create more balanced groups
        spatial_layers_per_stage = len(model.spatial_transformer_layers)
        cross_view_layers_per_stage = len(model.cross_view_transformer_layers)
        temporal_layers_per_stage = len(model.temporal_transformer_layers)
    
    # Add spatial transformer layers
    for i in range(0, len(model.spatial_transformer_layers), spatial_layers_per_stage):
        end_idx = min(i + spatial_layers_per_stage, len(model.spatial_transformer_layers))
        stage_layers = model.spatial_transformer_layers[i:end_idx]
        if stage_layers:
            layers_spec.append(LayerSpec(nn.Sequential, *stage_layers))
    
    # Add cross-view transformer layers
    for i in range(0, len(model.cross_view_transformer_layers), cross_view_layers_per_stage):
        end_idx = min(i + cross_view_layers_per_stage, len(model.cross_view_transformer_layers))
        stage_layers = model.cross_view_transformer_layers[i:end_idx]
        if stage_layers:
            layers_spec.append(LayerSpec(nn.Sequential, *stage_layers))
    
    # Add temporal transformer layers
    for i in range(0, len(model.temporal_transformer_layers), temporal_layers_per_stage):
        end_idx = min(i + temporal_layers_per_stage, len(model.temporal_transformer_layers))
        stage_layers = model.temporal_transformer_layers[i:end_idx]
        if stage_layers:
            layers_spec.append(LayerSpec(nn.Sequential, *stage_layers))
    
    # Last stage: add the output processor
    output_processor = OutputProcessor(loss_fn, input_processor)
    output_layer = nn.ModuleList([
        model.norm,
        model.drivable_space_decoder,
        model.image_reconstruction_decoder,
        model.future_prediction_head,
        output_processor  # Add output processor to handle loss
    ])
    layers_spec.append(LayerSpec(nn.Sequential, *output_layer))
    
    # Handle stage consolidation if needed
    if len(layers_spec) < num_stages:
        logger.warning(f"Requested {num_stages} pipeline stages but model only has {len(layers_spec)} natural divisions")
        logger.warning(f"Will use {len(layers_spec)} pipeline stages instead")
    elif len(layers_spec) > num_stages:
        _consolidate_layers(layers_spec, num_stages)
    
    # Create pipeline with a custom loss function that simply returns the loss
    # calculated by our output processor
    def simple_loss_fn(outputs, _):
        # The real loss calculation happens inside OutputProcessor
        # outputs already contains the loss value
        return outputs
    
    pipeline_model = PipelineModule(
        layers=layers_spec,
        loss_fn=simple_loss_fn,  # Use simple pass-through loss
        num_stages=min(num_stages, len(layers_spec)),
        activation_checkpoint_interval=0  # Disable activation checkpointing for now
    )
    
    return pipeline_model


def _consolidate_layers(layers_spec, num_stages):
    """Helper function to consolidate layers to match requested stages"""
    logger.warning(f"Model has {len(layers_spec)} layers but only {num_stages} stages requested")
    logger.warning("Consolidating layers to fit into requested number of stages")
    
    # Simple consolidation: merge layers until we have the right number
    while len(layers_spec) > num_stages:
        # Find the smallest layer
        smallest_idx = 0
        smallest_size = float('inf')
        
        for i, layer_spec in enumerate(layers_spec[:-1]):  # Don't merge the final layer
            try:
                # Try both module_kwargs (DeepSpeed's attribute) and kwargs
                # to work with different versions of DeepSpeed
                if hasattr(layer_spec, 'module_kwargs'):
                    layer_size = sum(p.numel() for p in layer_spec.module_kwargs.parameters())
                else:
                    layer_size = sum(p.numel() for p in layer_spec.kwargs.parameters())
            except (AttributeError, TypeError):
                # If we can't calculate the size, use an estimated value
                layer_size = 1000000  # Default large value
            
            if layer_size < smallest_size:
                smallest_size = layer_size
                smallest_idx = i
        
        # Merge the smallest layer with the next one
        if smallest_idx < len(layers_spec) - 1:
            try:
                if hasattr(layers_spec[smallest_idx], 'module_kwargs'):
                    combined_modules = list(layers_spec[smallest_idx].module_kwargs) + list(layers_spec[smallest_idx + 1].module_kwargs)
                else:
                    combined_modules = list(layers_spec[smallest_idx].kwargs) + list(layers_spec[smallest_idx + 1].kwargs)
            except (AttributeError, TypeError):
                # If we can't combine based on kwargs, just merge the indices
                logger.warning(f"Merging layers {smallest_idx} and {smallest_idx + 1} without detailed size calculation")
                combined_modules = []
            
            layers_spec[smallest_idx] = LayerSpec(nn.Sequential, *combined_modules)
            layers_spec.pop(smallest_idx + 1) 
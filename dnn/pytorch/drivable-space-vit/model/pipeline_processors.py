import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional


class ComprehensiveInputProcessor(nn.Module):
    """
    Processor for the first stage of the pipeline.
    Extracts and formats input data for the transformer backbone.
    """
    def __init__(self):
        super().__init__()
        self.batch_list = None
        
    def forward(self, x_list):
        """
        Process the batch list to extract images in the correct format.
        Stores the original batch for later use in loss computation.
        
        Args:
            x_list: List of tensors from RepeatingLoader
            
        Returns:
            Images tensor in FP16 format, ready for patch embedding
        """
        # Store the original batch for loss computation later
        self.batch_list = x_list
        
        # Extract and prepare the images tensor (first element in batch list)
        images = x_list[0]
        if isinstance(images, torch.Tensor):
            # Ensure images are in the correct format: contiguous and FP16
            images = images.contiguous().half()
        
        return images  # Return a single tensor to the next stage


class TupleAdapter(nn.Module):
    """
    Adapter layer that converts between different data formats in the pipeline.
    This helps bridge components that expect different input structures.
    """
    def __init__(self, extract_index=0):
        super().__init__()
        self.extract_index = extract_index
        
    def forward(self, x):
        """
        Extract a specific tensor from a tuple input, ensure it's contiguous,
        and convert to FP16 for mixed precision training.
        
        Args:
            x: A tuple of tensors, typically (images, ego_motion)
            
        Returns:
            The extracted tensor (typically images) made contiguous and converted to FP16
        """
        # If input is already a tensor, ensure it's contiguous and convert to FP16
        if isinstance(x, torch.Tensor):
            tensor = x.contiguous() if not x.is_contiguous() else x
            return tensor.half()  # Convert to FP16
            
        # If input is a tuple/list, extract the tensor at the specified index
        if isinstance(x, (tuple, list)) and len(x) > self.extract_index:
            tensor = x[self.extract_index]
            # Ensure the extracted tensor is contiguous before returning
            if isinstance(tensor, torch.Tensor):
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                return tensor.half()  # Convert to FP16
            return tensor
            
        # Fallback - might lead to errors downstream but at least we're explicit
        raise TypeError(f"TupleAdapter expected tuple or tensor input, got {type(x)}")


class OutputProcessor(nn.Module):
    """
    Processor for the final stage of the pipeline.
    Handles formatting output and computing loss.
    """
    def __init__(self, original_loss_fn, input_processor: ComprehensiveInputProcessor):
        super().__init__()
        self.original_loss_fn = original_loss_fn
        self.input_processor = input_processor  # Reference to access stored batch
        
    def forward(self, model_outputs):
        """
        Process model outputs and compute loss using the original format.
        
        Args:
            model_outputs: Outputs from the model's prediction heads
            
        Returns:
            Loss value as tensor
        """
        # Get the batch list stored in the input processor
        batch_list = self.input_processor.batch_list
        
        # Reconstruct the input dictionary using the expected keys
        # The exact keys and indices depend on your dataset
        reconstructed_batch = {
            'images': batch_list[0],
            'ego_motion': batch_list[1] if len(batch_list) > 1 else None,
            'labels': batch_list[2] if len(batch_list) > 2 else None,
            'drivable_masks': batch_list[3] if len(batch_list) > 3 else None
            # Add other keys as needed
        }
        
        # Ensure model outputs and targets have consistent dtype
        # (model_outputs will be in FP16 due to mixed precision training)
        # The loss function should handle this internally, but we can help ensure
        # that tensors in the target dict are properly converted if needed
        
        # Calculate loss with the reconstructed batch
        loss, _ = self.original_loss_fn(model_outputs, reconstructed_batch)
        
        return loss 
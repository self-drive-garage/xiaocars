#!/usr/bin/env python3
"""Quick script to check GPU availability."""

import torch
import torch.distributed as dist
import os

def check_single_node():
    """Check GPUs on single node."""
    print("=" * 60)
    print("Single Node GPU Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("No GPUs available!")
        
    print("\n" + "=" * 60)


def check_distributed():
    """Check distributed setup."""
    # This function will be called by each process when using torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Create a tensor on the GPU
    tensor = torch.tensor([rank], device=device)
    
    # Gather all ranks
    if rank == 0:
        gathered = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.gather(tensor, gathered, dst=0)
        
        print(f"Distributed Setup Successful!")
        print(f"World Size: {world_size}")
        print(f"All ranks present: {[int(t.item()) for t in gathered]}")
    else:
        dist.gather(tensor, dst=0)
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "distributed":
        check_distributed()
    else:
        check_single_node() 
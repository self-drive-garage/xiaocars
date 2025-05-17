# fixed_dist_test.py
import os
import sys
import torch
import torch.distributed as dist
import time
from datetime import timedelta  # Correct import for timedelta

def run():
    """Run minimal distributed test"""
    # Get rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    print(f"Process {rank}/{world_size} (local_rank={local_rank}) starting")
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {rank}: Using CUDA device {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    else:
        device = torch.device("cpu")
        print(f"Rank {rank}: Using CPU")
    
    # Print key environment variables
    print(f"Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"Rank {rank}: MASTER_PORT={os.environ.get('MASTER_PORT', 'not set')}")
    
    # Initialize process group with extra debugging
    print(f"Rank {rank}: About to initialize process group")
    try:
        # Very important: Set very short timeout so we don't wait forever
        timeout = timedelta(seconds=10)
        
        # Initialize the process group using environment variables set by torchrun
        print(f"Rank {rank}: Using environment variables set by torchrun")
        
        # Initialize the process group
        torch.distributed.init_process_group(
            "gloo",
            timeout=timeout
        )
        print(f"Rank {rank}: Process group initialized successfully")
        
        # Print process group state
        print(f"Rank {rank}: Process group state: is_initialized={dist.is_initialized()}")
        
        # Test communication
        tensor = torch.ones(1, device=device)
        print(f"Rank {rank}: Created tensor on {device}")
        
        # Test all_reduce
        print(f"Rank {rank}: About to perform all_reduce")
        dist.all_reduce(tensor)
        print(f"Rank {rank}: all_reduce completed successfully")
        
        # Clean up
        print(f"Rank {rank}: About to destroy process group")
        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed, test complete")
        
    except Exception as e:
        print(f"Rank {rank}: Error in distributed setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()
import torch
import torch.distributed as dist
import sys

def check_backend(backend_name):
    """Check if a specific backend is available."""
    try:
        # Try to initialize the backend with a dummy process group
        if dist.is_available():
            print(f"Testing {backend_name} backend...")
            # This is a check for availability, not actual initialization
            if backend_name == "nccl":
                available = torch.cuda.is_available() and dist.is_nccl_available()
            elif backend_name == "gloo":
                available = dist.is_gloo_available()
            elif backend_name == "mpi":
                available = dist.is_mpi_available()
            else:
                available = False
                
            if available:
                print(f"✓ {backend_name.upper()} backend is AVAILABLE")
            else:
                print(f"✗ {backend_name.upper()} backend is NOT AVAILABLE")
            return available
        else:
            print("PyTorch distributed is not available")
            return False
    except Exception as e:
        print(f"Error checking {backend_name} backend: {e}")
        return False

def print_system_info():
    """Print system information relevant to distributed training."""
    print("\n=== SYSTEM INFORMATION ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            
    print("\n=== DISTRIBUTED BACKEND AVAILABILITY ===")

if __name__ == "__main__":
    print_system_info()
    
    # Check common backends
    backends = ["nccl", "gloo", "mpi"]
    available_backends = []
    
    for backend in backends:
        if check_backend(backend):
            available_backends.append(backend)
    
    print("\n=== RECOMMENDATION ===")
    if "nccl" in available_backends and torch.cuda.is_available():
        print("✓ NCCL is available and recommended for GPU training")
    elif "gloo" in available_backends and torch.cuda.is_available():
        print("✓ Gloo is available for GPU training (but NCCL would be faster if you could install it)")
    elif "gloo" in available_backends:
        print("✓ Gloo is available for CPU training")
    elif "mpi" in available_backends:
        print("✓ MPI is available")
    else:
        print("✗ No distributed backends are available") 
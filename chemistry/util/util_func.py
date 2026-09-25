"""
Shared utility functions.
"""
def get_available_gpu_count():
    """
    Detect the number of available accelerator devices.
    
    Returns:
        The available device count, or 1 when none can be detected.
    """
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            print(f"Detected {gpu_count} available GPU(s)")
            return gpu_count
        else:
            print("No available GPU detected; using CPU mode")
            return 1
    except ImportError:
        print("Warning: PyTorch is not installed; returning a device count of 1")
        return 1
    except Exception as e:
        print(f"Warning: Failed to detect the GPU count ({e}); returning 1")
        return 1



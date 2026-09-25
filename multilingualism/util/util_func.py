"""Shared utility functions."""


def get_available_gpu_count():
    """Return the number of available GPUs, falling back to one worker."""
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            print(f"Detected {gpu_count} available GPU(s)")
            return gpu_count
        else:
            print("No GPU detected; using a single worker")
            return 1
    except ImportError:
        print("Warning: PyTorch is not installed; using a single worker")
        return 1
    except Exception as e:
        print(f"Warning: could not detect GPUs ({e}); using a single worker")
        return 1



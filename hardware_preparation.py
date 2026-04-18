# start with virtual enviroment
# python -m venv .venv
# add to gitignore
# activate enviroment
# add .env and do the HF_token then add it to .gitingore
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
import torch


def gpu_preparation():
    """
        This function is created to prepare our hardware to run the right models on it and utilize the power of GPUS
    """
    # check for GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        torch.set_default_device("cuda")
        print("PyTorch default device set to CUDA(GPU)")
    else:
        print("WARNING: No GPU detected. Performance will be slow")
    
# scripts/check_env.py
"""Quick environment sanity check for Medical VQA project."""
import sys
import torch
from datasets import load_dataset

def check_packages():
    required = ["torch", "transformers", "peft", "datasets", "PIL", "tqdm", "gradio"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing packages:", ", ".join(missing))
        return False
    print("All required packages installed")
    return True

def check_gpu():
    if not torch.cuda.is_available():
        print("GPU not available!")
        return False
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")
    return True

def check_dataset():
    """Check if VQA-RAD dataset is accessible on HuggingFace without downloading."""
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("flaviagiammarino/vqa-rad", repo_type="dataset") 
        if len(files) > 0:
            print("VQA-RAD dataset accessible on HuggingFace")
            return True
    except Exception as e:
        print(f"Cannot reach dataset: {e}!")
        return False
    return False

if __name__ == "__main__":
    ok = all([check_packages(), check_gpu(), check_dataset()])
    sys.exit(0 if ok else 1)
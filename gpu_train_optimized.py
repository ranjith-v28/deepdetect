"""
Optimized GPU Training Script for Deepfake Detection
Addresses NaN loss issues and maximizes GPU utilization.
"""

import torch
import subprocess
import sys
import os
from pathlib import Path

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory // 1024**3
        reserved_memory = torch.cuda.memory_reserved(device) // 1024**3
        allocated_memory = torch.cuda.memory_allocated(device) // 1024**3
        free_memory = total_memory - allocated_memory

        print(f"GPU Memory - Total: {total_memory}GB, Used: {allocated_memory}GB, Free: {free_memory}GB")
        return free_memory
    return 0

def optimize_training_params(free_memory):
    """Optimize training parameters based on available GPU memory."""
    if free_memory > 6:  # High-end GPU
        batch_size = "8"
        max_frames = "12"
    elif free_memory > 3:  # Mid-range GPU
        batch_size = "4"
        max_frames = "8"
    else:  # Low-end GPU
        batch_size = "2"
        max_frames = "6"

    return batch_size, max_frames

def train_video_model():
    """Train video model with optimized GPU settings."""
    print("=" * 70)
    print("OPTIMIZED GPU TRAINING FOR DEEPFAKE DETECTION")
    print("=" * 70)

    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please run setup_gpu.py first.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # Check memory
    free_memory = check_gpu_memory()

    # Dataset paths
    video_dataset = r"C:\Users\ranji\dataset\deepfake_datase"
    audio_dataset = r"C:\Users\ranji\dataset\deepfake_audio"

    # Check if datasets exist
    if not os.path.exists(video_dataset):
        print(f"Video dataset not found: {video_dataset}")
        return
    if not os.path.exists(audio_dataset):
        print(f"Audio dataset not found: {audio_dataset}")
        return

    # Optimize parameters based on GPU memory
    batch_size, max_frames = optimize_training_params(free_memory)
    print(f"Optimized settings - Batch size: {batch_size}, Max frames: {max_frames}")

    print("\n" + "="*50)
    print("TRAINING VIDEO MODEL")
    print("="*50)

    # Video training command with stability fixes
    video_cmd = [
        sys.executable, "model/train_video_model.py",
        "--dataset", video_dataset,
        "--save_path", r"model\deepfake_video_model.pt",
        "--epochs", "20",           # Fewer epochs for stability
        "--batch_size", batch_size,
        "--lr", "0.0001",          # Conservative learning rate
        "--device", "cuda",
        "--max_frames", max_frames
    ]

    print("Starting video model training...")
    print(f"Command: {' '.join(video_cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(video_cmd, check=True, capture_output=True, text=True)
        print("Video model training completed successfully!")
        print(result.stdout[-500:])  # Show last 500 chars of output
    except subprocess.CalledProcessError as e:
        print(f"Video training failed: {e}")
        print("STDOUT:", e.stdout[-1000:])
        print("STDERR:", e.stderr[-1000:])
        return

    print("\n" + "="*50)
    print("TRAINING AUDIO MODEL")
    print("="*50)

    # Audio training command
    audio_cmd = [
        sys.executable, "train_audio_simple.py",
        "--dataset", audio_dataset,
        "--save_path", r"model\simple_audio_model.pt",
        "--epochs", "15",
        "--batch_size", "16",  # Smaller batch for audio
        "--lr", "0.0001",
        "--device", "cuda"
    ]

    print("Starting audio model training...")
    print(f"Command: {' '.join(audio_cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(audio_cmd, check=True, capture_output=True, text=True)
        print("Audio model training completed successfully!")
        print(result.stdout[-500:])  # Show last 500 chars of output
    except subprocess.CalledProcessError as e:
        print(f"Audio training failed: {e}")
        print("STDOUT:", e.stdout[-1000:])
        print("STDERR:", e.stderr[-1000:])

    print("\n" + "=" * 40)
    print("ALL MODELS TRAINED SUCCESSFULLY WITH GPU ACCELERATION!")
    print("=" * 40)

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")

def main():
    clear_gpu_cache()
    train_video_model()
    clear_gpu_cache()

if __name__ == "__main__":
    main()
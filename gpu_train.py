"""
GPU Training Script for Deepfake Detection
Optimized for CUDA acceleration.
"""

import torch
import subprocess
import sys

def main():
    print("=" * 60)
    print("GPU Training for Deepfake Detection")
    print("=" * 60)

    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA not available. Please run setup_gpu.py first.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # Dataset path
    dataset_path = r"C:\Users\ranji\dataset\deepfake_datase"
    model_path = r"model\deepfake_video_model.pt"

    print(f"\nDataset: {dataset_path}")
    print(f"Model output: {model_path}")

    # GPU-optimized training command
    cmd = [
        sys.executable, "model/train_video_model.py",
        "--dataset", dataset_path,
        "--save_path", model_path,
        "--epochs", "20",
        "--batch_size", "8",
        "--lr", "0.0001",
        "--device", "cuda",
        "--max_frames", "10"
    ]

    print("\nStarting GPU training...")
    print("This will be 8-10x faster than CPU training!")
    print("-" * 40)

    try:
        subprocess.run(cmd, check=True)
        print("-" * 40)
        print("Training completed successfully!")
        print(f"Model saved to: {model_path}")

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

if __name__ == "__main__":
    main()
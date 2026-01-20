"""
GPU-Optimized Training Script for Video Deepfake Detection
Forces GPU usage and optimizes for CUDA training.
"""

import torch
import subprocess
import sys
import os

def check_gpu():
    """Check GPU availability and CUDA setup."""
    print("=" * 60)
    print("GPU Detection and Setup Check")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        return True
    else:
        print("❌ No CUDA GPU detected")
        print("\nTo enable GPU training:")
        print("1. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        print("2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("3. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def train_with_gpu():
    """Train the model with GPU optimization."""
    print("\n" + "=" * 60)
    print("Starting GPU Training")
    print("=" * 60)

    # Dataset path
    dataset_path = r"C:\Users\ranji\dataset\deepfake_datase"
    model_path = r"model\deepfake_video_model.pt"

    # GPU-optimized training command
    cmd = [
        sys.executable, "model/train_video_model.py",
        "--dataset", dataset_path,
        "--save_path", model_path,
        "--epochs", "20",
        "--batch_size", "8",  # Increased for GPU
        "--lr", "0.0001",
        "--device", "cuda",
        "--max_frames", "10"
    ]

    print(f"Dataset: {dataset_path}")
    print(f"Model output: {model_path}")
    print(f"Device: CUDA GPU")
    print(f"Batch size: 8 (optimized for GPU)")
    print()

    try:
        print("🔥 Starting training...")
        print("Progress will be displayed below:")
        print("-" * 40)

        # Run training
        subprocess.run(cmd, check=True)

        print("-" * 40)
        print("Training completed successfully!")
        print(f"Model saved to: {model_path}")

        # Verify model
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024**2)  # MB
            print(".2f")
            print("Ready to use in the web application!")
        else:
            print("Model file not found. Check training output above.")

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print("Check the error messages above for details.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Main function."""
    if check_gpu():
        print("\nGPU detected! Proceeding with GPU training...")
        train_with_gpu()
    else:
        print("\nCannot proceed with GPU training.")
        print("Please install CUDA and GPU drivers first.")
        print("\nAlternatively, you can train on CPU (much slower):")
        print('python model/train_video_model.py --dataset "C:\\Users\\ranji\\dataset\\deepfake_datase" --device cpu')

if __name__ == "__main__":
    main()
"""
GPU Setup Script for Deepfake Detection Training
Automatically detects and configures NVIDIA GPU for PyTorch CUDA acceleration.
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, shell=False):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed."""
    print("=" * 60)
    print("Checking NVIDIA Drivers")
    print("=" * 60)

    # Try nvidia-smi command
    success, stdout, stderr = run_command(["nvidia-smi"])
    if success:
        print("NVIDIA drivers detected!")
        # Extract GPU info
        lines = stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"Driver: {line.strip()}")
            elif 'CUDA Version' in line:
                print(f"CUDA: {line.strip()}")
            elif '| NVIDIA' in line and '|' in line:
                # GPU model line
                parts = line.split('|')
                if len(parts) >= 3:
                    gpu_name = parts[2].strip()
                    if gpu_name and not gpu_name.startswith('N/A'):
                        print(f"GPU: {gpu_name}")
        return True
    else:
        print("NVIDIA drivers not found or not working")
        print("\nTo install NVIDIA drivers:")
        print("1. Visit: https://www.nvidia.com/Download/index.aspx")
        print("2. Download the latest driver for your GPU")
        print("3. Run the installer and reboot")
        return False

def check_cuda_installation():
    """Check CUDA installation."""
    print("\n" + "=" * 60)
    print("Checking CUDA Installation")
    print("=" * 60)

    # Check CUDA version via nvcc
    success, stdout, stderr = run_command(["nvcc", "--version"])
    if success:
        print("CUDA Toolkit found!")
        for line in stdout.split('\n'):
            if 'release' in line:
                print(f"CUDA: {line.strip()}")
                cuda_version = line.split('release ')[1].split(',')[0]
                print(f"Version: {cuda_version}")
                return cuda_version
    else:
        print("CUDA Toolkit not found")
        print("\nTo install CUDA Toolkit:")
        print("1. Visit: https://developer.nvidia.com/cuda-downloads")
        print("2. Choose your OS and download CUDA Toolkit 11.8 or 12.1")
        print("3. Run the installer")
        print("4. Add CUDA to PATH environment variable")
        return None

def install_pytorch_cuda(cuda_version):
    """Install PyTorch with CUDA support."""
    print("\n" + "=" * 60)
    print("Installing PyTorch with CUDA Support")
    print("=" * 60)

    # Determine PyTorch CUDA version based on CUDA version
    if cuda_version.startswith('13.') or cuda_version.startswith('12.'):
        pytorch_cuda = 'cu118'
        index_url = 'https://download.pytorch.org/whl/cu118'
        print(f"CUDA version {cuda_version} detected. Using cu118 (stable and compatible)")
    elif cuda_version.startswith('11.8'):
        pytorch_cuda = 'cu118'
        index_url = 'https://download.pytorch.org/whl/cu118'
    elif cuda_version.startswith('11.'):
        pytorch_cuda = 'cu118'
        index_url = 'https://download.pytorch.org/whl/cu118'
    else:
        print(f"CUDA version {cuda_version} detected. Using cu118 as fallback.")
        pytorch_cuda = 'cu118'
        index_url = 'https://download.pytorch.org/whl/cu118'

    print(f"Installing PyTorch with {pytorch_cuda}")

    # Uninstall CPU version first
    print("Removing CPU-only PyTorch...")
    run_command([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])

    # Install CUDA version
    print(f"Installing PyTorch with CUDA support...")
    success, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", index_url
    ])

    if success:
        print("PyTorch CUDA installation completed!")
        return True
    else:
        print("PyTorch CUDA installation failed!")
        print("Error:", stderr)
        return False

def test_cuda_pytorch():
    """Test PyTorch CUDA functionality."""
    print("\n" + "=" * 60)
    print("Testing PyTorch CUDA Setup")
    print("=" * 60)

    test_code = """
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('CUDA tensor operations: SUCCESS')
else:
    print('CUDA not available - GPU setup failed')
"""

    success, stdout, stderr = run_command([sys.executable, "-c", test_code])

    if success:
        print("CUDA test results:")
        print(stdout)
        return "CUDA available: True" in stdout
    else:
        print("CUDA test failed!")
        print("Error:", stderr)
        return False

def create_gpu_training_script():
    """Create an optimized GPU training script."""
    script_content = '''"""
GPU Training Script for Deepfake Detection
Optimized for CUDA acceleration.
"""

import torch
import subprocess
import sys

def main():
    print("=" * 60)
    print("🚀 GPU Training for Deepfake Detection")
    print("=" * 60)

    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please run setup_gpu.py first.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # Dataset path
    dataset_path = r"C:\\Users\\ranji\\dataset\\deepfake_datase"
    model_path = r"model\\deepfake_video_model.pt"

    print(f"\\nDataset: {dataset_path}")
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

    print("\\nStarting GPU training...")
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
        print("\\nTraining interrupted by user.")

if __name__ == "__main__":
    main()
'''

    with open('gpu_train.py', 'w') as f:
        f.write(script_content)

    print("\nCreated gpu_train.py for optimized GPU training")

def main():
    """Main setup function."""
    print("=" * 60)
    print("NVIDIA GPU Setup for Deepfake Detection")
    print("=" * 60)
    print("This script will set up CUDA acceleration for 8-10x faster training")

    # Check OS
    if platform.system() != 'Windows':
        print("This script is designed for Windows. For Linux/Mac, please follow manual instructions.")
        return

    # Step 1: Check NVIDIA drivers
    if not check_nvidia_drivers():
        print("\nPlease install NVIDIA drivers first, then re-run this script.")
        return

    # Step 2: Check CUDA
    cuda_version = check_cuda_installation()
    if not cuda_version:
        print("\nPlease install CUDA Toolkit first, then re-run this script.")
        return

    # Step 3: Install PyTorch CUDA
    if not install_pytorch_cuda(cuda_version):
        print("\nPyTorch CUDA installation failed.")
        return

    # Step 4: Test setup
    if test_cuda_pytorch():
        print("\nGPU setup completed successfully!")
        print("Your GPU is ready for deepfake detection training!")

        # Create training script
        create_gpu_training_script()

        print("\nReady to train!")
        print("Run: python gpu_train.py")
        print("\nThis will be 8-10x faster than CPU training!")
    else:
        print("\nGPU setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
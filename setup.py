"""
Setup script for the deepfake detection system.
Initializes directories and checks dependencies.
"""

import os
import sys
from pathlib import Path
import subprocess


def create_directories():
    """Create necessary directories."""
    directories = [
        'model',
        'utils',
        'static/sample_outputs',
        'templates',
        'reports',
        'logs',
        'uploads'
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ Directories created successfully!\n")


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"  ❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    print("  This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\n✅ Dependencies installed successfully!\n")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("🔥 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠️  CUDA not available, will use CPU (slower)")
    except ImportError:
        print("  ⚠️  PyTorch not installed yet")


def create_config_files():
    """Create configuration files if they don't exist."""
    print("\n⚙️  Checking configuration files...")
    
    # Create .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# Environment variables\n")
            f.write("DEBUG=False\n")
            f.write("DEVICE=cuda\n")
        print("  ✓ Created .env file")
    else:
        print("  ✓ .env file exists")


def print_summary():
    """Print setup summary."""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\n📖 Quick Start Guide:\n")
    print("1. Install dependencies (if not done):")
    print("   pip install -r requirements.txt\n")
    print("2. Run the web application:")
    print("   streamlit run app.py\n")
    print("3. Train models (optional):")
    print("   python model/train_video_model.py --dataset path/to/dataset\n")
    print("4. Command-line prediction:")
    print("   python utils/predictor.py --file video.mp4 --video_model model/deepfake_video_model.pt\n")
    print("For detailed documentation, see README.md")
    print("=" * 60 + "\n")


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print("🚀 Deepfake Detection System - Setup")
    print("=" * 60 + "\n")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directories()
    
    # Create config files
    create_config_files()
    
    # Ask to install dependencies
    response = input("\n📦 Install dependencies now? (y/n): ").lower().strip()
    if response == 'y':
        install_dependencies()
        check_cuda()
    else:
        print("\n⚠️  Remember to run: pip install -r requirements.txt")
    
    # Print summary
    print_summary()


if __name__ == '__main__':
    main()

"""
Demo script to test the deepfake detection system.
Creates dummy models and runs sample predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from model.video_model import create_video_model
from model.audio_model import create_audio_model


def create_dummy_models():
    """Create dummy models for testing (when no trained models exist)."""
    print("Creating dummy models for testing...")
    
    model_dir = Path('model')
    model_dir.mkdir(exist_ok=True)
    
    # Create video model
    print("\n1. Creating video model...")
    video_model = create_video_model(device='cpu')
    
    # Save dummy checkpoint
    video_checkpoint = {
        'epoch': 0,
        'model_state_dict': video_model.state_dict(),
        'optimizer_state_dict': None,
        'val_loss': 0.5,
        'val_accuracy': 0.75,
        'history': {'train': [], 'val': []}
    }
    
    video_path = model_dir / 'deepfake_video_model.pt'
    torch.save(video_checkpoint, video_path)
    print(f"✓ Video model saved to {video_path}")
    
    # Create audio model
    print("\n2. Creating audio model...")
    audio_model = create_audio_model(device='cpu')
    
    # Save dummy checkpoint
    audio_checkpoint = {
        'epoch': 0,
        'model_state_dict': audio_model.state_dict(),
        'optimizer_state_dict': None,
        'val_loss': 0.5,
        'val_accuracy': 0.70,
        'history': {'train': [], 'val': []}
    }
    
    audio_path = model_dir / 'deepfake_audio_model.pt'
    torch.save(audio_checkpoint, audio_path)
    print(f"✓ Audio model saved to {audio_path}")
    
    print("\n✅ Dummy models created successfully!")
    print("\nNote: These are untrained models and will give random predictions.")
    print("For production use, train models on real datasets.\n")


def test_model_loading():
    """Test if models can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from utils.predictor import DeepfakeDetector
        
        detector = DeepfakeDetector(
            video_model_path='model/deepfake_video_model.pt',
            audio_model_path='model/deepfake_audio_model.pt',
            device='cpu'
        )
        
        print("✓ Models loaded successfully!")
        return True
    
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        return False


def create_sample_data():
    """Create sample video/audio for testing."""
    print("\nCreating sample test data...")
    
    # Create dummy video frames (simulated)
    print("Note: To test with real files, place videos in uploads/ directory")
    
    return True


def run_system_check():
    """Run comprehensive system check."""
    print("\n" + "=" * 60)
    print("🔍 Deepfake Detection System - Demo & System Check")
    print("=" * 60 + "\n")
    
    checks = []
    
    # Check 1: Python packages
    print("1. Checking required packages...")
    try:
        import torch
        import cv2
        import streamlit
        import librosa
        print("✓ Core packages installed")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        checks.append(False)
    
    # Check 2: Directory structure
    print("\n2. Checking directory structure...")
    required_dirs = ['model', 'utils', 'uploads', 'reports', 'logs']
    all_exist = all(Path(d).exists() for d in required_dirs)
    if all_exist:
        print("✓ All required directories exist")
        checks.append(True)
    else:
        print("✗ Some directories missing")
        checks.append(False)
    
    # Check 3: Model files
    print("\n3. Checking model files...")
    video_model_exists = Path('model/deepfake_video_model.pt').exists()
    audio_model_exists = Path('model/deepfake_audio_model.pt').exists()
    
    if not video_model_exists or not audio_model_exists:
        print("⚠️  Model files not found")
        response = input("   Create dummy models for testing? (y/n): ").lower().strip()
        if response == 'y':
            create_dummy_models()
            checks.append(True)
        else:
            print("   Skipping model creation")
            checks.append(False)
    else:
        print("✓ Model files found")
        checks.append(True)
    
    # Check 4: Model loading
    if checks[-1]:
        test_model_loading()
    
    # Check 5: CUDA availability
    print("\n4. Checking CUDA availability...")
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available (will use CPU)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 System Check Summary")
    print("=" * 60)
    print(f"Checks passed: {sum(checks)}/{len(checks)}")
    
    if all(checks):
        print("\n✅ System ready! You can now:")
        print("   1. Run: streamlit run app.py")
        print("   2. Or use CLI: python utils/predictor.py --file <path>")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before running.")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_system_check()

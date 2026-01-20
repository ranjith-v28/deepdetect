"""Package initialization file."""

__version__ = '1.0.0'
__author__ = 'Deepfake Detection Team'
__description__ = 'AI-Powered Deepfake Detection System using CNN-RNN Models'

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# Make imports easier
from .model import create_video_model, create_audio_model
from .utils import VideoPreprocessor, AudioPreprocessor

__all__ = [
    'create_video_model',
    'create_audio_model',
    'VideoPreprocessor',
    'AudioPreprocessor'
]

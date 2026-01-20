"""Configuration file for deepfake detection system."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'model'
UPLOAD_DIR = BASE_DIR / 'uploads'
REPORT_DIR = BASE_DIR / 'reports'
LOG_DIR = BASE_DIR / 'logs'
STATIC_DIR = BASE_DIR / 'static'

# Create directories if they don't exist
for directory in [MODEL_DIR, UPLOAD_DIR, REPORT_DIR, LOG_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model paths
VIDEO_MODEL_PATH = MODEL_DIR / 'deepfake_video_model.pt'
AUDIO_MODEL_PATH = MODEL_DIR / 'deepfake_audio_model.pt'

# File size limits (bytes)
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a'}

# Model parameters
VIDEO_MODEL_CONFIG = {
    'cnn_type': 'efficientnet',  # or 'resnet'
    'feature_dim': 512,
    'lstm_hidden': 256,
    'lstm_layers': 2,
    'num_classes': 2,
    'dropout': 0.3
}

AUDIO_MODEL_CONFIG = {
    'input_channels': 120,  # 40 MFCC + 40 delta + 40 delta-delta
    'gru_hidden': 256,
    'gru_layers': 2,
    'num_classes': 2,
    'dropout': 0.3
}

# Preprocessing parameters
VIDEO_PREPROCESSING = {
    'max_frames': 10,
    'img_size': 224,
    'frame_interval': 3,
    'face_detector': 'mtcnn'
}

AUDIO_PREPROCESSING = {
    'sample_rate': 16000,
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
    'max_duration': 30
}

# Training parameters
TRAINING_CONFIG = {
    'video': {
        'batch_size': 4,
        'learning_rate': 0.0001,
        'num_epochs': 20,
        'optimizer': 'adam'
    },
    'audio': {
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'optimizer': 'adam'
    }
}

# Device configuration
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'

# Security settings
FILE_CLEANUP_HOURS = 24
MAX_UPLOAD_RETRIES = 3

# Visualization settings
VIZ_DPI = 150
VIZ_FIGSIZE = (10, 6)

# Report settings
REPORT_PAGE_SIZE = 'letter'
REPORT_FONT_SIZE = 11

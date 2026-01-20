"""
Security utilities for file validation and logging.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import hashlib

# Optional import for magic (file type detection) - not required for basic validation
try:
    import magic
except ImportError:
    magic = None

logger = logging.getLogger(__name__)


class FileValidator:
    """Validate uploaded files for security."""
    
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB
    
    @staticmethod
    def validate_file(file_path):
        """Validate file type, size, and extension."""
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False, "File does not exist"
            
            # Check file size
            file_size = path.stat().st_size
            ext = path.suffix.lower()
            
            if ext in FileValidator.ALLOWED_VIDEO_EXTENSIONS:
                if file_size > FileValidator.MAX_VIDEO_SIZE:
                    return False, f"Video file too large (max {FileValidator.MAX_VIDEO_SIZE//1024//1024}MB)"
            elif ext in FileValidator.ALLOWED_AUDIO_EXTENSIONS:
                if file_size > FileValidator.MAX_AUDIO_SIZE:
                    return False, f"Audio file too large (max {FileValidator.MAX_AUDIO_SIZE//1024//1024}MB)"
            else:
                return False, f"Unsupported file type: {ext}"
            
            return True, "Valid file"
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, str(e)
    
    @staticmethod
    def sanitize_filename(filename):
        """Sanitize filename to prevent path traversal."""
        # Remove directory paths
        filename = os.path.basename(filename)
        # Remove dangerous characters
        filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
        return filename
    
    @staticmethod
    def get_file_hash(file_path):
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class ActivityLogger:
    """Log user activities for audit trail."""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / 'activity.log'
        
        # Setup logging
        self.logger = logging.getLogger('activity')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_upload(self, filename, file_type, file_size):
        """Log file upload."""
        self.logger.info(
            f"UPLOAD - File: {filename}, Type: {file_type}, Size: {file_size} bytes"
        )
    
    def log_detection(self, filename, result, processing_time):
        """Log detection result."""
        self.logger.info(
            f"DETECTION - File: {filename}, Result: {result.get('label')}, "
            f"Confidence: {result.get('confidence', 0):.2f}, Time: {processing_time:.2f}s"
        )
    
    def log_prediction(self, filename, file_hash, prediction, confidence, processing_time=0.0):
        """Log prediction result."""
        self.logger.info(
            f"PREDICTION - File: {filename}, Hash: {file_hash}, "
            f"Prediction: {prediction}, Confidence: {confidence:.2f}, Time: {processing_time:.2f}s"
        )
    
    def log_error(self, filename, error):
        """Log error."""
        self.logger.error(f"ERROR - File: {filename}, Error: {error}")


def cleanup_old_files(directory, max_age_hours=24):
    """Clean up files older than max_age_hours."""
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
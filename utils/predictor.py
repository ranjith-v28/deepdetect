"""
Enhanced Deepfake Detection Predictor with AI/ML Optimizations
Implements actual inference with model optimization, caching, and ensemble learning.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import time
from datetime import datetime

# Import model classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.video_model import CNNRNNDeepfakeDetector, get_model
from model.audio_model import AudioDeepfakeDetector, get_audio_model
from model.audio_model_simple import SimpleAudioClassifier
from utils.preprocess_video import VideoPreprocessor
from utils.preprocess_audio import AudioPreprocessor
from utils.preprocess_audio_simple import SimpleAudioPreprocessor
from utils.report_generator import SimpleReportGenerator
from utils.optimization import ModelOptimizer, ModelCache, optimize_model_for_deployment

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble detector that combines video and audio predictions.
    """
    
    def __init__(
        self,
        video_model: nn.Module,
        audio_model: nn.Module,
        video_weight: float = 0.6,
        audio_weight: float = 0.4
    ):
        self.video_model = video_model
        self.audio_model = audio_model
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        
    def predict(self, video_input=None, audio_input=None):
        """
        Combine predictions from video and audio models.
        
        Returns:
            prediction: Combined prediction
            confidence: Combined confidence
        """
        if video_input is not None and audio_input is not None:
            # Both modalities available
            video_pred, video_conf = self.video_model.predict(video_input)
            audio_pred, audio_conf = self.audio_model.predict(audio_input)
            
            # Weighted average of confidences
            combined_conf = (video_conf * self.video_weight + audio_conf * self.audio_weight)
            combined_pred = torch.argmax(
                video_pred * self.video_weight + audio_pred * self.audio_weight
            )
            
        elif video_input is not None:
            # Only video
            combined_pred, combined_conf = self.video_model.predict(video_input)
        elif audio_input is not None:
            # Only audio
            combined_pred, combined_conf = self.audio_model.predict(audio_input)
        else:
            raise ValueError("At least one input (video or audio) must be provided")
        
        return combined_pred, combined_conf


class DeepfakeDetector:
    """
    Enhanced deepfake detector with optimizations.
    Implements caching, model optimization, and ensemble learning.
    """
    
    def __init__(
        self,
        video_model_path: Optional[str] = None,
        audio_model_path: Optional[str] = None,
        device: str = 'cpu',
        use_quantization: bool = False,
        use_cache: bool = True,
        max_cache_size: int = 100
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.video_model = None
        self.audio_model = None
        self.video_preprocessor = None
        self.audio_preprocessor = None
        self.use_quantization = use_quantization
        self.ensemble = None
        
        # Initialize cache
        self.cache = ModelCache(max_cache_size) if use_cache else None
        
        # Load models
        if video_model_path and Path(video_model_path).exists():
            self._load_video_model(video_model_path)
        
        if audio_model_path and Path(audio_model_path).exists():
            self._load_audio_model(audio_model_path)
        
        # Initialize ensemble if both models loaded
        if self.video_model and self.audio_model:
            self.ensemble = EnsembleDetector(self.video_model, self.audio_model)
        
        logger.info(f"DeepfakeDetector initialized on {self.device}")
    
    def _load_video_model(self, model_path: str):
        """Load video model."""
        logger.info(f"Loading video model from {model_path}")
        
        # Create model
        self.video_model = get_model(model_type='efficientnet')
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.video_model.load_state_dict(state_dict)
            logger.info("Video model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load video weights: {e}. Using random initialization.")
        
        # Move to device
        self.video_model.to(self.device)
        self.video_model.eval()
        
        # Apply optimizations
        if self.use_quantization:
            optimizer = ModelOptimizer(self.video_model)
            self.video_model = optimizer.quantize_model('dynamic')
        
        # Initialize preprocessor
        self.video_preprocessor = VideoPreprocessor()
    
    def _load_audio_model(self, model_path: str):
        """Load audio model."""
        logger.info(f"Loading audio model from {model_path}")

        # Load state dict first to check model type
        try:
            state_dict = torch.load(model_path, map_location=self.device)
        except Exception as e:
            logger.error(f"Could not load model file: {e}")
            return

        # Try loading as SimpleAudioClassifier first
        try:
            # Check if it's a simple model by looking at the state dict keys
            if any('classifier' in key for key in state_dict.keys()):
                logger.info("Detected SimpleAudioClassifier model")
                self.audio_model = SimpleAudioClassifier()
                self.audio_model.load_state_dict(state_dict)
                self.audio_preprocessor = SimpleAudioPreprocessor()
                logger.info("SimpleAudioClassifier loaded successfully")
            else:
                raise ValueError("Not a simple model")

        except Exception as e:
            # Try loading as original AudioDeepfakeDetector
            try:
                logger.info("Trying to load as AudioDeepfakeDetector")
                self.audio_model = get_audio_model(model_type='gru')
                self.audio_model.load_state_dict(state_dict)
                self.audio_preprocessor = AudioPreprocessor()
                logger.info("AudioDeepfakeDetector loaded successfully")
            except Exception as e2:
                logger.error(f"Could not load audio model: Simple: {e}, Original: {e2}")
                return

        # Move to device
        self.audio_model.to(self.device)
        self.audio_model.eval()

        # Apply optimizations
        if self.use_quantization:
            optimizer = ModelOptimizer(self.audio_model)
            self.audio_model = optimizer.quantize_model('dynamic')
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for caching."""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def detect_video(self, video_path: str) -> Dict:
        """
        Detect deepfakes in video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            result: Detection result dictionary
        """
        if self.video_model is None:
            raise ValueError("Video model not loaded")
        
        start_time = time.time()
        
        # Check cache
        file_hash = self._get_file_hash(video_path)
        cache_key = f"video_{file_hash}"
        
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
        
        # Preprocess video
        logger.info("Preprocessing video...")
        frames = self.video_preprocessor.preprocess_video(video_path)
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Predict
        logger.info("Running video model inference...")
        with torch.no_grad():
            prediction, confidence, probabilities, attention_weights = self.video_model.predict(frames_tensor)
        
        # Convert to numpy
        prediction = prediction.cpu().item()
        confidence = confidence.cpu().item()
        probabilities = probabilities.cpu().numpy()[0]
        
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0],
                'fake': probabilities[1]
            },
            'file_name': Path(video_path).name,
            'file_type': 'video',
            'processing_time': processing_time,
            'frames_processed': len(frames),
            'device': str(self.device)
        }
        
        # Cache result
        if self.cache:
            self.cache.put(cache_key, result)
        
        logger.info(f"Video detection: {result['prediction']} (confidence: {confidence:.2%})")
        
        return result
    
    def detect_audio(self, audio_path: str) -> Dict:
        """
        Detect deepfakes in audio.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            result: Detection result dictionary
        """
        if self.audio_model is None:
            raise ValueError("Audio model not loaded")
        
        start_time = time.time()
        
        # Check cache
        file_hash = self._get_file_hash(audio_path)
        cache_key = f"audio_{file_hash}"
        
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
        
        # Preprocess audio
        logger.info("Preprocessing audio...")
        features = self.audio_preprocessor.preprocess_audio(audio_path)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Predict
        logger.info("Running audio model inference...")
        with torch.no_grad():
            prediction, confidence, probabilities, attention_weights = self.audio_model.predict(features_tensor)
        
        # Convert to numpy
        prediction = prediction.cpu().item()
        confidence = confidence.cpu().item()
        probabilities = probabilities.cpu().numpy()[0]
        
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0],
                'fake': probabilities[1]
            },
            'file_name': Path(audio_path).name,
            'file_type': 'audio',
            'processing_time': processing_time,
            'device': str(self.device)
        }
        
        # Cache result
        if self.cache:
            self.cache.put(cache_key, result)
        
        logger.info(f"Audio detection: {result['prediction']} (confidence: {confidence:.2%})")
        
        return result
    
    def detect(self, file_path: str, generate_report: bool = False, report_path: Optional[str] = None, 
               visualization_dir: Optional[str] = None) -> Dict:
        """
        Auto-detect file type and run appropriate detection.
        
        Args:
            file_path: Path to media file
            generate_report: Whether to generate PDF report
            report_path: Path to save report
            visualization_dir: Directory to save visualizations
        
        Returns:
            result: Detection result dictionary
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Determine file type
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            result = self.detect_video(file_path)
        elif file_ext in ['.wav', '.mp3', '.m4a']:
            result = self.detect_audio(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        
        # Generate report if requested
        if generate_report:
            result['report_path'] = self._generate_report(result, report_path, visualization_dir)
        
        return result
    
    def _generate_report(self, result: Dict, report_path: Optional[str] = None,
                         visualization_dir: Optional[str] = None) -> str:
        """Generate report using SimpleReportGenerator."""
        try:
            generator = SimpleReportGenerator()
            generated_path = generator.generate_report(result, report_path, visualization_dir)
            if generated_path:
                logger.info(f"Report generated successfully: {generated_path}")
                return generated_path
            else:
                logger.error("Report generation failed")
                return None
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            'device': str(self.device),
            'quantization_enabled': self.use_quantization,
            'cache_enabled': self.cache is not None,
            'video_model_loaded': self.video_model is not None,
            'audio_model_loaded': self.audio_model is not None,
            'ensemble_available': self.ensemble is not None
        }
        
        if self.cache:
            info['cache_size'] = self.cache.size()
            info['cache_max_size'] = self.cache.max_size
        
        return info
    
    def clear_cache(self):
        """Clear the cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
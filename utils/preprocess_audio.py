"""
Audio Preprocessing Module
Handles MFCC extraction and preprocessing for audio analysis.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocessor for audio deepfake detection.
    Extracts MFCC features and performs audio preprocessing.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_duration: float = 30.0,
        include_deltas: bool = True
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.include_deltas = include_deltas
        
        logger.info(f"AudioPreprocessor initialized: sample_rate={sample_rate}, n_mfcc={n_mfcc}")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with fallback handling for Windows security policies.

        Args:
            audio_path: Path to audio file

        Returns:
            audio: Audio signal
            sr: Sample rate
        """
        try:
            # Try librosa first (preferred)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {audio_path}, shape={audio.shape}, duration={len(audio)/sr:.2f}s")
            return audio, sr
        except Exception as e:
            error_msg = str(e).lower()
            if "dll load failed" in error_msg or "application control policy" in error_msg:
                logger.warning(f"Librosa blocked by Windows security policy. Trying alternative method for {audio_path}")

                # Fallback: Try soundfile directly
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    # Resample if needed
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
                    logger.info(f"Loaded audio (fallback): {audio_path}, shape={audio.shape}, duration={len(audio)/sr:.2f}s")
                    return audio, sr
                except Exception as e2:
                    logger.warning(f"Soundfile also failed: {e2}")

                    # Last resort: Try pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        # Convert to numpy array
                        audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
                        audio = audio / (2**15)  # Normalize 16-bit audio
                        sr = audio_segment.frame_rate
                        # Resample if needed
                        if sr != self.sample_rate:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                            sr = self.sample_rate
                        logger.info(f"Loaded audio (pydub fallback): {audio_path}, shape={len(audio)}, duration={len(audio)/sr:.2f}s")
                        return audio, sr
                    except Exception as e3:
                        logger.error(f"All audio loading methods failed for {audio_path}: Librosa: {e}, Soundfile: {e2}, Pydub: {e3}")
                        raise ValueError(f"Error loading audio file (all methods failed): {e}")

            else:
                # Not a DLL issue, re-raise the original error
                logger.error(f"Error loading audio: {e}")
                raise ValueError(f"Error loading audio file: {e}")
    
    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silence from beginning and end of audio.
        
        Args:
            audio: Audio signal
        
        Returns:
            trimmed_audio: Audio with silence removed
        """
        # Trim leading and trailing silence
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        return trimmed
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio signal
        
        Returns:
            normalized_audio: Normalized audio
        """
        return librosa.util.normalize(audio)
    
    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio to max_duration.
        
        Args:
            audio: Audio signal
        
        Returns:
            processed_audio: Audio of fixed length
        """
        target_length = int(self.max_duration * self.sample_rate)
        current_length = len(audio)
        
        if current_length > target_length:
            # Truncate
            return audio[:target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            padded = np.pad(audio, (0, padding), mode='constant', constant_values=0)
            return padded
        else:
            return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal
        
        Returns:
            mfcc: MFCC features (n_mfcc, time)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def extract_deltas(self, mfcc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract delta and delta-delta features.
        
        Args:
            mfcc: MFCC features
        
        Returns:
            delta: First-order delta features
            delta2: Second-order delta-delta features
        """
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        return delta, delta2
    
    def combine_features(self, mfcc: np.ndarray, delta: np.ndarray, delta2: np.ndarray) -> np.ndarray:
        """
        Combine MFCC, delta, and delta-delta features.
        
        Args:
            mfcc: MFCC features
            delta: Delta features
            delta2: Delta-delta features
        
        Returns:
            combined: Combined features (n_mfcc*3, time)
        """
        combined = np.concatenate([mfcc, delta, delta2], axis=0)
        return combined
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance.
        
        Args:
            features: Input features
        
        Returns:
            normalized_features: Normalized features
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized = (features - mean) / std
        
        return normalized
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline for audio.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            features: Preprocessed features (n_channels, time_steps)
        """
        # Load audio
        audio, _ = self.load_audio(audio_path)
        
        # Trim silence
        audio = self.trim_silence(audio)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Pad or truncate
        audio = self.pad_or_truncate(audio)
        
        # Extract MFCC
        mfcc = self.extract_mfcc(audio)
        
        # Extract deltas if enabled
        if self.include_deltas:
            delta, delta2 = self.extract_deltas(mfcc)
            features = self.combine_features(mfcc, delta, delta2)
        else:
            features = mfcc
        
        # Normalize features
        features = self.normalize_features(features)
        
        logger.info(f"Preprocessed audio: features shape={features.shape}")
        
        return features
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get audio metadata.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            info: Dictionary with audio information
        """
        audio, sr = self.load_audio(audio_path)
        
        info = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'n_samples': len(audio),
            'channels': 1  # librosa loads as mono
        }
        
        return info
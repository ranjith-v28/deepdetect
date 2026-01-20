"""
Simplified Audio Preprocessor
Works around Windows security restrictions by avoiding blocked libraries.
Uses basic waveform analysis instead of advanced MFCC features.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleAudioPreprocessor:
    """
    Simplified audio preprocessor that avoids Windows security restrictions.
    Uses basic waveform features instead of MFCC/spectrogram analysis.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        feature_dim: int = 120
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_dim = feature_dim
        self.target_length = int(max_duration * sample_rate)

        logger.info(f"SimpleAudioPreprocessor initialized: sample_rate={sample_rate}, max_duration={max_duration}s")

    def load_audio_simple(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio using basic file reading (avoids blocked libraries).

        Args:
            audio_path: Path to audio file

        Returns:
            audio: Audio signal as numpy array
            sample_rate: Detected or assumed sample rate
        """
        try:
            # Use pydub as primary method (most reliable)
            from pydub import AudioSegment

            # Load audio file
            audio_segment = AudioSegment.from_file(audio_path)

            # Convert to mono if stereo
            if audio_segment.channels == 2:
                audio_segment = audio_segment.set_channels(1)

            # Convert to numpy array
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            sample_rate = audio_segment.frame_rate

            logger.info(f"Loaded audio: {audio_path}, samples={len(audio)}, rate={sample_rate}")
            return audio, sample_rate

        except Exception as e:
            logger.warning(f"Pydub failed for {audio_path}, trying basic WAV reader: {e}")

            # Fallback: Basic WAV reader for .wav files only
            try:
                import wave
                import struct

                with wave.open(audio_path, 'rb') as wav_file:
                    # Get basic info
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    # Read raw audio data
                    raw_data = wav_file.readframes(n_frames)

                    # Convert based on sample width
                    if sample_width == 2:  # 16-bit
                        audio = np.frombuffer(raw_data, dtype=np.int16)
                        audio = audio.astype(np.float32) / 32768.0
                    elif sample_width == 4:  # 32-bit
                        audio = np.frombuffer(raw_data, dtype=np.int32)
                        audio = audio.astype(np.float32) / 2147483648.0
                    else:
                        audio = np.frombuffer(raw_data, dtype=np.int8)
                        audio = audio.astype(np.float32) / 128.0

                    # Convert stereo to mono if needed
                    if n_channels == 2:
                        audio = audio.reshape(-1, 2).mean(axis=1)

                    logger.info(f"Loaded WAV: {audio_path}, samples={len(audio)}, rate={framerate}")
                    return audio, framerate

            except Exception as e2:
                logger.error(f"All audio loading methods failed for {audio_path}: Pydub: {e}, WAV: {e2}")
                # Return dummy audio
                return np.zeros(self.target_length, dtype=np.float32), self.sample_rate

    def extract_basic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract basic waveform features that don't require blocked libraries.

        Args:
            audio: Audio signal

        Returns:
            features: Basic features array
        """
        features = []

        # 1. Statistical features
        features.extend([
            np.mean(audio),           # Mean amplitude
            np.std(audio),            # Standard deviation
            np.max(audio),            # Maximum amplitude
            np.min(audio),            # Minimum amplitude
            np.median(audio),         # Median amplitude
            np.sqrt(np.mean(audio**2)),  # RMS energy
        ])

        # 2. Zero crossing rate (simple)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)
        features.append(zero_crossings)

        # 3. Basic spectral centroid approximation
        # Simple frequency content estimation
        if len(audio) > 100:
            # Simple autocorrelation for pitch estimation
            autocorr = np.correlate(audio[:1000], audio[:1000], mode='full')
            peak_idx = np.argmax(autocorr[len(autocorr)//2:]) + len(autocorr)//2
            if peak_idx > 0:
                fundamental_freq = self.sample_rate / peak_idx
                features.append(fundamental_freq / 1000)  # Normalize
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        # 4. Temporal features
        # Divide audio into segments and get statistics
        n_segments = 10
        segment_length = len(audio) // n_segments

        if segment_length > 0:
            for i in range(n_segments):
                start = i * segment_length
                end = min((i + 1) * segment_length, len(audio))
                segment = audio[start:end]

                if len(segment) > 0:
                    features.extend([
                        np.mean(segment),
                        np.std(segment),
                        np.max(segment) - np.min(segment),  # Peak-to-peak
                    ])

        # Pad or truncate to fixed feature dimension
        features = np.array(features, dtype=np.float32)

        if len(features) < self.feature_dim:
            # Pad with zeros
            padding = np.zeros(self.feature_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        else:
            # Truncate
            features = features[:self.feature_dim]

        return features

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio to fixed length.

        Args:
            audio: Audio signal

        Returns:
            processed_audio: Audio of fixed length
        """
        current_length = len(audio)

        if current_length > self.target_length:
            # Truncate
            return audio[:self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            padding = np.zeros(self.target_length - current_length, dtype=np.float32)
            return np.concatenate([audio, padding])
        else:
            return audio

    def preprocess_audio_simple(self, audio_path: str) -> np.ndarray:
        """
        Complete simplified preprocessing pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            features: Simplified features array
        """
        try:
            # Load audio
            audio, sr = self.load_audio_simple(audio_path)

            # Resample if needed (simple method)
            if sr != self.sample_rate:
                # Simple downsampling by integer factor
                ratio = sr // self.sample_rate
                if ratio > 1:
                    audio = audio[::ratio]

            # Pad or truncate
            audio = self.pad_or_truncate(audio)

            # Extract basic features
            features = self.extract_basic_features(audio)

            return features

        except Exception as e:
            logger.error(f"Error in simplified preprocessing for {audio_path}: {e}")
            # Return zero features as fallback
            return np.zeros(self.feature_dim, dtype=np.float32)
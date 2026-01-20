"""
Video Preprocessing Module
Handles frame extraction, face detection, and preprocessing for video analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Preprocessor for video deepfake detection.
    Extracts frames and performs face detection.
    """
    
    def __init__(
        self,
        max_frames: int = 10,
        img_size: int = 224,
        frame_interval: int = 3,
        face_detector: str = 'mtcnn',
        detect_faces: bool = True
    ):
        self.max_frames = max_frames
        self.img_size = img_size
        self.frame_interval = frame_interval
        self.face_detector = face_detector
        self.detect_faces = detect_faces
        
        # Initialize face detector
        self.face_cascade = None
        if face_detector == 'opencv':
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        logger.info(f"VideoPreprocessor initialized: max_frames={max_frames}, img_size={img_size}")
    
    def extract_frames(
        self,
        video_path: str,
        return_rgb: bool = True
    ) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            return_rgb: Whether to convert BGR to RGB
        
        Returns:
            frames: List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frames at intervals
                if frame_count % self.frame_interval == 0:
                    if return_rgb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    extracted_count += 1
                    
                    if extracted_count >= self.max_frames:
                        break
                
                frame_count += 1
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            
        finally:
            cap.release()
        
        return frames
    
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop face from frame.
        
        Args:
            frame: Input frame
        
        Returns:
            face: Cropped face or None if no face detected
        """
        if not self.detect_faces:
            return frame
        
        if self.face_detector == 'opencv':
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add padding
                padding = int(0.2 * w)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                face = frame[y:y+h, x:x+w]
                return face
        
        return None
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame
        
        Returns:
            resized_frame: Resized frame
        """
        return cv2.resize(frame, (self.img_size, self.img_size))
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame pixel values.
        
        Args:
            frame: Input frame (0-255)
        
        Returns:
            normalized_frame: Normalized frame (0-1)
        """
        return frame.astype(np.float32) / 255.0
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single frame.
        
        Args:
            frame: Input frame
        
        Returns:
            processed_frame: Preprocessed frame
        """
        # Detect and crop face
        face = self.detect_face(frame)
        
        if face is None:
            # If no face detected, use center crop
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            start_y = (h - min_dim) // 2
            start_x = (w - min_dim) // 2
            face = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Resize
        resized = self.resize_frame(face)
        
        # Normalize
        normalized = self.normalize_frame(resized)
        
        return normalized
    
    def preprocess_video(self, video_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline for video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            processed_frames: Preprocessed frames as numpy array
                              Shape: (num_frames, H, W, C)
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            logger.error("No frames extracted from video")
            raise ValueError("No frames extracted from video")
        
        # Preprocess each frame
        processed_frames = []
        for frame in frames:
            processed = self.preprocess_frame(frame)
            processed_frames.append(processed)
        
        # Convert to numpy array
        processed_array = np.array(processed_frames)
        
        logger.info(f"Preprocessed video: shape={processed_array.shape}")
        
        return processed_array
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
        
        Returns:
            info: Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        
        return info
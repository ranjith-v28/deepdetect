"""
Simplified Audio Model Training
Uses basic waveform features instead of MFCC to avoid Windows security restrictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.preprocess_audio_simple import SimpleAudioPreprocessor
from model.audio_model_simple import SimpleAudioClassifier

logger = logging.getLogger(__name__)

class SimpleAudioDataset(Dataset):
    """Simplified dataset for basic audio features."""

    def __init__(
        self,
        data_dir: str,
        real_dir: str = 'real',
        fake_dir: str = 'fake',
        max_duration: float = 30.0,
        feature_dim: int = 120
    ):
        self.data_dir = Path(data_dir)
        self.samples = []

        # Collect audio file paths
        audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']

        real_path = self.data_dir / real_dir
        fake_path = self.data_dir / fake_dir

        if real_path.exists():
            for ext in audio_extensions:
                for audio_file in real_path.glob(ext):
                    self.samples.append((str(audio_file), 0))  # 0 = real

        if fake_path.exists():
            for ext in audio_extensions:
                for audio_file in fake_path.glob(ext):
                    self.samples.append((str(audio_file), 1))  # 1 = fake

        logger.info(f"Loaded {len(self.samples)} audio samples")

        self.preprocessor = SimpleAudioPreprocessor(
            max_duration=max_duration,
            feature_dim=feature_dim
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        try:
            # Extract simplified features
            features = self.preprocessor.preprocess_audio_simple(audio_path)

            # Convert to tensor
            features_tensor = torch.from_numpy(features).float()

            return features_tensor, label

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            # Return zero features as fallback
            return torch.zeros(self.preprocessor.feature_dim, dtype=torch.float32), label

class SimpleAudioTrainer:
    """Simplified trainer for basic audio classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Training history
        self.history = {'train': [], 'val': []}

        logger.info(f"SimpleAudioTrainer initialized on {device}")

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, _ = self.model(features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate_epoch(self, epoch: int) -> Dict:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {'loss': 0, 'accuracy': 0}

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.model(features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train'].append(train_metrics)

            # Validate
            val_metrics = self.validate_epoch(epoch)
            self.history['val'].append(val_metrics)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['loss'] < best_val_loss and val_metrics['loss'] > 0:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'])

            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%"
            )

        # Save training history
        self.save_history()

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / 'best_audio_model.pt'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved best model with val_loss: {val_loss:.4f}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / 'audio_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train simplified audio deepfake detection model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--save_path', type=str, default='model/simple_audio_model.pt', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max_duration', type=float, default=10.0, help='Maximum audio duration')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating simplified audio model...")
    model = SimpleAudioClassifier()

    # Create datasets
    logger.info(f"Loading dataset from {args.dataset}...")
    full_dataset = SimpleAudioDataset(
        args.dataset,
        max_duration=args.max_duration
    )

    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Create trainer
    trainer = SimpleAudioTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        save_dir=str(Path(args.save_path).parent)
    )

    # Train
    trainer.train(num_epochs=args.epochs)

    # Save final model
    torch.save(model.state_dict(), args.save_path)
    logger.info(f"Final model saved to {args.save_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
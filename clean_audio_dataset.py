"""
Audio Dataset Cleaner
Identifies and optionally removes corrupted audio files from the dataset.
"""

import os
import sys
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils.preprocess_audio_simple import SimpleAudioPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDatasetCleaner:
    """Clean corrupted audio files from dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.preprocessor = SimpleAudioPreprocessor()
        self.corrupted_files = []
        self.total_checked = 0

    def check_file(self, audio_path: str) -> tuple[str, bool]:
        """Check if an audio file can be loaded successfully."""
        try:
            # Try to load the file
            audio, sr = self.preprocessor.load_audio_simple(audio_path)

            # Check if we got dummy data (all zeros)
            if audio is not None and len(audio) > 0 and not (audio == 0).all():
                return audio_path, True  # File is good
            else:
                return audio_path, False  # File is corrupted (dummy data returned)
        except Exception as e:
            logger.debug(f"Error checking {audio_path}: {e}")
            return audio_path, False  # File is corrupted

    def scan_directory(self, directory: str, max_workers: int = 4) -> list[str]:
        """Scan directory for corrupted audio files."""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return []

        # Get all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(dir_path.glob(ext))

        logger.info(f"Found {len(audio_files)} audio files in {directory}")

        corrupted_files = []

        # Check files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.check_file, str(audio_file)) for audio_file in audio_files]

            for i, future in enumerate(as_completed(futures)):
                audio_path, is_valid = future.result()
                self.total_checked += 1

                if not is_valid:
                    corrupted_files.append(audio_path)
                    logger.warning(f"Corrupted: {audio_path}")
                else:
                    logger.debug(f"Valid: {Path(audio_path).name}")

                # Progress update
                if (i + 1) % 100 == 0:
                    logger.info(f"Checked {i + 1}/{len(audio_files)} files, found {len(corrupted_files)} corrupted")

        return corrupted_files

    def clean_dataset(self, remove_files: bool = False, max_workers: int = 4) -> dict:
        """Clean the entire dataset."""
        results = {
            'real': {'total': 0, 'corrupted': 0, 'removed': 0},
            'fake': {'total': 0, 'corrupted': 0, 'removed': 0}
        }

        for category in ['real', 'fake']:
            category_path = self.dataset_path / category
            if not category_path.exists():
                logger.warning(f"Category directory {category_path} does not exist")
                continue

            logger.info(f"Scanning {category} directory...")
            corrupted = self.scan_directory(str(category_path), max_workers)

            results[category]['total'] = len(list(category_path.glob('*')))
            results[category]['corrupted'] = len(corrupted)

            if remove_files and corrupted:
                logger.info(f"Removing {len(corrupted)} corrupted files from {category}...")
                for corrupted_file in corrupted:
                    try:
                        os.remove(corrupted_file)
                        results[category]['removed'] += 1
                        logger.info(f"Removed: {corrupted_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove {corrupted_file}: {e}")

        return results

def main():
    """Main cleaning function."""
    import argparse

    parser = argparse.ArgumentParser(description='Clean corrupted audio files from dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--remove', action='store_true', help='Actually remove corrupted files')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')

    args = parser.parse_args()

    cleaner = AudioDatasetCleaner(args.dataset)

    logger.info("Starting audio dataset cleaning...")
    logger.info(f"Dataset path: {args.dataset}")
    logger.info(f"Remove corrupted files: {args.remove}")

    results = cleaner.clean_dataset(remove_files=args.remove, max_workers=args.workers)

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("CLEANING SUMMARY")
    logger.info("="*50)

    total_corrupted = 0
    total_removed = 0

    for category, stats in results.items():
        logger.info(f"{category.upper()} DIRECTORY:")
        logger.info(f"  Total files: {stats['total']}")
        logger.info(f"  Corrupted files: {stats['corrupted']}")
        logger.info(f"  Removed files: {stats['removed']}")
        logger.info("")

        total_corrupted += stats['corrupted']
        total_removed += stats['removed']

    logger.info(f"TOTAL CORRUPTED FILES: {total_corrupted}")
    logger.info(f"TOTAL REMOVED FILES: {total_removed}")

    if not args.remove:
        logger.info("\nTo actually remove corrupted files, run with --remove flag")
        logger.info("Example: python clean_audio_dataset.py --dataset \"C:\\Users\\ranji\\dataset\\deepfake_audio\" --remove")

if __name__ == '__main__':
    main()
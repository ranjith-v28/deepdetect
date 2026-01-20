# Custom Training Script for User's Deepfake Dataset
# Uses the dataset at C:\Users\ranji\dataset\deepfake_datase

import subprocess
import sys
import os

# Dataset path (corrected from user's typo)
DATASET_PATH = "C:\\Users\\ranji\\dataset\\deepfake_datase"
MODEL_SAVE_PATH = "model\\deepfake_video_model.pt"

# Training parameters
TRAINING_ARGS = [
    "--dataset", DATASET_PATH,
    "--save_path", MODEL_SAVE_PATH,
    "--epochs", "20",
    "--batch_size", "4",
    "--lr", "0.0001",
    "--device", "cuda",  # Use 'cpu' if no GPU
    "--max_frames", "10"
]

def main():
    """Run the training with user's dataset."""
    print("=" * 60)
    print("Training Video Model with Your Dataset")
    print("=" * 60)
    print()

    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"💾 Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"🎯 Training for {TRAINING_ARGS[TRAINING_ARGS.index('--epochs') + 1]} epochs")
    print(f"📦 Batch size: {TRAINING_ARGS[TRAINING_ARGS.index('--batch_size') + 1]}")
    print(f"⚡ Device: {TRAINING_ARGS[TRAINING_ARGS.index('--device') + 1]}")
    print()

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please check the path and try again.")
        return

    if not os.path.exists(os.path.join(DATASET_PATH, "real")):
        print(f"Error: 'real' folder not found in {DATASET_PATH}")
        return

    if not os.path.exists(os.path.join(DATASET_PATH, "fake")):
        print(f"Error: 'fake' folder not found in {DATASET_PATH}")
        return

        print("Dataset structure verified!")
        print()

        # Run training
        print("Starting training...")
    print()

    cmd = [sys.executable, "model/train_video_model.py"] + TRAINING_ARGS

    try:
        subprocess.run(cmd, check=True)
        print()
        print("Training completed successfully!")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print()
        print("You can now use this model in the web application!")
        print("Run: streamlit run app.py")

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print("Check the error messages above for details.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
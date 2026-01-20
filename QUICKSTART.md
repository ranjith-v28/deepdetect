# Deepfake Detection System

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py
```

### 3. Access the Application
Open your browser and navigate to: `http://localhost:8501`

### 4. Upload and Analyze
- Click "Browse files" to upload a video or audio file
- Click "Analyze File" button
- View results and download PDF report

## Training Models

If you want to train your own models:

### Video Model
```bash
python model/train_video_model.py --dataset path/to/dataset --epochs 20 --device cuda
```

### Audio Model
```bash
python model/train_audio_model.py --dataset path/to/dataset --epochs 30 --device cuda
```

## Command-Line Usage

```bash
python utils/predictor.py --file video.mp4 --video_model model/deepfake_video_model.pt --report
```

## Supported File Formats

**Video:** .mp4, .avi, .mov, .mkv  
**Audio:** .wav, .mp3, .m4a

## File Size Limits

**Video:** Max 100 MB  
**Audio:** Max 50 MB

## System Requirements

- Python 3.10+
- 8GB RAM minimum
- GPU recommended (CUDA support)
- 10GB disk space

## Troubleshooting

### CUDA Out of Memory
Reduce `batch_size` in training scripts or use CPU mode.

### No Module Named 'X'
```bash
pip install -r requirements.txt
```

### Slow Performance
Use GPU if available, or reduce `max_frames` in preprocessing.

For detailed documentation, see README.md

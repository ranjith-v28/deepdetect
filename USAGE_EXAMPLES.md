# 🎬 USAGE EXAMPLES

## Complete Guide to Using the Deepfake Detection System

---

## 🌐 Method 1: Web Interface (Easiest)

### Launch the Application

```bash
# Navigate to project directory
cd c:\Users\STUDENT\Downloads\project\deepfake_detection

# Run the web app
streamlit run app.py
```

### Using the Web Interface

1. **Open Browser** → Navigate to `http://localhost:8501`

2. **Upload File**
   - Click "Browse files" button
   - Select a video (.mp4, .avi) or audio (.wav, .mp3) file
   - Max size: 100MB for video, 50MB for audio

3. **Analyze**
   - Click "🚀 Analyze File" button
   - Wait for processing (10-60 seconds)

4. **View Results**
   - See prediction: "Real" or "Fake"
   - View confidence score
   - Explore visualizations
   - Check sample frames/spectrograms

5. **Download Report**
   - Go to "📄 Report" tab
   - Click "📥 Download PDF Report"

---

## 💻 Method 2: Command Line

### Basic Prediction

```bash
# Video prediction
python utils/predictor.py \
    --file path/to/video.mp4 \
    --video_model model/deepfake_video_model.pt

# Audio prediction
python utils/predictor.py \
    --file path/to/audio.wav \
    --audio_model model/deepfake_audio_model.pt
```

### With Report Generation

```bash
python utils/predictor.py \
    --file video.mp4 \
    --video_model model/deepfake_video_model.pt \
    --report \
    --device cuda
```

### Output Example

```
==================================================
DETECTION RESULT
==================================================
File: sample_video.mp4
Type: video
Prediction: Fake
Confidence: 92.37%
Real probability: 7.63%
Fake probability: 92.37%
Report saved to: reports/detection_report_sample_video.pdf
==================================================
```

---

## 🐍 Method 3: Python API

### Basic Usage

```python
from utils.predictor import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(
    video_model_path='model/deepfake_video_model.pt',
    audio_model_path='model/deepfake_audio_model.pt',
    device='cuda'  # or 'cpu'
)

# Analyze a video
result = detector.detect_video('path/to/video.mp4')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
```

### Advanced Usage with Reports

```python
from utils.predictor import DeepfakeDetector

detector = DeepfakeDetector(
    video_model_path='model/deepfake_video_model.pt',
    device='cuda'
)

# Full analysis with report
result = detector.detect(
    file_path='suspicious_video.mp4',
    file_type='video',  # or 'audio', or None for auto-detect
    generate_report=True,
    report_path='reports/my_report.pdf',
    visualization_dir='static/sample_outputs'
)

# Access results
if 'error' not in result:
    print(f"✅ File: {result['file_name']}")
    print(f"📊 Prediction: {result['prediction']}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print(f"📈 Probabilities:")
    print(f"   Real: {result['probabilities']['real']:.2%}")
    print(f"   Fake: {result['probabilities']['fake']:.2%}")
    
    if 'report_path' in result:
        print(f"📄 Report: {result['report_path']}")
else:
    print(f"❌ Error: {result['error']}")
```

### Batch Processing

```python
from pathlib import Path
from utils.predictor import DeepfakeDetector

detector = DeepfakeDetector(
    video_model_path='model/deepfake_video_model.pt'
)

# Process multiple files
video_dir = Path('path/to/videos')
results = []

for video_file in video_dir.glob('*.mp4'):
    print(f"Processing: {video_file.name}")
    result = detector.detect_video(str(video_file))
    results.append({
        'file': video_file.name,
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })

# Print summary
for r in results:
    print(f"{r['file']}: {r['prediction']} ({r['confidence']:.2%})")
```

---

## 🎓 Training Custom Models

### Prepare Your Dataset

```
my_dataset/
├── real/
│   ├── real_video_001.mp4
│   ├── real_video_002.mp4
│   └── ...
└── fake/
    ├── fake_video_001.mp4
    ├── fake_video_002.mp4
    └── ...
```

### Train Video Model

```bash
python model/train_video_model.py \
    --dataset path/to/my_dataset \
    --save_path model/my_video_model.pt \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.0001 \
    --device cuda
```

### Train Audio Model

```bash
python model/train_audio_model.py \
    --dataset path/to/audio_dataset \
    --save_path model/my_audio_model.pt \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --device cuda
```

### Monitor Training

```python
import json

# Load training history
with open('model/my_video_model_history.json', 'r') as f:
    history = json.load(f)

# Check final metrics
final_epoch = history['val'][-1]
print(f"Final Validation Accuracy: {final_epoch['accuracy']:.4f}")
print(f"Final Validation F1: {final_epoch['f1']:.4f}")
```

---

## 🔧 Advanced Configuration

### Custom Model Parameters

```python
from model.video_model import CNNRNNDeepfakeDetector

# Create custom model
custom_model = CNNRNNDeepfakeDetector(
    cnn_type='resnet',        # or 'efficientnet'
    feature_dim=768,          # increase for more capacity
    lstm_hidden=512,          # increase for more memory
    lstm_layers=3,            # deeper network
    num_classes=2,
    dropout=0.4               # adjust for regularization
)
```

### Custom Preprocessing

```python
from utils.preprocess_video import VideoPreprocessor

# Custom video preprocessor
preprocessor = VideoPreprocessor(
    face_detector='mtcnn',
    img_size=299,             # larger images
    max_frames=20             # more frames
)

# Process video
tensor, metadata, frames = preprocessor.process_video(
    'video.mp4',
    frame_interval=2          # every 2nd frame
)
```

---

## 📊 Visualization Examples

### Generate Custom Visualizations

```python
from utils.visualization import (
    plot_confidence_bar,
    plot_frame_grid,
    plot_mel_spectrogram
)
import matplotlib.pyplot as plt

# Confidence bar
fig = plot_confidence_bar(
    prediction='Fake',
    confidence=0.92,
    save_path='output/confidence.png'
)
plt.close()

# Frame grid
fig = plot_frame_grid(
    frames=video_frames,
    max_frames=9,
    save_path='output/frames.png'
)
plt.close()

# Spectrogram
fig = plot_mel_spectrogram(
    audio=audio_waveform,
    sr=16000,
    save_path='output/spectrogram.png'
)
plt.close()
```

### Create Custom PDF Report

```python
from utils.report_generator import create_detection_report

success = create_detection_report(
    output_path='reports/custom_report.pdf',
    file_name='test_video.mp4',
    file_type='video',
    prediction='Fake',
    confidence=0.92,
    visualization_paths={
        'confidence': 'output/confidence.png',
        'frames': 'output/frames.png'
    },
    metrics={
        'accuracy': 0.89,
        'precision': 0.91,
        'recall': 0.87,
        'f1_score': 0.89
    }
)
```

---

## 🛡️ Security Best Practices

### File Validation

```python
from utils.security import FileValidator

validator = FileValidator()

# Validate file
is_valid, message = validator.validate_file(
    file_path='uploaded_video.mp4',
    expected_type='video'
)

if is_valid:
    # Safe to process
    print(f"✓ {message}")
else:
    # Reject file
    print(f"✗ {message}")
```

### Activity Logging

```python
from utils.security import ActivityLogger

logger = ActivityLogger(log_dir='logs')

# Log upload
logger.log_upload(
    filename='video.mp4',
    file_size=15728640,
    file_hash='abc123...',
    user_ip='127.0.0.1'
)

# Log prediction
logger.log_prediction(
    filename='video.mp4',
    file_hash='abc123...',
    prediction='Fake',
    confidence=0.92,
    processing_time=23.5,
    user_ip='127.0.0.1'
)

# Get recent logs
recent = logger.get_recent_logs(n=10)
for log in recent:
    print(log)
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
python model/train_video_model.py --batch_size 2

# Solution 2: Use CPU
python utils/predictor.py --file video.mp4 --device cpu
```

### Issue: No Faces Detected

```python
# Increase detection threshold or try different detector
preprocessor = VideoPreprocessor(
    face_detector='mtcnn',
    max_frames=50  # Try more frames
)
```

### Issue: Slow Performance

```bash
# Use GPU if available
python utils/predictor.py --file video.mp4 --device cuda

# Reduce number of frames
# Edit config.py:
VIDEO_PREPROCESSING = {
    'max_frames': 5,        # Reduce from 10
    'frame_interval': 5      # Skip more frames
}
```

---

## 📞 Getting Help

1. **Check Documentation**: See `README.md` and `QUICKSTART.md`
2. **Run Demo**: Execute `python demo.py` for system check
3. **Check Logs**: Review `logs/activity.log` for errors
4. **Verify Models**: Ensure model files exist in `model/` directory

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Run web app | `streamlit run app.py` |
| Predict video | `python utils/predictor.py --file video.mp4 --video_model model/deepfake_video_model.pt` |
| Train model | `python model/train_video_model.py --dataset path/to/data` |
| System check | `python demo.py` |
| Create dummy models | `python demo.py` (answer 'y' when prompted) |

---

<p align="center">
  <strong>Happy Detecting! 🔍</strong>
</p>

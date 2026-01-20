# Quick Start Guide - AI/ML Enhanced DeepDetect

This guide helps you get started with the enhanced AI/ML features of DeepDetect.

---

## 🎯 What's New?

The DeepDetect system now includes:
- ✅ Real CNN-RNN and GRU deep learning models
- ✅ Model quantization (2-4x faster inference)
- ✅ Smart caching (instant repeated results)
- ✅ Ensemble learning (video + audio combined)
- ✅ GPU acceleration with mixed precision training
- ✅ Comprehensive training scripts

---

## 🚀 Quick Start

### Option 1: Test with Dummy Models (Fastest)

If you just want to test the UI without training:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The system will work with simulated results for demonstration.

---

### Option 2: Train Your Own Models (Recommended)

#### Step 1: Prepare Dataset

Organize your data like this:

```
dataset/
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fake/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

**Recommended Datasets:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics) - Video deepfakes
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) - High-quality deepfakes
- [DFDC](https://ai.facebook.com/datasets/dfdc/) - Large-scale dataset

#### Step 2: Train Video Model

```bash
# Basic training (CPU)
python model/train_video_model.py \
    --dataset path/to/video_dataset \
    --save_path model/deepfake_video_model.pt \
    --epochs 20 \
    --batch_size 4 \
    --device cpu

# GPU training (recommended - much faster)
python model/train_video_model.py \
    --dataset path/to/video_dataset \
    --save_path model/deepfake_video_model.pt \
    --epochs 20 \
    --batch_size 8 \
    --device cuda
```

**Training Parameters:**
- `--epochs`: Number of training passes (20-50 recommended)
- `--batch_size`: Batch size (4-8 for video, depends on GPU memory)
- `--lr`: Learning rate (default: 0.0001)
- `--max_frames`: Frames per video (default: 10)

#### Step 3: Train Audio Model

```bash
# Basic training (CPU)
python model/train_audio_model.py \
    --dataset path/to/audio_dataset \
    --save_path model/deepfake_audio_model.pt \
    --epochs 30 \
    --batch_size 16 \
    --device cpu

# GPU training (recommended)
python model/train_audio_model.py \
    --dataset path/to/audio_dataset \
    --save_path model/deepfake_audio_model.pt \
    --epochs 30 \
    --batch_size 32 \
    --device cuda
```

**Training Parameters:**
- `--epochs`: Number of training passes (30-100 recommended)
- `--batch_size`: Batch size (16-32 for audio)
- `--lr`: Learning rate (default: 0.001)
- `--sample_rate`: Audio sample rate (default: 16000 Hz)

#### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will automatically load your trained models and provide real predictions!

---

## 🎓 Using the Enhanced Predictor

### Basic Usage

```python
from utils.predictor import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(
    video_model_path='model/deepfake_video_model.pt',
    audio_model_path='model/deepfake_audio_model.pt',
    device='cuda'  # or 'cpu'
)

# Detect video
result = detector.detect_video('video.mp4')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Processing time: {result['processing_time']:.2f}s")

# Detect audio
result = detector.detect_audio('audio.wav')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### With Optimizations

```python
# Enable quantization (2-4x faster)
detector = DeepfakeDetector(
    video_model_path='model/deepfake_video_model.pt',
    audio_model_path='model/deepfake_audio_model.pt',
    device='cuda',
    use_quantization=True,  # Enable quantization
    use_cache=True          # Enable caching
)

# First run (slower)
result = detector.detect('video.mp4')
print(f"First run: {result['processing_time']:.2f}s")

# Second run (instant - from cache)
result = detector.detect('video.mp4')
print(f"Second run: {result['processing_time']:.2f}s")
```

### Batch Processing

```python
from pathlib import Path

# Process all videos in a directory
video_dir = Path('test_videos')
for video_file in video_dir.glob('*.mp4'):
    result = detector.detect_video(str(video_file))
    print(f"{video_file.name}: {result['prediction']} ({result['confidence']:.2%})")
```

---

## 🔧 Optimization Tips

### For Faster Training

1. **Use GPU** (if available)
   ```bash
   --device cuda
   ```

2. **Use Mixed Precision** (automatic in training scripts)
   - 2x faster training
   - Minimal accuracy loss

3. **Increase Batch Size** (if GPU memory allows)
   ```bash
   --batch_size 8  # or higher for GPU
   ```

4. **Reduce Frames** (for faster preprocessing)
   ```bash
   --max_frames 5  # instead of 10
   ```

### For Faster Inference

1. **Enable Quantization**
   ```python
   use_quantization=True
   ```
   - 2-4x faster inference
   - 75% smaller model size

2. **Enable Caching**
   ```python
   use_cache=True
   ```
   - Instant results for repeated files

3. **Use Lightweight Models**
   ```python
   from model.video_model import get_model
   model = get_model(model_type='lightweight')
   ```

4. **Use GPU**
   ```python
   device='cuda'
   ```

---

## 📊 Expected Performance

### Training Time (on GPU)

| Model | Dataset Size | Epochs | Time |
|-------|--------------|--------|------|
| Video | 1000 videos | 20 | ~2-4 hours |
| Video | 5000 videos | 20 | ~8-12 hours |
| Audio | 1000 clips | 30 | ~30-60 minutes |
| Audio | 10000 clips | 30 | ~4-6 hours |

### Inference Time

| Model | Device | Quantization | Time |
|-------|--------|--------------|------|
| Video | CPU | No | 5-10s |
| Video | CPU | Yes | 2-4s |
| Video | GPU | No | 1-2s |
| Video | GPU | Yes | 0.3-0.5s |
| Audio | CPU | No | 0.5-1s |
| Audio | CPU | Yes | 0.2-0.5s |
| Audio | GPU | No | 0.1-0.3s |
| Audio | GPU | Yes | 0.05-0.1s |

### Expected Accuracy

| Model | Dataset | Accuracy |
|-------|---------|----------|
| Video | FaceForensics++ | 85-95% |
| Video | Celeb-DF | 80-90% |
| Audio | Custom dataset | 80-92% |
| Ensemble | Combined | 2-5% improvement |

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Problem:** Training fails with "CUDA out of memory"

**Solutions:**
1. Reduce batch size
   ```bash
   --batch_size 2  # or lower
   ```

2. Reduce max frames
   ```bash
   --max_frames 5  # instead of 10
   ```

3. Use CPU (slower but works)
   ```bash
   --device cpu
   ```

### Slow Inference

**Problem:** Detection takes too long

**Solutions:**
1. Enable GPU
   ```python
   device='cuda'
   ```

2. Enable quantization
   ```python
   use_quantization=True
   ```

3. Use caching
   ```python
   use_cache=True
   ```

4. Use lightweight model
   ```python
   from model.video_model import get_model
   model = get_model(model_type='lightweight')
   ```

### No Faces Detected

**Problem:** Video processing fails with "No faces detected"

**Solutions:**
1. Ensure videos have clear, visible faces
2. Try different face detector
   ```python
   preprocessor = VideoPreprocessor(face_detector='opencv')
   ```

3. Disable face detection (use full frames)
   ```python
   preprocessor = VideoPreprocessor(detect_faces=False)
   ```

---

## 📚 Next Steps

1. **Train Models** - Follow the training steps above
2. **Evaluate** - Test on validation set
3. **Optimize** - Apply quantization for deployment
4. **Deploy** - Use Streamlit or create API
5. **Monitor** - Track performance in production

For detailed documentation, see:
- [AI/ML Integration Guide](AI_ML_INTEGRATION_GUIDE.md) - Complete technical details
- [README.md](README.md) - General documentation

---

## 💡 Pro Tips

1. **Start Small** - Train with 100-200 videos first to test the pipeline
2. **Use GPU** - 10x faster training than CPU
3. **Save Checkpoints** - Training scripts automatically save best model
4. **Monitor Training** - Check validation loss to prevent overfitting
5. **Use Cache** - Enable caching for faster repeated analysis
6. **Quantize for Production** - Always quantize before deployment
7. **Combine Models** - Use ensemble for better accuracy

---

## 🎉 Summary

You now have a production-ready deepfake detection system with:

✅ State-of-the-art CNN-RNN and GRU models
✅ 2-4x faster inference with quantization
✅ Instant caching for repeated analysis
✅ GPU acceleration support
✅ Comprehensive training infrastructure
✅ Ensemble learning for improved accuracy

**Start detecting deepfakes today!** 🚀
# 🔍 AI-Powered Deepfake Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A cutting-edge **full-stack AI web application** that detects deepfakes in videos and audio files using advanced **CNN-RNN deep learning models**. Built for cybersecurity applications with a focus on accuracy, performance, and user experience.

---

## 🎯 Features

### Core Capabilities
- ✅ **Video Deepfake Detection** - Analyzes facial motion, blinking patterns, and temporal inconsistencies
- 🎵 **Audio Deepfake Detection** - Identifies voice cloning and audio manipulation
- 📊 **Confidence Scoring** - Provides probability scores for predictions
- 📈 **Rich Visualizations** - Interactive charts, spectrograms, and frame analysis
- 📄 **PDF Report Generation** - Downloadable analysis reports
- 🔒 **Security Features** - File validation, sanitization, and activity logging

### Technical Highlights
- **CNN-RNN Architecture** - EfficientNet/ResNet50 + LSTM for video analysis
- **GRU-based Audio Model** - MFCC feature extraction with temporal modeling
- **Attention Mechanism** - Enhanced feature selection
- **GPU Acceleration** - CUDA support for faster inference
- **Streamlit Dashboard** - Modern, responsive web interface

---

## 🏗️ System Architecture

```
Input (Video/Audio)
    ↓
Preprocessing (Frame/MFCC Extraction)
    ↓
CNN Feature Extraction (Spatial)
    ↓
RNN/LSTM Temporal Analysis
    ↓
Attention + Classification
    ↓
Output (Real/Fake + Confidence)
```

---

## 📦 Project Structure

```
deepfake_detection/
│
├── app.py                          # Main Streamlit web application
│
├── model/
│   ├── __init__.py
│   ├── video_model.py              # CNN-RNN video detection model
│   ├── audio_model.py              # GRU audio detection model
│   ├── train_video_model.py        # Video model training script
│   └── train_audio_model.py        # Audio model training script
│
├── utils/
│   ├── __init__.py
│   ├── preprocess_video.py         # Video preprocessing (OpenCV, MTCNN)
│   ├── preprocess_audio.py         # Audio preprocessing (Librosa, MFCC)
│   ├── visualization.py            # Plotting and visualization utilities
│   ├── report_generator.py         # PDF report generation
│   ├── predictor.py                # Inference engine
│   └── security.py                 # File validation and logging
│
├── static/
│   └── sample_outputs/             # Visualization outputs
│
├── templates/                      # HTML templates (if using Flask)
├── reports/                        # Generated PDF reports
├── logs/                           # Activity logs
├── uploads/                        # Uploaded files
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- 10GB+ disk space

### Step 1: Clone the Repository
```bash
cd c:\Users\STUDENT\Downloads\project\deepfake_detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models (Optional)
If you have pre-trained models, place them in the `model/` directory:
- `deepfake_video_model.pt` - Video detection model
- `deepfake_audio_model.pt` - Audio detection model

Or train your own models (see Training section below).

---

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Upload File** - Choose a video (.mp4, .avi) or audio (.wav, .mp3) file
2. **Analyze** - Click "Analyze File" button
3. **View Results** - See prediction, confidence score, and visualizations
4. **Download Report** - Generate and download PDF report

### Command-Line Prediction

```bash
python utils/predictor.py --file path/to/video.mp4 --video_model model/deepfake_video_model.pt --report
```

**Arguments:**
- `--file`: Path to media file (required)
- `--video_model`: Path to video model weights
- `--audio_model`: Path to audio model weights
- `--type`: File type ('video' or 'audio'), auto-detected if omitted
- `--report`: Generate PDF report
- `--device`: Device to use ('cuda' or 'cpu')

---

## 🎓 Training Models

### Prepare Dataset

Organize your dataset in the following structure:
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
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)

### Train Video Model

```bash
python model/train_video_model.py \
    --dataset path/to/dataset \
    --save_path model/deepfake_video_model.pt \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.0001 \
    --device cuda
```

**Parameters:**
- `--dataset`: Path to dataset directory
- `--save_path`: Where to save trained model
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: Training device ('cuda' or 'cpu')

### Train Audio Model

```bash
python model/train_audio_model.py \
    --dataset path/to/audio_dataset \
    --save_path model/deepfake_audio_model.pt \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --device cuda
```

### Training Tips

- **GPU Memory**: Reduce batch size if you encounter OOM errors
- **Data Augmentation**: Implement in dataset class for better generalization
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Learning Rate**: Use scheduler for adaptive learning rate
- **Checkpointing**: Models are saved when validation loss improves

---

## 📊 Model Performance

### Video Detection Model
- **Architecture**: EfficientNet-B0 + BiLSTM + Attention
- **Input**: 10 frames @ 224x224 RGB
- **Parameters**: ~10M trainable
- **Expected Accuracy**: 85-95% (depends on dataset)

### Audio Detection Model
- **Architecture**: GRU + Attention
- **Input**: MFCC features (120 x 300)
- **Parameters**: ~2M trainable
- **Expected Accuracy**: 80-92% (depends on dataset)

---

## 🔒 Security Features

### File Validation
- Extension whitelist (.mp4, .avi, .wav, .mp3)
- MIME type verification
- File size limits (100MB video, 50MB audio)
- Filename sanitization

### Activity Logging
All detection activities are logged:
- Upload events with file hash
- Prediction results with timestamps
- Error events with details

Logs are stored in `logs/activity.log`

### Data Privacy
- Uploaded files are stored temporarily
- Automatic cleanup of old files (24 hours)
- No data is transmitted externally

---

## 🛠️ Configuration

### Adjusting Model Parameters

Edit model hyperparameters in the respective files:

**Video Model** (`model/video_model.py`):
```python
model = CNNRNNDeepfakeDetector(
    cnn_type='efficientnet',      # or 'resnet'
    feature_dim=512,
    lstm_hidden=256,
    lstm_layers=2,
    num_classes=2,
    dropout=0.3
)
```

**Audio Model** (`model/audio_model.py`):
```python
model = AudioDeepfakeDetector(
    input_channels=120,
    gru_hidden=256,
    gru_layers=2,
    num_classes=2,
    dropout=0.3
)
```

### Adjusting Preprocessing

**Video** (`utils/preprocess_video.py`):
- `max_frames`: Maximum frames to extract (default: 100)
- `img_size`: Target image size (default: 224)
- `frame_interval`: Extract every nth frame (default: 1)

**Audio** (`utils/preprocess_audio.py`):
- `sample_rate`: Target sample rate (default: 16000)
- `n_mfcc`: Number of MFCC coefficients (default: 40)
- `max_duration`: Maximum audio duration (default: 30s)

---

## 🧪 Testing

### Unit Tests (Coming Soon)
```bash
pytest tests/
```

### Manual Testing
1. Use sample videos/audios from test dataset
2. Verify predictions against known labels
3. Check PDF report generation
4. Test with various file formats

---

## 📚 API Reference

### DeepfakeDetector Class

```python
from utils.predictor import DeepfakeDetector

detector = DeepfakeDetector(
    video_model_path='model/deepfake_video_model.pt',
    audio_model_path='model/deepfake_audio_model.pt',
    device='cuda'
)

# Detect video
result = detector.detect_video('path/to/video.mp4')

# Detect audio
result = detector.detect_audio('path/to/audio.wav')

# Auto-detect and analyze
result = detector.detect(
    file_path='path/to/file',
    generate_report=True,
    report_path='reports/output.pdf'
)
```

**Result Format:**
```python
{
    'prediction': 'Real' or 'Fake',
    'confidence': 0.95,  # 0-1
    'probabilities': {
        'real': 0.95,
        'fake': 0.05
    },
    'file_name': 'video.mp4',
    'file_type': 'video',
    'metadata': {...},
    'report_path': 'reports/report.pdf'  # if generated
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch_size in training or use CPU
```

**2. No faces detected in video**
```
Solution: Ensure video has clear facial visibility
         Try different face detector (Dlib)
```

**3. Import errors (librosa, magic, etc.)**
```
Solution: pip install -r requirements.txt
         On Windows, install python-magic-bin
```

**4. Slow inference**
```
Solution: Use GPU (CUDA)
         Reduce max_frames or frame_interval
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🌟 Acknowledgments

- **FaceForensics++** - Dataset for training
- **PyTorch** - Deep learning framework
- **Streamlit** - Web framework
- **OpenCV** - Computer vision library
- **Librosa** - Audio processing

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@software{deepfake_detection_2025,
  author = {Your Name},
  title = {AI-Powered Deepfake Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/deepfake-detection}
}
```

---

<p align="center">
  Made with ❤️ using PyTorch, OpenCV & Streamlit
</p>

# 🎓 Video Model Training Guide

Complete step-by-step instructions for training the video deepfake detection model.

---

## 📋 Prerequisites

1. **Python 3.10+** installed
2. **All dependencies** installed (`pip install -r requirements.txt`)
3. **Dataset** prepared (see Dataset Preparation below)
4. **GPU (optional but recommended)** - Training on CPU will be very slow

---

## 📁 Step 1: Prepare Your Dataset

### Dataset Structure

Organize your videos in the following directory structure:

```
your_dataset/
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mp4
│   └── ...
└── fake/
    ├── fake_video1.mp4
    ├── fake_video2.mp4
    ├── fake_video3.mp4
    └── ...
```

### Dataset Requirements

- **Format**: MP4 files (other formats can be converted)
- **Minimum videos**: At least 50-100 videos per class (real/fake) for basic training
- **Recommended**: 500+ videos per class for good performance
- **Video quality**: Clear, well-lit videos with visible faces
- **Duration**: 5-30 seconds per video recommended

### Recommended Datasets

1. **FaceForensics++**
   - Download: https://github.com/ondyari/FaceForensics
   - Contains real and manipulated videos
   - High quality, well-labeled

2. **Celeb-DF**
   - Download: https://github.com/yuezunli/celeb-deepfakeforensics
   - Large-scale dataset with celebrity deepfakes

3. **DFDC (Deepfake Detection Challenge)**
   - Download: https://ai.facebook.com/datasets/dfdc/
   - Facebook's challenge dataset

---

## 🚀 Step 2: Basic Training Command

### Windows PowerShell

```powershell
cd "C:\Users\ranji\Deepfake AI"

python model/train_video_model.py `
    --dataset "path/to/your/dataset" `
    --save_path "model/deepfake_video_model.pt" `
    --epochs 20 `
    --batch_size 4 `
    --lr 0.0001 `
    --device cuda
```

### Linux/Mac

```bash
python model/train_video_model.py \
    --dataset path/to/your/dataset \
    --save_path model/deepfake_video_model.pt \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.0001 \
    --device cuda
```

### If you don't have GPU (CPU only)

```powershell
python model/train_video_model.py `
    --dataset "path/to/your/dataset" `
    --save_path "model/deepfake_video_model.pt" `
    --epochs 20 `
    --batch_size 2 `
    --lr 0.0001 `
    --device cpu
```

---

## ⚙️ Step 3: Understanding Parameters

### Required Parameters

- `--dataset`: **REQUIRED** - Path to your dataset directory
  - Example: `--dataset "C:\datasets\deepfake_dataset"`
  - Must contain `real/` and `fake/` subdirectories

### Optional Parameters

- `--save_path`: Where to save the trained model (default: `model/deepfake_video_model.pt`)
  - Example: `--save_path "model/my_custom_model.pt"`

- `--epochs`: Number of training epochs (default: 20)
  - More epochs = longer training but potentially better accuracy
  - Recommended: 20-50 for small datasets, 10-20 for large datasets
  - Example: `--epochs 30`

- `--batch_size`: Batch size (default: 4)
  - Larger = faster training but needs more GPU memory
  - GPU: Try 4-8, CPU: Use 1-2
  - Example: `--batch_size 8`

- `--lr`: Learning rate (default: 0.0001)
  - Controls how fast the model learns
  - Lower = slower but more stable
  - Example: `--lr 0.00005`

- `--device`: Training device (default: `cuda`)
  - `cuda` for GPU, `cpu` for CPU only
  - Example: `--device cpu`

- `--max_frames`: Maximum frames per video (default: 10)
  - More frames = better temporal understanding but slower
  - Example: `--max_frames 15`

---

## 📊 Step 4: Training Examples

### Example 1: Quick Test Training (Small Dataset)

```powershell
python model/train_video_model.py `
    --dataset "C:\datasets\small_test" `
    --save_path "model/test_model.pt" `
    --epochs 5 `
    --batch_size 2 `
    --device cpu
```

### Example 2: Full Training (Large Dataset with GPU)

```powershell
python model/train_video_model.py `
    --dataset "C:\datasets\faceforensics" `
    --save_path "model/deepfake_video_model.pt" `
    --epochs 30 `
    --batch_size 8 `
    --lr 0.0001 `
    --max_frames 15 `
    --device cuda
```

### Example 3: Fine-tuning (Resume Training)

If you want to continue training from a checkpoint:

```python
# Modify train_video_model.py to load checkpoint
# Or use the saved checkpoint from checkpoints/ directory
```

---

## 📈 Step 5: Monitor Training

During training, you'll see output like:

```
Epoch 1/20
Train Loss: 0.6234, Train Acc: 0.6500
Val Loss: 0.5123, Val Acc: 0.7200
Saving checkpoint...
```

### What to Watch For

- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease (if it increases, model may be overfitting)
- **Accuracy**: Should increase over time
- **Checkpoints**: Saved automatically when validation improves

### Training Output Location

- **Model checkpoints**: Saved in `checkpoints/` directory
- **Training history**: Saved as `checkpoints/training_history.json`
- **Final model**: Saved to path specified in `--save_path`

---

## 🎯 Step 6: Training Tips

### For Better Results

1. **Balanced Dataset**: Ensure equal number of real and fake videos
2. **Data Augmentation**: The training script includes basic augmentation
3. **Learning Rate**: Start with default (0.0001), reduce if loss doesn't decrease
4. **Early Stopping**: Stop if validation loss stops improving for 5+ epochs
5. **GPU Memory**: If you get OOM errors, reduce `--batch_size`

### Troubleshooting

**Problem**: "CUDA out of memory"
- **Solution**: Reduce `--batch_size` to 2 or 1

**Problem**: "No videos found"
- **Solution**: Check dataset path and ensure videos are in `real/` and `fake/` folders

**Problem**: Training is very slow
- **Solution**: Use GPU (`--device cuda`) or reduce `--max_frames`

**Problem**: Model not improving
- **Solution**: 
  - Check dataset quality
  - Try different learning rate (`--lr 0.00005`)
  - Increase training epochs
  - Ensure balanced dataset

---

## ✅ Step 7: Verify Training

After training completes:

1. **Check model file exists**:
   ```powershell
   Test-Path "model/deepfake_video_model.pt"
   ```

2. **Test the model**:
   ```powershell
   python -c "import torch; model = torch.load('model/deepfake_video_model.pt'); print('Model loaded successfully!')"
   ```

3. **Use in application**:
   - The trained model will automatically be used by `app.py`
   - Or use with predictor: `python utils/predictor.py --file video.mp4 --video_model model/deepfake_video_model.pt`

---

## 📝 Complete Training Workflow

```powershell
# 1. Navigate to project directory
cd "C:\Users\ranji\Deepfake AI"

# 2. Prepare your dataset (create real/ and fake/ folders)

# 3. Start training
python model/train_video_model.py `
    --dataset "path/to/your/dataset" `
    --save_path "model/deepfake_video_model.pt" `
    --epochs 20 `
    --batch_size 4 `
    --lr 0.0001 `
    --device cuda

# 4. Wait for training to complete (may take hours depending on dataset size)

# 5. Test your model
python utils/predictor.py `
    --file "test_video.mp4" `
    --video_model "model/deepfake_video_model.pt"
```

---

## 🎓 Expected Training Time

- **Small dataset (100 videos)**: 1-2 hours on GPU, 4-8 hours on CPU
- **Medium dataset (500 videos)**: 4-6 hours on GPU, 12-24 hours on CPU
- **Large dataset (5000+ videos)**: 12-24 hours on GPU, 2-5 days on CPU

---

## 📚 Additional Resources

- Check `model/train_video_model.py` for advanced training options
- See `README.md` for general project information
- Check `USAGE_EXAMPLES.md` for usage examples after training

---

**Good luck with your training! 🚀**

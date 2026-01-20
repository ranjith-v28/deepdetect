"""
Flask Web Application for Deepfake Detection
Replaces Streamlit with a traditional Flask web server.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
from pathlib import Path
import sys
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Lazy imports to speed up startup
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize components
detector = None
file_validator = None
activity_logger = None

# Ensure directories exist
Path('uploads').mkdir(exist_ok=True)
Path('static/results').mkdir(parents=True, exist_ok=True)
Path('reports').mkdir(exist_ok=True)


def initialize_components():
    """Initialize all components lazily."""
    global detector, file_validator, activity_logger
    
    if file_validator is None:
        from utils.security import FileValidator, ActivityLogger
        file_validator = FileValidator()
        activity_logger = ActivityLogger()
    
    if detector is None:
        from utils.predictor import DeepfakeDetector
        
        model_dir = Path('model')
        video_model_path = model_dir / 'deepfake_video_model.pt'
        audio_model_path = model_dir / 'deepfake_audio_model.pt'
        
        # Check if models exist
        video_model = str(video_model_path) if video_model_path.exists() else None
        audio_model = str(audio_model_path) if audio_model_path.exists() else None
        
        if video_model is not None or audio_model is not None:
            try:
                detector = DeepfakeDetector(
                    video_model_path=video_model,
                    audio_model_path=audio_model,
                    device='cuda'
                )
            except Exception as e:
                print(f"Error initializing detector: {str(e)}")
    
    return detector


@app.route('/')
def index():
    """Home page."""
    model_dir = Path('model')
    video_model_exists = (model_dir / 'deepfake_video_model.pt').exists()
    audio_model_exists = (model_dir / 'deepfake_audio_model.pt').exists()
    
    device = 'CUDA' if torch.cuda.is_available() else 'CPU'
    
    return render_template('index.html',
                         video_model_exists=video_model_exists,
                         audio_model_exists=audio_model_exists,
                         device=device,
                         pytorch_version=torch.__version__)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    # Initialize components
    initialize_components()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Sanitize filename
    safe_filename = file_validator.sanitize_filename(file.filename)
    file_path = Path(app.config['UPLOAD_FOLDER']) / safe_filename
    
    # Save file
    file.save(str(file_path))
    
    # Validate file
    is_valid, message = file_validator.validate_file(str(file_path))
    
    if not is_valid:
        file_path.unlink()
        return jsonify({'error': message}), 400
    
    if detector is None:
        return jsonify({'error': 'No trained models available. Please train models first.'}), 500
    
    try:
        # Analyze file
        result = detector.detect(
            file_path=str(file_path),
            generate_report=True,
            report_path=f"reports/report_{safe_filename.rsplit('.', 1)[0]}.pdf",
            visualization_dir="static/results"
        )
        
        # Log activity
        file_hash = file_validator.get_file_hash(str(file_path))
        activity_logger.log_prediction(
            filename=safe_filename,
            file_hash=file_hash,
            prediction=result.get('prediction', 'Unknown'),
            confidence=result.get('confidence', 0.0),
            processing_time=0.0
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'prediction': result.get('prediction', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'probabilities': result.get('probabilities', {}),
            'file_type': result.get('file_type', 'unknown'),
            'file_name': safe_filename,
            'report_path': result.get('report_path', '')
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_report/<filename>')
def download_report(filename):
    """Download PDF report."""
    report_path = Path('reports') / filename
    
    if not report_path.exists():
        return jsonify({'error': 'Report not found'}), 404
    
    return send_file(str(report_path), as_attachment=True)


@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old files."""
    try:
        from utils.security import cleanup_old_files
        cleanup_old_files('uploads', max_age_hours=24)
        cleanup_old_files('static/results', max_age_hours=24)
        return jsonify({'success': True, 'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/help')
def help_page():
    """Help and documentation page."""
    return render_template('help.html')


if __name__ == '__main__':
    print("=" * 60)
    print("🔍 AI Deepfake Detection System - Flask Server")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

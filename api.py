import os
import re
import tempfile
from flask import Flask, request, jsonify
import torch
from werkzeug.utils import secure_filename
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables to store loaded models
models = None
face_extractor = None
video_reader = None

def initialize_models(weights_dir, model_names):
    """Initialize models once when the app starts"""
    global models, face_extractor, video_reader

    models = []
    model_paths = [os.path.join(weights_dir, model) for model in model_names]

    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print(f"Loading state dict {path}")
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        models.append(model.half())

    # Initialize video reader and face extractor
    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)

    print(f"Successfully loaded {len(models)} models")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models is not None and len(models) > 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict deepfake on uploaded video(s)
    Accepts: multipart/form-data with video file(s)
    Returns: JSON with predictions
    """
    if models is None:
        return jsonify({'error': 'Models not initialized'}), 500

    # Check if files are present in request
    if 'video' not in request.files and 'videos' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    # Handle single or multiple files
    files = request.files.getlist('video') if 'video' in request.files else request.files.getlist('videos')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    results = []
    temp_files = []

    try:
        # Save uploaded videos to temporary directory
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                temp_files.append((filename, temp_path))

        if not temp_files:
            return jsonify({'error': 'No valid video files provided'}), 400

        # Extract just filenames and paths
        video_filenames = [name for name, _ in temp_files]
        temp_dir = app.config['UPLOAD_FOLDER']

        # Run prediction
        frames_per_video = 32
        input_size = 380
        strategy = confident_strategy

        predictions = predict_on_video_set(
            face_extractor=face_extractor,
            input_size=input_size,
            models=models,
            strategy=strategy,
            frames_per_video=frames_per_video,
            videos=video_filenames,
            num_workers=6,
            test_dir=temp_dir
        )

        # Format results
        for filename, prediction in zip(video_filenames, predictions):
            results.append({
                'filename': filename,
                'prediction': float(prediction),
                'label': 'FAKE' if prediction > 0.5 else 'REAL'
            })

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    finally:
        # Clean up temporary files
        for _, temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file {temp_path}: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("DeepFake Detection API")
    parser.add_argument('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
    parser.add_argument('--models', nargs='+', required=True, help="checkpoint files")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="host to run the server on")
    parser.add_argument('--port', type=int, default=5000, help="port to run the server on")
    args = parser.parse_args()

    # Initialize models before starting the server
    print("Initializing models...")
    initialize_models(args.weights_dir, args.models)
    print("Models initialized successfully!")

    # Start Flask server
    app.run(host='0.0.0.0', port=8000, debug=False)
import os
import re
import tempfile
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from facenet_pytorch.models.mtcnn import MTCNN
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set, predict_on_image, detect_faces_in_image, DEVICE
from training.zoo.classifiers import DeepFakeClassifier
from flask_cors import CORS
from db_utils import init_db, save_prediction, get_history
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables to store loaded models
models = None
face_extractor = None
video_reader = None
face_detector = None  # MTCNN detector for standalone image face detection

def initialize_models(weights_dir, model_names):
    """Initialize models once when the app starts"""
    global models, face_extractor, video_reader

    models = []
    model_paths = [os.path.join(weights_dir, model) for model in model_names]

    # Use the detected DEVICE (cuda, mps, or cpu)
    print(f"Loading models on device: {DEVICE}")

    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(DEVICE)
        print(f"Loading state dict {path}")
        checkpoint = torch.load(path, weights_only=False, map_location=DEVICE)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        if DEVICE == "cuda":
            models.append(model.half())
        else:
            models.append(model.float())

    # Initialize video reader and face extractor
    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)

    # Initialize standalone face detector for image processing
    global face_detector
    # MTCNN handling is done inside FaceExtractor/kernel_utils with logic for device selection
    # But for this standalone instance, we should match logic or check device availability
    mtcnn_device = "cuda" if torch.cuda.is_available() else "cpu"
    face_detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=mtcnn_device)

    print(f"Successfully loaded {len(models)} models")

def allowed_file(filename):
    """Check if file extension is allowed for videos"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

def allowed_image_file(filename):
    """Check if file extension is allowed for images"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp'}

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
            label = 'FAKE' if prediction > 0.5 else 'REAL'
            
            # Save to history
            save_prediction(filename, float(prediction), label)
            
            results.append({
                'filename': filename,
                'prediction': float(prediction),
                'label': label
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

@app.route('/detect-faces', methods=['POST'])
def detect_faces():
    """
    Detect faces in uploaded image(s)
    Accepts: multipart/form-data with image file(s)
    Returns: JSON with detected faces, bounding boxes, and confidence scores
    """
    if face_detector is None:
        return jsonify({'error': 'Face detector not initialized'}), 500

    # Check if files are present in request
    if 'image' not in request.files and 'images' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Handle single or multiple files
    files = request.files.getlist('image') if 'image' in request.files else request.files.getlist('images')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    results = []

    try:
        for file in files:
            if file and allowed_image_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Read image from file
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Detect faces using kernel_utils
                detected_faces = detect_faces_in_image(img_rgb, face_detector)
                
                faces_data = []
                for face in detected_faces:
                    # Convert face crop to base64 for response
                    face_pil = Image.fromarray(face['face_crop'])
                    buffered = BytesIO()
                    face_pil.save(buffered, format="JPEG")
                    face_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    faces_data.append({
                        'face_index': face['face_index'],
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'face_image_base64': face_base64
                    })
                
                results.append({
                    'filename': filename,
                    'faces_detected': len(faces_data),
                    'faces': faces_data
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


@app.route('/predict-images', methods=['POST'])
def predict_images():
    """
    Predict deepfake on uploaded image(s) containing faces
    Accepts: multipart/form-data with image file(s)
    Returns: JSON with predictions (FAKE/REAL) for each detected face
    """
    if models is None:
        return jsonify({'error': 'Models not initialized'}), 500
    
    if face_detector is None:
        return jsonify({'error': 'Face detector not initialized'}), 500

    # Check if files are present in request
    if 'image' not in request.files and 'images' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Handle single or multiple files
    files = request.files.getlist('image') if 'image' in request.files else request.files.getlist('images')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    results = []
    input_size = 380

    try:
        for file in files:
            if file and allowed_image_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Read image from file
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Detect faces using kernel_utils
                detected_faces = detect_faces_in_image(img_rgb, face_detector)
                
                faces_predictions = []
                for face in detected_faces:
                    if face['confidence'] > 0.5:  # Only process confident face detections
                        # Use predict_on_image from kernel_utils WITH heatmap
                        avg_pred, heatmap_b64 = predict_on_image(face['face_crop'], input_size, models, confident_strategy, with_heatmap=True)
                        
                        faces_predictions.append({
                            'face_index': face['face_index'],
                            'bbox': face['bbox'],
                            'face_confidence': face['confidence'],
                            'prediction': float(avg_pred),
                            'label': 'FAKE' if avg_pred > 0.5 else 'REAL',
                            'heatmap': heatmap_b64
                        })
                
                # Determine overall image prediction (if any face is fake, image is fake)
                if faces_predictions:
                    max_pred = max(fp['prediction'] for fp in faces_predictions)
                    overall_label = 'FAKE' if max_pred > 0.5 else 'REAL'
                    
                    # Save to history
                    save_prediction(filename, float(max_pred), overall_label)
                else:
                    max_pred = None
                    overall_label = 'NO_FACE_DETECTED'
                
                results.append({
                    'filename': filename,
                    'faces_detected': len(faces_predictions),
                    'overall_prediction': max_pred,
                    'overall_label': overall_label,
                    'faces': faces_predictions
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


@app.route('/history', methods=['GET'])
def history():
    """
    Get scan history
    """
    try:
        limit = request.args.get('limit', default=50, type=int)
        rows = get_history(limit)
        return jsonify({
            'success': True,
            'history': rows
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
    init_db() # Initialize database
    initialize_models(args.weights_dir, args.models)
    print("Models initialized successfully!")

    # Start Flask server
    app.run(host='0.0.0.0', port=8000, debug=False)
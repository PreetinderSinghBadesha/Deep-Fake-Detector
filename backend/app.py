import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from mtcnn import MTCNN
import logging
import tempfile
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Path to the model directory from the backend directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')

# Define model paths in order of preference
MODEL_PATHS = [
    os.path.join(MODEL_DIR, 'deepfake_detector_best_model.h5'),
    os.path.join(MODEL_DIR, 'deepfake_detector_model_fine_tuned.h5'),
    os.path.join(MODEL_DIR, 'deepfake_detector_model.h5')
]

# Path to optimal threshold
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'optimal_threshold.txt')

# Initialize variables
model = None
detector = None
optimal_threshold = 0.5  # Default threshold
model_input_shape = (224, 224)  # Updated default to match our new model

def load_model():
    """Load the trained model, optimal threshold, and MTCNN face detector"""
    global model, detector, optimal_threshold, model_input_shape
    
    logger.info("Loading DeepFake detection model...")
    
    # Try models in order of preference
    model_loaded = False
    loaded_model_path = None
    
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                # Compile the model with basic settings
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model_loaded = True
                loaded_model_path = model_path
                logger.info(f"Model loaded successfully from: {os.path.basename(model_path)}")
                break
            except Exception as e:
                logger.error(f"Error loading model {os.path.basename(model_path)}: {str(e)}")
                continue
    
    if not model_loaded:
        logger.error("No models available! Please train a model first.")
        raise FileNotFoundError("No valid model found in the model directory")
    
    # Get the input shape that the model expects
    input_shape = model.input_shape
    if input_shape is not None:
        height, width = input_shape[1:3]
        model_input_shape = (height, width)
        logger.info(f"Detected model input shape: {height}x{width}")
    else:
        logger.warning(f"Could not determine model input shape, using default: {model_input_shape}")
    
    # Load optimal threshold if available
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, 'r') as f:
                optimal_threshold = float(f.read().strip())
                logger.info(f"Optimal threshold loaded: {optimal_threshold}")
        except Exception as e:
            logger.warning(f"Error loading threshold, using default: {str(e)}")
    else:
        logger.warning(f"No threshold file found, using default: {optimal_threshold}")
    
    logger.info("Initializing face detector...")
    detector = MTCNN()
    logger.info("Face detector initialized!")

def assess_face_quality(face_img):
    """
    Assess the quality of the detected face
    Returns quality score and boolean indicating if the face passes quality check
    """
    if face_img is None:
        return 0, False
        
    # Convert to uint8 if floating point
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8)
    
    # Check image size
    height, width = face_img.shape[:2]
    if height < 64 or width < 64:
        return 0.2, False
        
    # Calculate face sharpness using Laplacian
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate face brightness
    brightness = np.mean(gray)
    
    # Normalize scores
    sharpness_score = min(1.0, laplacian_var / 500)
    brightness_score = 1.0 - abs(brightness - 128) / 128
    
    # Combine scores
    quality_score = 0.7 * sharpness_score + 0.3 * brightness_score
    
    # Pass if quality is good enough
    passes_quality = quality_score > 0.4
    
    return quality_score, passes_quality

def assess_video_face_quality(face_img):
    """
    Assess the quality of a face from a video frame
    Uses more relaxed thresholds compared to image assessment
    
    Returns quality score and boolean indicating if the face passes quality check
    """
    if face_img is None:
        return 0, False
        
    # Convert to uint8 if floating point
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8)
    
    # Check image size - lower minimum for videos
    height, width = face_img.shape[:2]
    if height < 48 or width < 48:  # Smaller minimum size for videos
        return 0.1, False
        
    # Calculate face sharpness using Laplacian
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate face brightness
    brightness = np.mean(gray)
    
    # Normalize scores - more lenient for videos
    sharpness_score = min(1.0, laplacian_var / 300)  # Lower threshold for videos
    brightness_score = 1.0 - abs(brightness - 128) / 150  # More tolerance for brightness variations
    
    # Combine scores with different weights
    quality_score = 0.6 * sharpness_score + 0.4 * brightness_score
    
    # More relaxed quality threshold for videos
    passes_quality = quality_score > 0.25  # Lower threshold for videos
    
    return quality_score, passes_quality

def preprocess_image(image):
    """
    Preprocess image for model prediction with improved techniques:
    1. Detect face
    2. Extract and resize face
    3. Apply color correction
    4. Enhance image quality
    5. Normalize pixel values
    """
    # Use the model's expected input shape
    target_size = model_input_shape
    
    # Detect faces
    try:
        faces = detector.detect_faces(image)
        if not faces:
            return None, "No face detected in the image"
    except Exception as e:
        logger.error(f"Face detection error: {str(e)}")
        return None, f"Face detection error: {str(e)}"
    
    # Get the largest face
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face['box']
    
    # Add margin (30% as in training)
    margin_x = int(w * 0.3)
    margin_y = int(h * 0.3)
    
    # Apply margins safely
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(image.shape[1] - x, w + 2 * margin_x)
    h = min(image.shape[0] - y, h + 2 * margin_y)
    
    # Extract the face
    face_img = image[y:y+h, x:x+w]
    
    # Check face quality
    quality_score, passes_quality = assess_face_quality(face_img)
    if not passes_quality:
        return None, f"Face quality too low (score: {quality_score:.2f}). Please use a clearer image."
    
    # Resize to target size
    face_img = cv2.resize(face_img, target_size)
    
    # Apply advanced preprocessing
    # 1. Color correction - normalize color channels
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    updated_lab = cv2.merge((cl, a, b))
    face_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
    
    # 2. Normalize pixel values
    face_img = face_img.astype('float32') / 255.0
    
    return np.expand_dims(face_img, axis=0), None  # Add batch dimension

def preprocess_video_frame(image):
    """
    Preprocess video frame for model prediction with more relaxed thresholds:
    1. Detect face with more lenient parameters
    2. Extract and resize face
    3. Apply color correction
    4. Normalize pixel values
    
    Uses more lenient face detection and quality assessment for videos
    """
    # Use the model's expected input shape
    target_size = model_input_shape
    
    # Detect faces
    try:
        faces = detector.detect_faces(image)
        if not faces:
            return None, "No face detected in the video frame"
    except Exception as e:
        logger.error(f"Video frame face detection error: {str(e)}")
        return None, f"Face detection error: {str(e)}"
    
    # Get the largest face
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face['box']
    
    # Add margin (30% as in training)
    margin_x = int(w * 0.3)
    margin_y = int(h * 0.3)
    
    # Apply margins safely
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(image.shape[1] - x, w + 2 * margin_x)
    h = min(image.shape[0] - y, h + 2 * margin_y)
    
    # Extract the face
    face_img = image[y:y+h, x:x+w]
    
    # Check face quality - using video-specific quality function
    quality_score, passes_quality = assess_video_face_quality(face_img)
    if not passes_quality:
        return None, f"Face quality too low for video frame (score: {quality_score:.2f})"
    
    # Resize to target size
    face_img = cv2.resize(face_img, target_size)
    
    # Apply preprocessing
    # 1. Color correction - normalize color channels
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    updated_lab = cv2.merge((cl, a, b))
    face_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
    
    # 2. Normalize pixel values
    face_img = face_img.astype('float32') / 255.0
    
    return np.expand_dims(face_img, axis=0), None  # Add batch dimension

def process_video_frames(video_path, max_frames=20):
    """
    Process video frames to extract faces and make predictions
    with improved detection for challenging videos
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process initially
        
    Returns:
        results: Dict containing predictions and statistics
    """
    if not os.path.exists(video_path):
        return None, "Video file not found"
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Could not open video file"
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {frame_count} frames, {fps} fps, {width}x{height}")
    
    if frame_count <= 0:
        return None, "Invalid video file with 0 frames"
    
    # Calculate frames to extract (evenly distributed)
    frames_to_extract = min(max_frames, frame_count)
    if frames_to_extract < 1:
        return None, "Not enough frames in video"
    
    # Calculate frame intervals for even distribution
    interval = frame_count / frames_to_extract
    frame_indices = [int(i * interval) for i in range(frames_to_extract)]
    
    logger.info(f"Processing video with {frame_count} frames, extracting {frames_to_extract} frames")
    
    predictions = []
    processed_frames = 0
    detected_faces = 0
    failed_frames = []
    
    # First pass: process selected frames
    for idx in frame_indices:
        # Set video to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {idx}")
            continue
        
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        processed_frame, error = preprocess_video_frame(frame_rgb)
        processed_frames += 1
        
        if processed_frame is not None:
            # Make prediction on the extracted face
            detected_faces += 1
            try:
                pred = model.predict(processed_frame, verbose=0)[0][0]
                predictions.append(float(pred))
                logger.info(f"Frame {idx}: Successfully detected face and made prediction: {pred:.4f}")
            except Exception as e:
                logger.error(f"Error during prediction on frame {idx}: {str(e)}")
        else:
            logger.info(f"Frame {idx}: {error}")
            failed_frames.append(idx)
    
    # If no faces detected in first pass, try a second pass with more frames
    if detected_faces == 0 and frame_count > frames_to_extract:
        logger.info("No faces detected in initial frames, trying additional frames...")
        
        # Try more frames, focusing on the middle part of the video where faces are often visible
        additional_frames = min(frame_count - frames_to_extract, 40)  # Try up to 40 more frames
        
        # Create indices focused on the middle section of the video
        middle = frame_count // 2
        quarter = frame_count // 4
        additional_indices = []
        
        # Add frames from the middle section of the video
        for i in range(additional_frames):
            # Distribute frames around the middle section with some randomization
            offset = (i % 2 * 2 - 1) * (i // 2) * quarter // (additional_frames // 4)
            frame_idx = middle + offset
            frame_idx = max(0, min(frame_count - 1, frame_idx))
            if frame_idx not in frame_indices and frame_idx not in additional_indices:
                additional_indices.append(frame_idx)
        
        # Process these additional frames
        for idx in additional_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, error = preprocess_video_frame(frame_rgb)
            processed_frames += 1
            
            if processed_frame is not None:
                detected_faces += 1
                try:
                    pred = model.predict(processed_frame, verbose=0)[0][0]
                    predictions.append(float(pred))
                    logger.info(f"Additional frame {idx}: Successfully detected face: {pred:.4f}")
                except Exception as e:
                    logger.error(f"Error during prediction on additional frame {idx}: {str(e)}")
    
    # If we still have no faces, try a fallback to the normal image preprocessor
    # which might detect some faces the video processor misses
    if detected_faces == 0 and len(failed_frames) > 0:
        logger.info("Trying fallback image preprocessor for failed frames...")
        for idx in failed_frames[:10]:  # Try up to 10 failed frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, _ = preprocess_image(frame_rgb)  # Use the standard image preprocessor as fallback
            
            if processed_frame is not None:
                detected_faces += 1
                try:
                    pred = model.predict(processed_frame, verbose=0)[0][0]
                    predictions.append(float(pred))
                    logger.info(f"Fallback on frame {idx}: Successfully detected face: {pred:.4f}")
                except Exception as e:
                    logger.error(f"Error during fallback prediction on frame {idx}: {str(e)}")
    
    # Release the video capture
    cap.release()
    
    # Calculate statistics
    if not predictions:
        # Last resort - if we really can't detect any faces, provide a helpful error
        logger.warning(f"No valid faces detected in video of {frame_count} frames, {processed_frames} processed")
        return None, f"No valid faces detected in the video. Please ensure the video contains clear, visible faces. Processed {processed_frames} frames."
    
    avg_prediction = np.mean(predictions)
    is_fake = bool(avg_prediction > optimal_threshold)
    confidence = float(avg_prediction) if is_fake else float(1 - avg_prediction)
    
    results = {
        "is_fake": is_fake,
        "confidence": round(confidence * 100, 2),
        "prediction_value": float(avg_prediction),
        "threshold_used": float(optimal_threshold),
        "frames_analyzed": processed_frames,
        "faces_detected": detected_faces,
        "predictions": [round(p, 4) for p in predictions]
    }
    
    logger.info(f"Video analysis complete: {detected_faces} faces found in {processed_frames} frames, prediction: {avg_prediction:.4f}")
    return results, None

@app.route('/')
def home():
    return "DeepFake Detector API is running!"

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    # Ensure model is loaded
    if model is None or detector is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Read and process the image
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        return jsonify({"error": f"Error reading image: {str(e)}"}), 400
    
    # Preprocess the image
    processed_image, error = preprocess_image(image)
    if processed_image is None:
        return jsonify({"error": error}), 400
    
    # Make prediction
    try:
        # Use smaller batch size if needed for memory constraints
        prediction = model.predict(processed_image, verbose=0)[0][0]
        is_fake = bool(prediction > optimal_threshold) 
        confidence = float(prediction) if is_fake else float(1 - prediction)
        
        # Log prediction details
        logger.info(f"Prediction: {prediction:.4f}, Threshold: {optimal_threshold}, Is Fake: {is_fake}")
        
        return jsonify({
            "is_fake": is_fake,
            "confidence": round(confidence * 100, 2),
            "prediction_value": float(prediction),
            "threshold_used": float(optimal_threshold)
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/api/info', methods=['GET'])
def model_info():
    """Return information about the model"""
    if model is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Get loaded model name
    model_name = "Unknown Model"
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path) and model is not None:
            model_name = os.path.basename(model_path)
            break
    
    # Get model parameters count
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return jsonify({
        "model_name": model_name,
        "input_shape": f"{model_input_shape[0]}x{model_input_shape[1]} RGB images",
        "output": "Binary classification (Real/Fake)",
        "threshold": float(optimal_threshold),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "status": "Loaded and ready"
    })

@app.route('/api/model/history', methods=['GET'])
def get_model_history():
    """Return the training history graph"""
    history_path = os.path.join(MODEL_DIR, 'training_history.png')
    
    if os.path.exists(history_path):
        return send_from_directory(MODEL_DIR, 'training_history.png')
    else:
        return jsonify({"error": "Training history not available"}), 404

@app.route('/api/video/detect', methods=['POST'])
def detect_video_deepfake():
    """Process a video file for deepfake detection"""
    # Ensure model is loaded
    if model is None or detector is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Check if video was uploaded
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400
    
    video_file = request.files['video']
    
    # Save the video to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"deepfake_video_{uuid.uuid4().hex}.mp4")
    
    try:
        video_file.save(temp_video_path)
    except Exception as e:
        logger.error(f"Error saving video file: {str(e)}")
        return jsonify({"error": f"Error saving video file: {str(e)}"}), 400
    
    try:
        # Process the video
        results, error = process_video_frames(temp_video_path, max_frames=30)
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary video file: {str(e)}")
        
        if results is None:
            return jsonify({"error": error or "Failed to process video"}), 400
        
        return jsonify(results)
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
                
        logger.error(f"Video processing error: {str(e)}")
        return jsonify({"error": f"Video processing error: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model at startup
    try:
        load_model()
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        # Continue running the app even if model loading fails
        # It will try to load again when an endpoint is accessed
    
    app.run(debug=True, host='0.0.0.0', port=5000)

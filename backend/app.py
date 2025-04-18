import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Path to the model directory from the backend directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')

# Define model paths in order of preference
MODEL_PATHS = [
    os.path.join(MODEL_DIR, 'deepfake_detector_best_model.h5'),
    os.path.join(MODEL_DIR, 'deepfake_detector_model_best.h5'),
    os.path.join(MODEL_DIR, 'deepfake_detector_model_fine_tuned.h5'),
    os.path.join(MODEL_DIR, 'deepfake_detector_model.h5')
]

# Path to optimal threshold
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'optimal_threshold.txt')

# Initialize variables
model = None
detector = None
optimal_threshold = 0.5  # Default threshold
model_input_shape = (128, 128)  # Default input shape, will be updated when model loads

def load_model():
    """Load the trained model, optimal threshold, and MTCNN face detector"""
    global model, detector, optimal_threshold, model_input_shape
    
    print("Loading DeepFake detection model...")
    
    # Try models in order of preference
    model_loaded = False
    loaded_model_path = None
    
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                model_loaded = True
                loaded_model_path = model_path
                print(f"Model loaded successfully from: {os.path.basename(model_path)}")
                break
            except Exception as e:
                print(f"Error loading model {os.path.basename(model_path)}: {str(e)}")
                continue
    
    if not model_loaded:
        print("No models available! Please train a model first.")
        raise FileNotFoundError("No valid model found in the model directory")
    
    # Get the input shape that the model expects
    input_shape = model.input_shape
    if input_shape is not None:
        height, width = input_shape[1:3]
        model_input_shape = (height, width)
        print(f"Detected model input shape: {height}x{width}")
    else:
        print(f"Could not determine model input shape, using default: {model_input_shape}")
    
    # Load optimal threshold if available
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, 'r') as f:
                optimal_threshold = float(f.read().strip())
                print(f"Optimal threshold loaded: {optimal_threshold}")
        except Exception as e:
            print(f"Error loading threshold, using default: {str(e)}")
    else:
        print(f"No threshold file found, using default: {optimal_threshold}")
    
    print("Initializing face detector...")
    detector = MTCNN()
    print("Face detector initialized!")

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
    faces = detector.detect_faces(image)
    if not faces:
        return None, "No face detected in the image"
    
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
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Read and process the image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image
    processed_image, error = preprocess_image(image)
    if processed_image is None:
        return jsonify({"error": error}), 400
    
    # Make prediction
    try:
        prediction = model.predict(processed_image, verbose=0)[0][0]
        is_fake = bool(prediction > optimal_threshold) 
        confidence = float(prediction) if is_fake else float(1 - prediction)
        
        return jsonify({
            "is_fake": is_fake,
            "confidence": round(confidence * 100, 2),
            "prediction_value": float(prediction),
            "threshold_used": float(optimal_threshold)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/api/info', methods=['GET'])
def model_info():
    """Return information about the model"""
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Get loaded model name
    model_name = "Unknown Model"
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path) and model is not None:
            model_name = os.path.basename(model_path)
            break
    
    return jsonify({
        "model_name": model_name,
        "input_shape": f"{model_input_shape[0]}x{model_input_shape[1]} RGB images",
        "output": "Binary classification (Real/Fake)",
        "threshold": float(optimal_threshold),
        "status": "Loaded and ready"
    })

if __name__ == '__main__':
    # Load model at startup
    load_model()
    app.run(debug=True)

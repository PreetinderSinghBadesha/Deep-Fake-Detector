import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from mtcnn import MTCNN
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

def preprocess_image(image_path, detector, target_size):
    """
    Preprocess an image for the DeepFake detection model:
    1. Load the image
    2. Detect the face
    3. Extract and resize the face
    4. Normalize pixel values
    
    Parameters:
    - image_path: Path to the image file
    - detector: MTCNN face detector instance
    - target_size: Tuple (height, width) required by the model
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to RGB (MTCNN expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(image)
    if not faces:
        print(f"Warning: No face detected in {image_path}")
        # Return resized original image if no face is detected
        return cv2.resize(image, target_size) / 255.0
    
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
    
    # Resize to target size
    face_img = cv2.resize(face_img, target_size)
    
    # Normalize pixel values (as done during training)
    face_img = face_img.astype('float32') / 255.0
    
    return np.expand_dims(face_img, axis=0)  # Add batch dimension for model input

def analyze_image(image_path, model, detector, target_size):
    """Analyze a single image for DeepFake detection"""
    # Preprocess the image
    processed_image = preprocess_image(image_path, detector, target_size)
    if processed_image is None:
        return None
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)[0][0]
    is_fake = prediction > 0.5
    confidence = prediction if is_fake else 1 - prediction
    
    result = {
        "path": image_path,
        "is_fake": bool(is_fake),
        "confidence": float(confidence) * 100,  # Convert to percentage
        "prediction_value": float(prediction)
    }
    
    return result

def visualize_results(image_path, result):
    """
    Visualize the detection results with heatmap overlay
    """
    # Load and display the original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a simple visualization
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Result visualization
    plt.subplot(1, 2, 2)
    prediction_text = "FAKE" if result["is_fake"] else "REAL"
    confidence = result["confidence"]
    
    # Choose color based on prediction (red for fake, green for real)
    color = 'red' if result["is_fake"] else 'green'
    
    # Create a result panel with colored background
    result_img = np.ones((img_rgb.shape[0], img_rgb.shape[1], 3))
    if result["is_fake"]:
        # Red for fake with intensity based on confidence
        intensity = confidence / 100
        result_img[:,:,0] = intensity  # Red channel
    else:
        # Green for real with intensity based on confidence
        intensity = confidence / 100
        result_img[:,:,1] = intensity  # Green channel
    
    plt.imshow(result_img, alpha=0.5)
    plt.title(f"Prediction: {prediction_text}\nConfidence: {confidence:.2f}%", 
              color=color, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Save the visualization
    output_file = f"visualization_{os.path.basename(image_path)}.jpg"
    plt.savefig(output_file)
    print(f"Visualization saved as {output_file}")
    
    # Optionally display the plot (commented out for non-GUI environments)
    # plt.show()
    
    return output_file

def main():
    # Create a better formatted argument parser with clear help messages
    parser = argparse.ArgumentParser(
        description="Deep Fake Detection Model Test Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model.py --image path/to/image.jpg
  python test_model.py --folder path/to/images/folder
  python test_model.py --folder path/to/images/folder --model my_custom_model.h5
        """
    )
    
    # Required arguments group
    input_group = parser.add_argument_group('Input Options (required, choose one)')
    input_group_exclusive = input_group.add_mutually_exclusive_group(required=True)
    input_group_exclusive.add_argument(
        "--image", "-i", 
        help="Path to a single image to analyze"
    )
    input_group_exclusive.add_argument(
        "--folder", "-f", 
        help="Path to a folder of images to analyze"
    )
    
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument(
        "--model", "-m", 
        default="deepfake_detector_model_best.h5",  # Using the best model by default
        help="Model filename to use (default: deepfake_detector_model_best.h5)"
    )
    optional_group.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations of the detection results"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the model with progress bar
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
    print(f"Loading model from {model_path}...")
    
    # First try the specified model, then fall back to alternatives
    model_options = [
        args.model,
        "deepfake_detector_model_best.h5",
        "deepfake_detector_model_fine_tuned.h5",
        "deepfake_detector_model.h5"
    ]
    
    model = None
    for model_name in model_options:
        try_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
        if os.path.exists(try_path):
            model_path = try_path
            print(f"Using model: {model_name}")
            break
    
    if not os.path.exists(model_path):
        print(f"Error: No suitable model file found")
        print("Available models:")
        model_dir = os.path.dirname(os.path.abspath(__file__))
        found_models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        if found_models:
            for model_file in found_models:
                print(f"  - {model_file}")
            print(f"\nTry using: python test_model.py --image <path> --model {found_models[0]}")
        else:
            print("  No .h5 model files found in the current directory.")
            print("  Make sure to train the model first using main.py")
        return
    
    # Create progress bar for model loading
    pbar = tqdm(total=100, desc="Loading model", unit="%")
    
    try:
        # Simulate loading progress
        for i in range(80):
            time.sleep(0.01)  # Small delay to simulate loading
            pbar.update(1)
        
        # Actually load the model
        model = tf.keras.models.load_model(model_path)
        pbar.update(10)
        
        # Get the input shape that the model expects
        input_shape = model.input_shape
        if input_shape is None:
            # Fall back to default size if input shape can't be determined
            target_size = (128, 128)
            print("Warning: Could not determine model input shape, using default (128x128)")
        else:
            height, width = input_shape[1:3]
            target_size = (height, width)
            print(f"Detected model input shape: {height}x{width}")
        
        # Complete the progress bar
        pbar.update(10)
        pbar.close()
        print("Model loaded successfully!")
    except Exception as e:
        pbar.close()
        print(f"Error loading model: {str(e)}")
        return
    
    # Initialize face detector
    print("Initializing face detector...")
    detector = MTCNN()
    
    # Process a single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image {args.image} not found")
            return
        
        print(f"Analyzing image: {args.image}")
        result = analyze_image(args.image, model, detector, target_size)
        
        if result:
            print("\n===== RESULTS =====")
            print(f"Image: {result['path']}")
            print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Raw prediction value: {result['prediction_value']:.4f}")
            
            # Generate visualization if requested
            if args.visualize:
                visualization_file = visualize_results(args.image, result)
                print(f"Visualization saved to {visualization_file}")
        else:
            print("Could not analyze the image.")
            
    # Process a folder of images
    elif args.folder:
        if not os.path.exists(args.folder) or not os.path.isdir(args.folder):
            print(f"Error: Folder {args.folder} not found or is not a directory")
            return
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']  # Added webp support
        image_files = []
        
        # Find all images in the folder
        print("Scanning for images...")
        for filename in os.listdir(args.folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(args.folder, filename))
        
        if not image_files:
            print(f"No image files found in {args.folder}")
            print(f"Supported image formats: {', '.join(image_extensions)}")
            return
            
        print(f"Found {len(image_files)} images. Starting analysis...")
        
        # Process each image with progress bar
        results = []
        for img_path in tqdm(image_files, desc="Analyzing images", unit="img"):
            # Process without printing each file name (progress bar shows progress)
            result = analyze_image(img_path, model, detector, target_size)
            if result:
                results.append(result)
                # Optionally generate visualization
                if args.visualize:
                    visualize_results(img_path, result)
        
        # Print summary
        print("\n===== SUMMARY =====")
        print(f"Total images analyzed: {len(results)}")
        fake_count = sum(1 for r in results if r['is_fake'])
        real_count = len(results) - fake_count
        print(f"Detected as FAKE: {fake_count} ({fake_count/len(results)*100:.1f}%)")
        print(f"Detected as REAL: {real_count} ({real_count/len(results)*100:.1f}%)")
        
        print("\n===== DETAILED RESULTS =====")
        for result in results:
            filename = os.path.basename(result['path'])
            prediction = "FAKE" if result['is_fake'] else "REAL"
            confidence = result['confidence']
            print(f"{filename}: {prediction} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()
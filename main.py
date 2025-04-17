import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import glob

# Configuration for the GPU
print("Checking for GPU availability...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ GPU detected: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print(f"GPU Details:")
        if hasattr(tf.sysconfig, 'get_build_info'):
            build_info = tf.sysconfig.get_build_info()
            if 'cuda_version' in build_info:
                print(f"  CUDA Version: {build_info['cuda_version']}")
            if 'cudnn_version' in build_info:
                print(f"  CUDNN Version: {build_info['cudnn_version']}")
        print(f"  TensorFlow GPU Support: {tf.test.is_built_with_cuda()}")
        print(f"  Is TensorFlow using GPU: {tf.test.is_gpu_available()}")
        
        # Run a quick test operation on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"  Test GPU computation result: {c.numpy().sum()}")
            print("  GPU is working correctly!")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("❌ No GPU found. Using CPU instead.")
    print("\nTroubleshooting tips:")
    print("1. Verify NVIDIA drivers are installed with 'nvidia-smi'")
    print("2. Check TensorFlow GPU support with 'pip list | findstr tensorflow'")
    print("3. Install GPU support with 'pip install tensorflow[and-cuda]'")
    print("4. Make sure to restart Python after installing GPU packages")

# Parameters
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
MARGIN_PERCENT = 0.3  # 30% margin for face crop

# Create a face detector
detector = MTCNN()

def extract_face(image_path, required_size=(IMAGE_SIZE, IMAGE_SIZE), margin_percent=MARGIN_PERCENT):
    """
    Extract face from an image with a margin around it
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB (MTCNN expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(img)
        if not faces:
            # If no face detected, resize the whole image
            return cv2.resize(img, required_size)
        
        # Get the largest face
        face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, width, height = face['box']
        
        # Calculate margins
        margin_x = int(width * margin_percent)
        margin_y = int(height * margin_percent)
        
        # Apply margins to the bounding box
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        width = min(img.shape[1] - x, width + 2 * margin_x)
        height = min(img.shape[0] - y, height + 2 * margin_y)
        
        # Extract the face
        face_img = img[y:y+height, x:x+width]
        
        # Resize to the required size
        face_img = cv2.resize(face_img, required_size)
        
        return face_img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_dataset(metadata_path, dataset_dir):
    """
    Load dataset from metadata and apply preprocessing
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    X_real = []
    X_fake = []
    
    # Process real videos
    real_files = metadata[metadata['label'] == 'REAL']['filename'].tolist()
    print(f"Processing {len(real_files)} real videos...")
    
    real_count = 0
    for filename in real_files:
        # Get the directory name (without the file extension)
        dir_name = os.path.splitext(filename)[0]
        dir_path = os.path.join(dataset_dir, dir_name)
        
        # Get all frame images from the directory (supporting both jpg and png)
        if os.path.exists(dir_path):
            # Look for both .jpg and .png files
            jpg_files = glob.glob(os.path.join(dir_path, "*.jpg"))
            png_files = glob.glob(os.path.join(dir_path, "*.png"))
            image_files = jpg_files + png_files
            
            if not image_files:
                print(f"  No images found in {dir_path}")
                continue
                
            # Take only a subset of frames to avoid memory issues
            for img_path in image_files[:10]:  # Process up to 10 frames per video
                face_img = extract_face(img_path)
                if face_img is not None:
                    X_real.append(face_img)
                    real_count += 1
                    if real_count % 10 == 0:
                        print(f"  Processed {real_count} real frames...")
    
    # Process fake videos
    fake_files = metadata[metadata['label'] == 'FAKE']['filename'].tolist()
    print(f"Processing {len(fake_files)} fake videos...")
    
    fake_count = 0
    for filename in fake_files:
        dir_name = os.path.splitext(filename)[0]
        dir_path = os.path.join(dataset_dir, dir_name)
        
        if os.path.exists(dir_path):
            # Look for both .jpg and .png files
            jpg_files = glob.glob(os.path.join(dir_path, "*.jpg"))
            png_files = glob.glob(os.path.join(dir_path, "*.png"))
            image_files = jpg_files + png_files
            
            if not image_files:
                print(f"  No images found in {dir_path}")
                continue
                
            for img_path in image_files[:10]:  # Process up to 10 frames per video
                face_img = extract_face(img_path)
                if face_img is not None:
                    X_fake.append(face_img)
                    fake_count += 1
                    if fake_count % 10 == 0:
                        print(f"  Processed {fake_count} fake frames...")
    
    print(f"Successfully loaded {len(X_real)} real frames and {len(X_fake)} fake frames.")
    
    # Check if we have any data
    if len(X_real) == 0 and len(X_fake) == 0:
        raise ValueError("Failed to load any valid images. Please check your dataset directories and image files.")
    
    # Handle the case where only one class has data
    if len(X_real) == 0:
        print("WARNING: No real frames loaded. Check your dataset paths.")
    if len(X_fake) == 0:
        print("WARNING: No fake frames loaded. Check your dataset paths.")
    
    # Create labels
    y_real = np.ones(len(X_real))
    y_fake = np.zeros(len(X_fake))
    
    # Combine data
    X = np.array(X_real + X_fake)
    y = np.concatenate([y_real, y_fake])
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    print(f"Final dataset shape: X: {X.shape}, y: {y.shape}")
    return X, y

def create_data_augmentation():
    """
    Create data augmentation pipeline
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])

def build_model():
    """
    Build a deepfake detection model with EfficientNetB0 as base
    """
    # Load the EfficientNetB0 model
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    model = keras.Sequential([
        # Data augmentation
        create_data_augmentation(),
        
        # Pre-trained base model
        base_model,
        
        # Global average pooling
        layers.GlobalAveragePooling2D(),
        
        # Dropout layer for regularization
        layers.Dropout(0.5),
        
        # Dense layer with ReLU
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer with sigmoid activation
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Define file paths
    metadata_path = os.path.join("dataset", "metadata.csv")
    dataset_dir = "dataset"
    
    # Check if metadata exists
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return
    
    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    try:
        X, y = load_dataset(metadata_path, dataset_dir)
        print(f"Dataset loaded. Shape: {X.shape}, Labels shape: {y.shape}")
        
        # Split the dataset into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
        
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Build the model
        print("Building model...")
        model = build_model()
        model.summary()
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train the model
        print("Training model...")
        history = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Evaluate the model
        print("Evaluating model on test set...")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Save the model
        print("Saving model...")
        model.save('deepfake_detector_model.h5')
        print("Model saved as 'deepfake_detector_model.h5'")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
        
        # Fine-tune the model (optional)
        print("Fine-tuning the model by unfreezing some layers...")
        # Unfreeze the top layers of the base model
        for layer in model.layers[1].layers[-20:]:  # Unfreeze the last 20 layers
            layer.trainable = True
            
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model again (fine-tuning)
        history_fine_tune = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=10,  # fewer epochs for fine-tuning
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Evaluate the fine-tuned model
        print("Evaluating fine-tuned model...")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Fine-tuned Test Accuracy: {accuracy:.4f}")
        print(f"Fine-tuned Precision: {precision:.4f}")
        print(f"Fine-tuned Recall: {recall:.4f}")
        print(f"Fine-tuned F1 Score: {f1:.4f}")
        
        # Save the fine-tuned model
        model.save('deepfake_detector_model_fine_tuned.h5')
        print("Fine-tuned model saved as 'deepfake_detector_model_fine_tuned.h5'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import glob
from tqdm import tqdm
from sklearn.utils import class_weight
import time

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
IMAGE_SIZE = 224  # Increased from 128 to 224 for more detail
BATCH_SIZE = 32
EPOCHS = 30  # Increased from 20 to 30
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
    print("Loading metadata file...")
    # Get file size for progress bar
    file_size = os.path.getsize(metadata_path)
    chunk_size = 1024 * 1024  # 1MB chunks
    chunks = file_size // chunk_size + 1
    
    # Create progress bar for loading metadata
    with open(metadata_path, 'r') as f:
        pbar = tqdm(total=chunks, desc="Loading metadata", unit="MB")
        chunks_read = 0
        for _ in pd.read_csv(metadata_path, chunksize=100000):
            chunks_read += 1
            if chunks_read <= chunks:
                pbar.update(1)
        pbar.close()
    
    metadata = pd.read_csv(metadata_path)
    print(f"Metadata loaded successfully with {len(metadata)} entries")
    
    X_real = []
    X_fake = []
    
    # Process real videos
    real_files = metadata[metadata['label'] == 'REAL']['filename'].tolist()
    print(f"Processing {len(real_files)} real videos...")
    
    real_count = 0
    progress_bar_real = tqdm(total=len(real_files), desc="Processing real videos", unit="video")
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
                progress_bar_real.update(1)
                continue
                
            # Take only a subset of frames to avoid memory issues
            for img_path in image_files[:10]:  # Process up to 10 frames per video
                face_img = extract_face(img_path)
                if face_img is not None:
                    X_real.append(face_img)
                    real_count += 1
            
            progress_bar_real.update(1)
    
    progress_bar_real.close()
    print(f"Processed {real_count} real frames from {len(real_files)} videos")
    
    # Process fake videos
    fake_files = metadata[metadata['label'] == 'FAKE']['filename'].tolist()
    print(f"Processing {len(fake_files)} fake videos...")
    
    fake_count = 0
    progress_bar_fake = tqdm(total=len(fake_files), desc="Processing fake videos", unit="video")
    for filename in fake_files:
        dir_name = os.path.splitext(filename)[0]
        dir_path = os.path.join(dataset_dir, dir_name)
        
        if os.path.exists(dir_path):
            # Look for both .jpg and .png files
            jpg_files = glob.glob(os.path.join(dir_path, "*.jpg"))
            png_files = glob.glob(os.path.join(dir_path, "*.png"))
            image_files = jpg_files + png_files
            
            if not image_files:
                progress_bar_fake.update(1)
                continue
                
            for img_path in image_files[:10]:  # Process up to 10 frames per video
                face_img = extract_face(img_path)
                if face_img is not None:
                    X_fake.append(face_img)
                    fake_count += 1
            
            progress_bar_fake.update(1)
    
    progress_bar_fake.close()
    print(f"Processed {fake_count} fake frames from {len(fake_files)} videos")
    
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
    Create enhanced data augmentation pipeline for better generalization
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),  # Increased rotation range
        layers.RandomBrightness(0.2),  # Increased brightness range
        layers.RandomContrast(0.2),  # Increased contrast range
        layers.RandomZoom(0.2),  # Added random zoom
        layers.RandomTranslation(0.1, 0.1),  # Added random translation
        layers.GaussianNoise(0.01),  # Added noise for robustness
    ])

def attention_module(x):
    """
    Simple self-attention mechanism to focus on important features
    """
    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, avg_pool.shape[-1]))(avg_pool)
    avg_pool = layers.Conv2D(filters=x.shape[-1] // 8, kernel_size=1, activation='relu')(avg_pool)
    avg_pool = layers.Conv2D(filters=x.shape[-1], kernel_size=1, activation='sigmoid')(avg_pool)
    
    # Spatial attention
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, max_pool.shape[-1]))(max_pool)
    max_pool = layers.Conv2D(filters=x.shape[-1] // 8, kernel_size=1, activation='relu')(max_pool)
    max_pool = layers.Conv2D(filters=x.shape[-1], kernel_size=1, activation='sigmoid')(max_pool)
    
    # Combine attentions
    attention = layers.Add()([avg_pool, max_pool])
    
    # Apply attention
    return layers.Multiply()([x, attention])

def build_model():
    """
    Build an improved deepfake detection model with EfficientNetB3 base
    """
    # Load the EfficientNetB3 model (upgraded from B0)
    base_model = applications.EfficientNetB3(
        weights='imagenet',
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Create the improved model
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Data augmentation during training only
    augmented = create_data_augmentation()(inputs)
    
    # Pre-trained base model
    x = base_model(augmented, training=False)
    
    # Add attention mechanism to focus on important features
    x = attention_module(x)
    
    # Global pooling with concatenated max and average pooling for better feature representation
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    pooled = layers.Concatenate()([avg_pool, max_pool])
    
    # Dropout for regularization
    x = layers.Dropout(0.5)(pooled)
    
    # First dense layer with regularization
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Second dense layer
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer with sigmoid activation
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
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
        
        # Calculate class weights to handle imbalance
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Build the model
        print("Building improved model...")
        model = build_model()
        model.summary()
        
        # Define callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_auc',  # Monitor AUC instead of just loss
            mode='max',
            patience=7,  # Increased patience
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Add model checkpoint to save best model during training
        model_checkpoint = ModelCheckpoint(
            'deepfake_detector_best_model.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model with more metrics tracking
        print("Training improved model...")
        start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            class_weight=class_weight_dict  # Added class weights to handle imbalance
        )
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Evaluate the model with more metrics
        print("Evaluating improved model on test set...")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Save the model
        print("Saving initial model...")
        model.save('deepfake_detector_model.h5')
        print("Model saved as 'deepfake_detector_model.h5'")
        
        # Plot training history with more metrics
        plt.figure(figsize=(15, 10))
        
        # Accuracy subplot
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Loss subplot
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # AUC subplot
        plt.subplot(2, 2, 3)
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Precision-Recall subplot
        plt.subplot(2, 2, 4)
        plt.plot(history.history['precision'])
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_precision'])
        plt.plot(history.history['val_recall'])
        plt.title('Precision and Recall')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(['Train Precision', 'Train Recall', 'Val Precision', 'Val Recall'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Enhanced training history plot saved as 'training_history.png'")
        
        # Advanced fine-tuning strategy
        print("Performing advanced fine-tuning with gradual unfreezing...")
        
        # Step 1: Unfreeze the top layers of the base model (last 50 layers)
        for layer in model.layers[1].layers[-50:]:
            layer.trainable = True
            
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),  # Lower learning rate
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Train with unfrozen layers (first round)
        print("Fine-tuning stage 1 - training with top 50 layers unfrozen...")
        history_fine_tune1 = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE // 2,  # Smaller batch size for fine-tuning
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            class_weight=class_weight_dict
        )
        
        # Step 2: Unfreeze all layers for final tuning
        for layer in model.layers[1].layers:
            layer.trainable = True
            
        # Recompile with even lower learning rate for final tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Train with all layers unfrozen (final round)
        print("Fine-tuning stage 2 - training with all layers unfrozen...")
        history_fine_tune2 = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE // 4,  # Even smaller batch size
            epochs=5,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            class_weight=class_weight_dict
        )
        
        # Final evaluation on test set
        print("Evaluating final fine-tuned model...")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate comprehensive metrics for fine-tuned model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Fine-tuned Test Accuracy: {accuracy:.4f}")
        print(f"Fine-tuned Precision: {precision:.4f}")
        print(f"Fine-tuned Recall: {recall:.4f}")
        print(f"Fine-tuned F1 Score: {f1:.4f}")
        print(f"Fine-tuned AUC: {auc:.4f}")
        print("Fine-tuned Confusion Matrix:")
        print(conf_matrix)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        labels = ['FAKE', 'REAL']
        cm_display = confusion_matrix(y_test, y_pred)
        plt.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        
        # Add text annotations
        thresh = cm_display.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, format(cm_display[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm_display[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.png')
        
        # Save the final fine-tuned model
        model.save('deepfake_detector_model_fine_tuned.h5')
        print("Fine-tuned model saved as 'deepfake_detector_model_fine_tuned.h5'")
        
        # Load the best model from checkpoints and save it as the final model
        print("Loading best model from checkpoint...")
        best_model = keras.models.load_model('deepfake_detector_best_model.h5')
        best_model.save('deepfake_detector_model_best.h5')
        print("Best model from training saved as 'deepfake_detector_model_best.h5'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed error information

if __name__ == "__main__":
    main()
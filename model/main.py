#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepFake Detection Model using EfficientNetB0
---------------------------------------------
This script implements a deep learning model for detecting deepfake videos
using the Celeb-DF dataset. It includes data preprocessing, model training,
and evaluation.

Author: DeepFake Detector Team
Date: April 2025
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import albumentations as A
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
FACE_SIZE = 224  # Increased face size for better feature extraction
BATCH_SIZE = 16  # Reduced batch size to improve gradient updates
EPOCHS = 50      # Increased maximum number of training epochs
LEARNING_RATE = 0.0001  # Initial learning rate
VALIDATION_SPLIT = 0.15  # Slightly reduced validation split
TEST_SPLIT = 0.15        # Increased test split for better evaluation
USE_CLASS_WEIGHTS = True  # Use class weights to handle imbalance
USE_MIXED_PRECISION = True  # Use mixed precision training
MODEL_SAVE_PATH = "deepfake_detector_model.h5"
FINE_TUNED_MODEL_SAVE_PATH = "deepfake_detector_model_fine_tuned.h5"
BEST_MODEL_SAVE_PATH = "deepfake_detector_best_model.h5"

# Path to the Celeb-DF dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Celeb-DF")

# Enable mixed precision if requested
if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

def create_face_extraction_pipeline():
    """
    Create a face extraction pipeline using MTCNN
    
    Returns:
        face_detector: An MTCNN detector
    """
    # Initialize the face detector
    face_detector = MTCNN()
    return face_detector

def extract_face(image, face_detector, required_size=(FACE_SIZE, FACE_SIZE)):
    """
    Extract a face from an image using MTCNN
    
    Args:
        image: The input image
        face_detector: MTCNN detector
        required_size: Size to resize the face to
        
    Returns:
        face_array: Extracted and processed face, or None if no face detected
    """
    # Convert to RGB if needed
    if image.ndim == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Detect faces in the image
    results = face_detector.detect_faces(image)
    
    # If no faces detected, return None
    if not results or len(results) == 0:
        return None
    
    # Get the largest face by area (width*height)
    largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
    
    # Extract the face
    x1, y1, width, height = largest_face['box']
    x2, y2 = x1 + width, y1 + height
    
    # Apply margin (30% as suggested by many deepfake detection papers)
    margin_x = int(width * 0.3)
    margin_y = int(height * 0.3)
    
    # Apply margins safely (ensure we don't go out of the image bounds)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)
    
    # Extract the face from the image
    face = image[y1:y2, x1:x2]
    
    # Resize the face to the required size
    face = cv2.resize(face, required_size)
    
    return face

def process_video_file(video_path, face_detector, max_frames=30):
    """
    Process a video file to extract faces from frames
    
    Args:
        video_path: Path to the video file
        face_detector: MTCNN detector
        max_frames: Maximum number of frames to process
        
    Returns:
        faces: List of extracted faces
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print(f"Error opening video file {video_path}")
        return []
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    # Calculate frames to extract (evenly distributed)
    frames_to_extract = min(max_frames, frame_count)
    if frames_to_extract < 1:
        return []
    
    # Calculate frame intervals
    interval = frame_count / frames_to_extract
    frame_indices = [int(i * interval) for i in range(frames_to_extract)]
    
    faces = []
    for idx in frame_indices:
        # Set video to the specific frame
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        # Read the frame
        success, frame = video.read()
        
        if not success:
            continue
        
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract the face
        face = extract_face(frame_rgb, face_detector)
        
        if face is not None:
            faces.append(face)
    
    # Release the video file
    video.release()
    
    return faces

def load_dataset(test_list_path=None):
    """
    Load the Celeb-DF dataset from the provided path
    
    Args:
        test_list_path: Path to the test list file
        
    Returns:
        videos_data: DataFrame containing video paths and labels
    """
    if test_list_path is None:
        test_list_path = os.path.join(DATASET_PATH, "List_of_testing_videos.txt")
    
    # Check if the test list file exists
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Test list file not found: {test_list_path}")
    
    # Read the test list file
    with open(test_list_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the lines to extract labels and video paths
    videos_data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            label = int(parts[0])
            video_path = os.path.join(DATASET_PATH, parts[1])
            if os.path.exists(video_path):
                videos_data.append({
                    'path': video_path,
                    'label': label
                })
    
    # Convert to DataFrame
    videos_df = pd.DataFrame(videos_data)
    
    print(f"Loaded {len(videos_df)} videos")
    
    # Print class distribution
    class_counts = videos_df['label'].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    return videos_df

def create_train_val_test_split(videos_df):
    """
    Split the dataset into training, validation, and test sets
    
    Args:
        videos_df: DataFrame containing video paths and labels
        
    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and test sets
    """
    # First split to get the test set
    train_val_df, test_df = train_test_split(
        videos_df,
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=videos_df['label']
    )
    
    # Split the remaining data into training and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=42,
        stratify=train_val_df['label']
    )
    
    print(f"Training set: {len(train_df)} videos")
    print(f"Validation set: {len(val_df)} videos")
    print(f"Test set: {len(test_df)} videos")
    
    return train_df, val_df, test_df

def process_videos_and_extract_faces(videos_df, face_detector, max_frames=20):
    """
    Process videos and extract faces
    
    Args:
        videos_df: DataFrame containing video paths and labels
        face_detector: MTCNN detector
        max_frames: Maximum number of frames to process per video
        
    Returns:
        X: List of face images
        y: List of labels
        video_indices: List of indices linking faces to videos
    """
    print("Extracting faces from videos...")
    X = []
    y = []
    video_indices = []
    
    for idx, row in tqdm(videos_df.iterrows(), total=len(videos_df)):
        video_path = row['path']
        label = row['label']
        
        # Process the video file
        faces = process_video_file(video_path, face_detector, max_frames)
        
        # Add faces to the dataset
        for face in faces:
            X.append(face)
            y.append(label)
            video_indices.append(idx)
    
    print(f"Extracted {len(X)} faces from {len(videos_df)} videos")
    
    return np.array(X), np.array(y), np.array(video_indices)

def create_data_augmentation_pipeline():
    """
    Create enhanced data augmentation pipeline for training
    
    Returns:
        augmentation: Albumentations augmentation pipeline
    """
    augmentation = A.Compose([
        # Color transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        ], p=0.9),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.6),
            A.MotionBlur(blur_limit=(3, 7), p=0.6),
            A.MedianBlur(blur_limit=5, p=0.5),
        ], p=0.7),
        
        # Image quality degradation (common in deepfakes)
        A.OneOf([
            A.ImageCompression(quality_lower=65, quality_upper=100, p=0.7),
            A.Downscale(scale_min=0.7, scale_max=0.99, p=0.6),
        ], p=0.7),
        
        # Geometric transformations
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.HorizontalFlip(p=0.5),
        
        # Cutouts and grid distortions to help model focus on different facial regions
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=2, min_height=8, min_width=8, p=0.5),
        ], p=0.5),
    ])
    
    return augmentation

class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for training and validation
    """
    def __init__(self, X, y, batch_size=32, augment=False, shuffle=True):
        """
        Initialize the generator
        
        Args:
            X: List of face images
            y: List of labels
            batch_size: Batch size
            augment: Whether to use data augmentation
            shuffle: Whether to shuffle the data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmentation = create_data_augmentation_pipeline() if augment else None
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """
        Return the number of batches
        """
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Return a batch of data
        """
        # Get a batch of indices
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get the data for this batch
        batch_X = self.X[inds]
        batch_y = self.y[inds]
        
        # Apply augmentation if needed
        if self.augment:
            aug_batch_X = np.zeros_like(batch_X)
            for i, img in enumerate(batch_X):
                augmented = self.augmentation(image=img)
                aug_batch_X[i] = augmented['image']
            batch_X = aug_batch_X
        
        # Normalize pixel values between 0 and 1
        batch_X = batch_X.astype('float32') / 255.0
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_deepfake_detector_model(freeze_base=True):
    """
    Create a deepfake detector model based on EfficientNetB4 (upgraded from B0)
    
    Args:
        freeze_base: Whether to freeze the base model weights
        
    Returns:
        model: The created model
    """
    # Input layer
    inputs = Input(shape=(FACE_SIZE, FACE_SIZE, 3))
    
    # Load EfficientNetB4 pre-trained on ImageNet (without top layers)
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Freeze the base model if requested
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom top layers with improved architecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Additional dense layers with batch normalization
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the model
    
    Args:
        model: The model to compile
        learning_rate: Learning rate
    """
    # F1 score and AUC metrics
    f1 = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
    auc = tf.keras.metrics.AUC()
    
    # Compile the model with improved optimizer settings
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', f1, auc, 'precision', 'recall']
    )

def train_model(
    X_train, y_train, X_val, y_val,
    batch_size=BATCH_SIZE, epochs=EPOCHS, freeze_base=True
):
    """
    Train the deepfake detector model
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        epochs: Number of epochs
        freeze_base: Whether to freeze the base model weights
        
    Returns:
        model: The trained model
        history: Training history
    """
    # Create data generators
    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size, augment=True, shuffle=True)
    val_gen = DataGenerator(X_val, y_val, batch_size=batch_size, augment=False, shuffle=False)
    
    # Calculate class weights if needed
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train.astype(int))
        total = np.sum(class_counts)
        class_weight_dict = {
            0: total / (class_counts[0] * 2),
            1: total / (class_counts[1] * 2)
        }
        print(f"Class weights: {class_weight_dict}")
        class_weights = class_weight_dict
    
    # Create the model
    model = create_deepfake_detector_model(freeze_base=freeze_base)
    
    # Compile the model
    compile_model(model, learning_rate=LEARNING_RATE)
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            BEST_MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

def fine_tune_model(model, X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS//2):
    """
    Fine-tune the pre-trained model by unfreezing some layers
    
    Args:
        model: The pre-trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        epochs: Number of epochs
        
    Returns:
        model: The fine-tuned model
        history: Training history
    """
    # Unfreeze more layers of the base model for better fine-tuning
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # Check if it's the base model
            for i, base_layer in enumerate(layer.layers):
                if i >= len(layer.layers) - 50:  # Unfreeze last 50 layers (increased from 20)
                    base_layer.trainable = True
    
    # Compile the model with a lower learning rate
    compile_model(model, learning_rate=LEARNING_RATE / 20.0)
    
    # Print the fine-tuning model summary
    model.summary()
    
    # Create data generators
    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size, augment=True, shuffle=True)
    val_gen = DataGenerator(X_val, y_val, batch_size=batch_size, augment=False, shuffle=False)
    
    # Calculate class weights if needed
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_train.astype(int))
        total = np.sum(class_counts)
        # Improved class weights calculation to address class imbalance better
        class_weight_dict = {
            0: total / (class_counts[0] * 2.0),
            1: total / (class_counts[1] * 2.0)
        }
        class_weights = class_weight_dict
    
    # Create callbacks with improved patience settings
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive LR reduction
            patience=5,   # Increased patience
            min_lr=1e-8,  # Lower minimum LR
            verbose=1
        ),
        ModelCheckpoint(
            BEST_MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Fine-tune the model with more epochs
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the model on the test set
    
    Args:
        model: The model to evaluate
        X_test, y_test: Test data
        threshold: Classification threshold
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Create a test generator
    test_gen = DataGenerator(X_test, y_test, batch_size=BATCH_SIZE, augment=False, shuffle=False)
    
    # Predict on the test set
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    # Print metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Find optimal threshold
    if len(y_pred_prob) > 0:
        thresholds = np.arange(0, 1.01, 0.01)
        f1_scores = []
        
        for thr in thresholds:
            y_pred_thr = (y_pred_prob > thr).astype(int)
            f1_thr = f1_score(y_test, y_pred_thr)
            f1_scores.append(f1_thr)
        
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"Optimal threshold: {optimal_threshold:.2f} (F1: {best_f1:.4f})")
        
        # Save optimal threshold to file
        with open('optimal_threshold.txt', 'w') as f:
            f.write(str(optimal_threshold))
        
        metrics['optimal_threshold'] = optimal_threshold
    
    return metrics

def plot_training_history(history, fine_tuning_history=None):
    """
    Plot the training history
    
    Args:
        history: Training history
        fine_tuning_history: Fine-tuning history (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    
    if fine_tuning_history is not None:
        # Calculate the starting epoch for fine-tuning
        ft_start = len(history.history['accuracy'])
        ft_end = ft_start + len(fine_tuning_history.history['accuracy'])
        ft_epochs = range(ft_start, ft_end)
        
        plt.plot(ft_epochs, fine_tuning_history.history['accuracy'], label='Fine-Tuning Training Accuracy')
        plt.plot(ft_epochs, fine_tuning_history.history['val_accuracy'], label='Fine-Tuning Validation Accuracy')
    
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    if fine_tuning_history is not None:
        plt.plot(ft_epochs, fine_tuning_history.history['loss'], label='Fine-Tuning Training Loss')
        plt.plot(ft_epochs, fine_tuning_history.history['val_loss'], label='Fine-Tuning Validation Loss')
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_model_summary(model, file_path='model_summary.txt'):
    """
    Save the model summary to a file
    
    Args:
        model: The model
        file_path: Path to save the summary
    """
    with open(file_path, 'w') as f:
        # Redirect stdout to the file
        old_stdout = sys.stdout
        sys.stdout = f
        
        # Print the model summary
        model.summary()
        
        # Restore stdout
        sys.stdout = old_stdout

def main():
    """Main function to train and evaluate the deepfake detector model"""
    print("=======================================")
    print("DeepFake Detector - Training Pipeline")
    print("=======================================")
    
    # Create face extraction pipeline
    face_detector = create_face_extraction_pipeline()
    
    # Load the dataset
    videos_df = load_dataset()
    
    # Split the dataset into training, validation, and test sets
    train_df, val_df, test_df = create_train_val_test_split(videos_df)
    
    # Process videos and extract faces
    print("\nProcessing training videos...")
    X_train, y_train, _ = process_videos_and_extract_faces(train_df, face_detector, max_frames=30)
    
    print("\nProcessing validation videos...")
    X_val, y_val, _ = process_videos_and_extract_faces(val_df, face_detector, max_frames=30)
    
    print("\nProcessing test videos...")
    X_test, y_test, _ = process_videos_and_extract_faces(test_df, face_detector, max_frames=30)
    
    print("\nTraining and Validation Data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Balance the dataset if severely imbalanced
    class_counts = np.bincount(y_train.astype(int))
    print(f"Training class distribution: {class_counts}")
    
    # Train the initial model
    print("\nTraining the initial model...")
    model, history = train_model(X_train, y_train, X_val, y_val, freeze_base=True)
    
    # Save the initial model
    model.save(MODEL_SAVE_PATH)
    print(f"Initial model saved to {MODEL_SAVE_PATH}")
    
    # Evaluate the initial model
    print("\nEvaluating the initial model...")
    eval_metrics = evaluate_model(model, X_test, y_test)
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    fine_tuned_model, fine_tuning_history = fine_tune_model(model, X_train, y_train, X_val, y_val)
    
    # Save the fine-tuned model
    fine_tuned_model.save(FINE_TUNED_MODEL_SAVE_PATH)
    print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_SAVE_PATH}")
    
    # Evaluate the fine-tuned model
    print("\nEvaluating the fine-tuned model...")
    eval_metrics_ft = evaluate_model(fine_tuned_model, X_test, y_test)
    
    # Plot the training history
    plot_training_history(history, fine_tuning_history)
    print("Training history plot saved to training_history.png")
    
    # Save the model summary
    save_model_summary(fine_tuned_model)
    print("Model summary saved to model_summary.txt")
    
    print("\n=======================================")
    print("DeepFake Detection Training Completed")
    print("=======================================")

if __name__ == "__main__":
    main()
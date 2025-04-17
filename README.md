# DeepFake Detector

A deep learning-based solution for detecting deepfake videos and manipulated facial content using computer vision techniques and neural networks.

## Overview

This project implements a deepfake detection system using EfficientNetB0 as the base architecture. It automatically extracts faces from video frames, processes them, and determines whether the content is genuine or artificially manipulated.

## Features

- **Face Extraction**: Automatically detects and extracts faces from video frames using MTCNN
- **Data Augmentation**: Implements image augmentation for improved model robustness
- **Transfer Learning**: Utilizes EfficientNetB0 pre-trained on ImageNet for feature extraction
- **Fine-tuning**: Includes a two-stage training process with optional fine-tuning
- **Performance Metrics**: Provides comprehensive evaluation with accuracy, precision, recall, and F1 score

## Requirements

- Python 3.8+
- TensorFlow 2.15.0
- OpenCV 4.8.0
- NumPy 1.24.3
- Pandas 2.0.3
- scikit-learn 1.3.0
- matplotlib 3.7.2
- MTCNN 0.1.1

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Deep-Fake-Detector.git
   cd Deep-Fake-Detector
   ```

2. Set up a virtual environment:
   ```
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Verify GPU support (optional but recommended):
   ```
   python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
   ```

## Dataset Structure

The project expects the dataset to be organized as follows:

```
dataset/
├── metadata.csv       # Contains columns: filename, label (REAL/FAKE)
├── video_folder_1/    # Each folder contains frames from a video
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── video_folder_2/
└── ...
```

## Usage

Run the main script to train and evaluate the model:

```
python main.py
```

This will:
1. Load and preprocess the dataset
2. Train the initial model
3. Evaluate performance on the test set
4. Fine-tune the model with unfrozen layers
5. Save both models and training visualizations

## Model Architecture

- Base model: EfficientNetB0 (pretrained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Dropout (0.5)
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.3)
  - Output layer (1 unit, sigmoid activation)

## Results

The model produces:
- A trained model file: `deepfake_detector_model.h5`
- A fine-tuned model file: `deepfake_detector_model_fine_tuned.h5`
- Training history visualization: `training_history.png`
- Performance metrics in the console output

## Future Improvements

- Implement video-level prediction by aggregating frame predictions
- Add support for different architectures (EfficientNetB1-B7, ResNet, etc.)
- Create a user-friendly interface for testing individual videos
- Expand detection capabilities to other manipulation techniques

## License

[MIT License](LICENSE)

## Acknowledgments

- The EfficientNet paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- MTCNN face detection: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
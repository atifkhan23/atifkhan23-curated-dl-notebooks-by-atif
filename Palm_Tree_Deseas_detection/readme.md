# Palm Leaf Disease Classification with EfficientNet and SE Attention

This project implements a deep learning pipeline for classifying diseases in date palm leaves using **EfficientNetB0** with a **Squeeze-and-Excitation (SE) attention block**. The model is built using **TensorFlow** and **Keras** and is designed to detect and classify various diseases based on images of infected palm leaves.

## Project Overview

The goal of this project is to develop a robust deep learning model to classify diseases in date palm leaves. The dataset consists of images organized into categories representing different disease classes. The pipeline includes data preprocessing, model training, evaluation, and visualization of results.

### Key Features
- **Dataset Exploration**: Analyzes image distribution across classes and visualizes sample images.
- **Model Architecture**: Combines EfficientNetB0 (pre-trained on ImageNet) with an SE attention block for enhanced feature representation.
- **Data Augmentation**: Includes random flipping, rotation, and zooming to improve model generalization.
- **Evaluation**: Provides accuracy, classification report, confusion matrix, and ROC curves with AUC scores.
- **Model Saving**: Saves the trained model for future use.

## Dataset

The dataset is sourced from `/kaggle/input/palm-dises-detection/Diseases of date palm leaves dataset/Infected Date Palm Leaves Dataset/Processed`. It is organized into subfolders, each representing a disease class.

### Dataset Analysis
- **Image Count per Category**: A bar chart visualizes the number of images per class to check for data imbalance.
- **Sample Images**: Displays 4 sample images per class for visual inspection of data quality and class-specific features.

### Dataset Preparation
- Images are resized to `224x224`.
- Split into 80% training and 20% testing sets.
- Uses `.cache()` and `.prefetch()` for optimized data loading.
- Labels are one-hot encoded for multi-class classification.

## Model Architecture

The model is based on **EfficientNetB0** with the following components:

1. **Input Augmentations**:
   - Rescaling: Normalizes pixel values from `[0, 255]` to `[0, 1]`.
   - Random horizontal flipping, rotation, and zooming.

2. **Base Model**:
   - EfficientNetB0 pre-trained on ImageNet, fine-tuned for this task.

3. **Squeeze-and-Excitation (SE) Block**:
   - Applies channel-wise attention to enhance important features.
   - Uses global average pooling, dense layers with ReLU and sigmoid activations, and multiplies attention weights with feature maps.

4. **Classification Head**:
   - Global Average Pooling.
   - Dropout (0.2) for regularization.
   - Dense layer with `softmax` for 9-class classification.

## Training

- **Optimizer**: Adam (learning rate = `1e-4`).
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.
- **Callbacks**:
  - `ModelCheckpoint`: Saves the best model based on training accuracy.
  - `ReduceLROnPlateau`: Reduces learning rate when loss plateaus.
- **Epochs**: 50.

## Evaluation

The model is evaluated on the test set with the following metrics:
- **Accuracy**: Computed using `model.evaluate()`.
- **Classification Report**: Includes precision, recall, F1-score, and support per class.
- **Confusion Matrix**: Visualized as a heatmap using Seaborn.
- **ROC Curve and AUC**: Plotted for each class to assess classification performance.

## Results

- **Test Accuracy**: Reported after training (e.g., `XX.XX%`).
- **Visualizations**:
  - Bar chart of image counts per category.
  - Grid of sample images (4 per class).
  - Confusion matrix heatmap.
  - ROC curves with AUC scores for each class.
- **Model Output**: Saved as `final_palm_leaf_model.keras`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd palm-leaf-disease-classification
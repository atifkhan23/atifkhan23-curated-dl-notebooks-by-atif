Sidewalk Hazard Classification
This project implements a deep learning pipeline for classifying sidewalk pavement images as Hazard (damaged) or Not Hazard (not damaged) using multiple convolutional neural network (CNN) models. The models include a Custom CNN, ResNet50, EfficientNetB0, ConvNeXt Tiny, Swin Tiny, and Inception v3, leveraging PyTorch, Albumentations, and timm for robust image classification. The dataset consists of images from damaged and not-damaged pavements, with preprocessing techniques like CLAHE, histogram equalization, and low-light enhancement to handle real-world conditions.
Project Overview
The goal is to detect hazards in sidewalk pavement images to aid in urban safety and maintenance. The pipeline includes data preprocessing, model training, evaluation, and visualization of results, with a focus on robustness to low-light conditions and data augmentation for generalization.
Key Features

Dataset: Images from /kaggle/input/dataset/damaged-20250526T075728Z-1-001/damaged and /kaggle/input/dataset/not-damaged-20250526T075830Z-1-001/not-damaged.
Preprocessing:
Low-light enhancement using CLAHE, histogram equalization, and random darkening.
Data augmentation: Random brightness/contrast, horizontal flips, rotations, and scaling.


Models:
Custom CNN: 5 convolutional layers with batch normalization, ReLU, and max pooling.
ResNet50: Pre-trained, fine-tuned with a 2-class output layer.
EfficientNetB0: Pre-trained, fine-tuned with CLAHE-based preprocessing.
ConvNeXt Tiny and Swin Tiny: Modern architectures from timm, pre-trained and fine-tuned.
Inception v3: Pre-trained with auxiliary logits, fine-tuned for 2 classes.


Evaluation: Accuracy, per-class precision/recall/F1, confusion matrices, and sample predictions.
Visualizations: Training curves (loss/accuracy), confusion matrices, per-class accuracy heatmaps, and sample predictions.

Dataset
The dataset contains images of sidewalks categorized as:

Damaged: Pavements with hazards (e.g., cracks, potholes).
Not Damaged: Pavements without hazards.

Dataset Analysis

Exploration: Visualizes original, CLAHE-enhanced, and augmented images to inspect quality and preprocessing effects.
Preprocessing:
Images resized to 224x224 (299x299 for Inception v3).
Normalized using ImageNet means/std: [0.485, 0.456, 0.406]/[0.229, 0.224, 0.225].
Low-light enhancement via CLAHE or histogram equalization.
Augmentations: Random flips, rotations, brightness/contrast adjustments, and noise.


Split: 70% train, 15% validation, 15% test with stratification to preserve class distribution.

Methodology
Preprocessing

Low-Light Enhancement:
CLAHE (Custom CNN, EfficientNetB0) for contrast enhancement.
Histogram equalization (ConvNeXt, Swin Tiny, Inception v3) for V-channel in HSV.
Random darkening to simulate low-light conditions.


Augmentations:
Custom CNN: Manual augmentations (gamma correction, flips, rotations, scaling).
ResNet50/EfficientNetB0: Albumentations with CLAHE, flips, rotations, and brightness/contrast.
ConvNeXt/Swin Tiny/Inception v3: Torchvision transforms with random cropping, flips, and color jitter.


Data Loading: Uses torch.utils.data.Dataset and DataLoader with stratified splits.

Models

Custom CNN:
5 Conv2d layers (32→64→128→256→512 filters), batch normalization, ReLU, max pooling.
Adaptive average pooling, dropout (0.5), and linear layer for 2 classes.
Parameters: ~2.6M (estimated).


ResNet50:
Pre-trained on ImageNet, fine-tuned with a 2-class fully connected layer.
Parameters: ~25.6M.


EfficientNetB0:
Pre-trained, fine-tuned with a 2-class classifier.
Parameters: ~5.3M.


ConvNeXt Tiny:
Pre-trained via timm, fine-tuned for 2 classes.
Parameters: ~28.6M.


Swin Tiny:
Pre-trained via timm, fine-tuned for 2 classes.
Parameters: ~28.3M.


Inception v3:
Pre-trained with auxiliary logits, fine-tuned for 2 classes.
Parameters: ~27.2M.



Training

Optimizer: Adam (learning rate = 1e-4).
Loss: CrossEntropyLoss.
Epochs: 5–10, with early saving of best model based on validation accuracy.
Batch Size: 16 (Custom CNN, ResNet50, EfficientNetB0), 32 (ConvNeXt, Swin Tiny, Inception v3).

Evaluation

Metrics: Accuracy, precision, recall, F1-score per class, and confusion matrices.
Visualizations:
Training/validation loss and accuracy curves.
Confusion matrices and per-class accuracy heatmaps.
Sample predictions with true/predicted labels.



Results

Custom CNN: Validation/test accuracy reported with detailed classification metrics.
ResNet50: High accuracy due to deep architecture and pre-training.
EfficientNetB0: Efficient performance with CLAHE preprocessing.
ConvNeXt Tiny/Swin Tiny: Modern architectures with competitive accuracy.
Inception v3: Robust performance with auxiliary logits for training stability.
Challenges: Potential class imbalance, low-light robustness, and computational cost of larger models.

Installation

Clone the repository:
git clone <repository-url>
cd sidewalk-hazard-classification


Install dependencies:
pip install -r requirements.txt


Ensure the dataset is available at:

/kaggle/input/dataset/damaged-20250526T075728Z-1-001/damaged
/kaggle/input/dataset/not-damaged-20250526T075830Z-1-001/not-damaged



Usage

Run Image Preprocessing and Visualization:

Execute the preprocessing script to visualize original, CLAHE-enhanced, and augmented images:python preprocess_visualize.py




Run Model Training and Evaluation:

Each model has a separate script:python custom_cnn.py
python resnet50.py
python efficientnet.py
python convnext_swin.py
python inception_v3.py


Alternatively, run all models in a single script:python run_all_models.py




Outputs:

Models: Saved as custom_cnn_best.pth, resnet50_hazard_model.pth, best_convnext_tiny.pth, best_swin_tiny.pth, best_inceptionv3.pth.
Visualizations: Training curves, confusion matrices, per-class accuracy heatmaps, and sample predictions saved as PNGs.
Metrics: Printed classification reports with accuracy, precision, recall, and F1-scores.



Requirements
See requirements.txt for the full list of dependencies. Key libraries include:

PyTorch
torchvision
timm
Albumentations
OpenCV
NumPy
Matplotlib
Seaborn
Scikit-learn

Project Structure
sidewalk-hazard-classification/
├── preprocess_visualize.py      # Script for preprocessing and visualization
├── custom_cnn.py                # Custom CNN training and evaluation
├── resnet50.py                  # ResNet50 training and evaluation
├── efficientnet.py              # EfficientNetB0 training and evaluation
├── convnext_swin.py             # ConvNeXt Tiny and Swin Tiny training
├── inception_v3.py              # Inception v3 training and evaluation
├── run_all_models.py            # Script to run all models
├── custom_cnn_best.pth          # Saved Custom CNN model
├── resnet50_hazard_model.pth    # Saved ResNet50 model
├── best_convnext_tiny.pth       # Saved ConvNeXt Tiny model
├── best_swin_tiny.pth           # Saved Swin Tiny model
├── best_inceptionv3.pth         # Saved Inception v3 model
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation

Future Improvements

Class Imbalance: Apply oversampling or class weighting to handle uneven class distribution.
Advanced Augmentations: Explore CutMix or MixUp for better generalization.
Model Ensemble: Combine predictions from multiple models for improved accuracy.
Hyperparameter Tuning: Perform grid search for learning rate, batch size, and dropout.
Deployment: Create an API for real-time hazard detection using FastAPI or Flask.

License
This project is licensed under the MIT License.
Acknowledgments

Dataset: Sidewalk hazard dataset from Kaggle.
Libraries: PyTorch, torchvision, timm, Albumentations, OpenCV, Scikit-learn, Matplotlib, Seaborn.
Models: Custom CNN, ResNet50, EfficientNetB0, ConvNeXt Tiny, Swin Tiny, Inception v3.


For issues or contributions, please open an issue or submit a pull request.

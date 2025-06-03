Deepfake Detection Project
This project implements a deep learning pipeline for detecting deepfake images by classifying them as Real or Fake. It processes video frames from multiple datasets (DFD, Celeb-DF v2, FaceForensics++), extracts spatial and frequency-domain features (DCT/FFT), and trains multiple models including ResNet50, Vision Transformer (ViT), an FFT-based MLP, and a FusionModel combining spatial and frequency features. The pipeline leverages PyTorch, torchvision, transformers, and OpenCV for robust classification, with evaluation metrics and visualizations to assess performance.
Project Overview
The goal is to develop a robust deepfake detection system capable of distinguishing real images from manipulated ones. The pipeline includes frame extraction, preprocessing with augmentations, feature extraction (DCT/FFT), model training, and comprehensive evaluation using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
Key Features

Datasets:
DeepFake Detection (DFD): Original and manipulated video sequences.
Celeb-DF v2: Real and fake face images.
FaceForensics++ (C23): Original and Deepfakes frames.


Preprocessing:
Frame extraction from videos at 1-second intervals.
Image resizing (224x224), normalization, and augmentations (flips, rotations, Gaussian noise).
Frequency-domain feature extraction using DCT and FFT.


Models:
ResNet50: Pre-trained CNN fine-tuned for binary classification.
Vision Transformer (ViT): Pre-trained google/vit-base-patch16-224-in21k fine-tuned for 2 classes.
FFT-based MLP: Processes flattened DCT features for classification.
FusionModel: Combines ResNet50 spatial features with FFT frequency features.


Evaluation:
Metrics: Accuracy, precision, recall, F1-score, ROC-AUC.
Visualizations: Confusion matrices, accuracy curves, DCT distributions, and sample predictions.
Statistical analysis: ANOVA to compare model accuracies.


Explainability: Supports integration with torchcam and captum for model interpretability.

Datasets
The project combines three datasets to create a balanced dataset of 9,000 real and 9,000 fake images:

DFD: Frames extracted from original and manipulated videos.
Celeb-DF v2: Real and fake face images.
FaceForensics++ (C23): Frames from original and Deepfakes videos.

Dataset Analysis

Frame Extraction: Extracts one frame per second from videos using OpenCV, saving as JPEGs.
Preprocessing:
Images resized to 224x224 and normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
Augmentations: Horizontal flips, ±15° rotations, Gaussian noise.
DCT features: Computed from grayscale images and saved as .npy files.
FFT features: Computed on-the-fly during training for the FusionModel.


Split: 80% train, 20% validation with stratified sampling to maintain class balance.
Combined Dataset: 3,000 images per class from each dataset, stored in /kaggle/working/combined_dataset/real and /kaggle/working/combined_dataset/fake.

Methodology
Preprocessing

Frame Extraction: Extracts frames at 1 fps from videos in DFD dataset, saving to /kaggle/working/frames/real and /kaggle/working/frames/manipulated.
Image Preprocessing:
Resizing to 224x224 (or 299x299 for Inception v3 in earlier iterations).
Normalization and augmentation using torchvision.transforms or manual OpenCV functions.


Frequency Features:
DCT: Computed on grayscale images, flattened, and saved as .npy files.
FFT: Computed during training for FusionModel, using magnitude of 2D FFT on grayscale images.


Data Loading: Custom Dataset classes (ImageDataset, ViTDataset, FFTDataset, FusionDataset) for efficient loading of images and features.

Models

ResNet50:
Pre-trained on ImageNet, fine-tuned with a 2-class fully connected layer.
Input: 3x224x224 RGB images.
Parameters: ~25.6M.


Vision Transformer (ViT):
Pre-trained google/vit-base-patch16-224-in21k, fine-tuned for 2 classes.
Input: 3x224x224 RGB images processed via AutoImageProcessor.
Parameters: ~86M.


FFT-based MLP:
Three-layer MLP (input_dim→512→128→2) for flattened DCT features.
Input: Flattened DCT coefficients (50176 dimensions).
Parameters: ~25.7M (depends on input_dim).


FusionModel:
Combines ResNet50 (spatial) and MLP (frequency) branches.
Spatial: ResNet50 backbone with a linear layer (2048→256).
Frequency: MLP (50176→512→256) on FFT magnitude.
Fusion: Concatenates features (512) and passes through a classifier (512→128→2).
Parameters: ~26M.



Training

Optimizer: Adam (learning rate = 1e-4 for ResNet50/MLP/FusionModel, 1e-5 for ViT).
Loss: CrossEntropyLoss.
Epochs: 3–5, with early saving of best model based on validation accuracy.
Batch Size: 4 (ViT), 16 (ResNet50, FusionModel), 32 (MLP).
Hardware: GPU (CUDA) for accelerated training.

Evaluation

Metrics: Accuracy, precision, recall, F1-score, ROC-AUC.
Visualizations:
Bar plots of dataset counts (DFD, Celeb-DF v2, FaceForensics++).
Sample real/fake faces and DCT distribution histograms.
Training/validation accuracy and loss curves.
Confusion matrices and per-metric comparison bar plots.


Statistical Analysis: ANOVA to assess significant differences in model accuracies.
Prediction: Single-image prediction script for evaluating new images.

Results

ResNet50: Achieves ~95.99% accuracy (epoch 5).
ViT: Achieves ~96.00% accuracy (epoch 3).
FFT-based MLP: Achieves ~79.33% accuracy (epoch 5), limited by frequency-only features.
FusionModel: Combines spatial and frequency features, with performance dependent on tuning.
ANOVA: P-value indicates whether model accuracies differ significantly (threshold: 0.05).
Challenges: Class imbalance in original datasets, computational cost of ViT, and feature dimensionality in FFT.

Installation

Clone the repository:
git clone <repository-url>
cd deepfake-detection


Install dependencies:
pip install -r requirements.txt


Ensure datasets are available at:

/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/
/kaggle/input/celebdf-v2/celebdfv2/
/kaggle/input/faceforensics-c23-processed/ff/ff++/frames/



Usage

Frame Extraction:

Extract frames from DFD videos:python extract_frames.py




Dataset Combination and Preprocessing:

Combine datasets and preprocess images (resize, augment, DCT):python preprocess_dataset.py




Model Training:

Train individual models:python train_resnet.py
python train_vit.py
python train_fft_mlp.py
python train_fusion.py


Alternatively, run all models:python run_all_models.py




Evaluation:

Evaluate models and generate metrics/visualizations:python evaluate_models.py




Single Image Prediction:

Predict on new images:python predict_image.py


Follow prompts to input image paths.


Outputs:

Models: Saved as best_fusion_model_lr0.00005_drop0.25_fold3.pth, etc.
Visualizations: Dataset counts, sample faces, DCT distributions, training curves, confusion matrices, and metric comparisons saved as PNGs.
Metrics: Printed classification reports and ANOVA results.



Requirements
See requirements.txt for the full list of dependencies. Key libraries include:

PyTorch
torchvision
transformers
OpenCV
NumPy
Matplotlib
Seaborn
Scikit-learn
torchcam
captum
tensorboardX

Project Structure
deepfake-detection/
├── extract_frames.py            # Frame extraction from DFD videos
├── preprocess_dataset.py        # Dataset combination and preprocessing
├── train_resnet.py              # ResNet50 training
├── train_vit.py                 # ViT training
├── train_fft_mlp.py             # FFT-based MLP training
├── train_fusion.py              # FusionModel training
├── evaluate_models.py           # Model evaluation and visualizations
├── predict_image.py             # Single image prediction
├── run_all_models.py            # Script to run all models
├── best_fusion_model_*.pth      # Saved FusionModel
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation

Future Improvements

Dataset Expansion: Include more diverse deepfake datasets (e.g., DFDC).
Advanced Augmentations: Implement CutMix or MixUp for better generalization.
Model Ensemble: Combine predictions from ResNet50, ViT, and FusionModel.
Hyperparameter Tuning: Grid search for learning rate, batch size, and dropout.
Explainability: Fully integrate torchcam and captum for Grad-CAM and attribution maps.
Real-Time Detection: Develop a video processing pipeline for real-time deepfake detection.

License
This project is licensed under the MIT License.
Acknowledgments

Datasets: DFD, Celeb-DF v2, FaceForensics++ (C23).
Libraries: PyTorch, torchvision, transformers, OpenCV, Scikit-learn, Matplotlib, Seaborn, torchcam, captum.
Models: ResNet50, ViT, FFT-based MLP, FusionModel.


For issues or contributions, please open an issue or submit a pull request.

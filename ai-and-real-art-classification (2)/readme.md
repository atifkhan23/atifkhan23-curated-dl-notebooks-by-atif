Real vs. AI-Generated Image Classification

This project implements a deep learning pipeline for classifying images as Real or AI-generated using pre-trained convolutional neural network (CNN) and transformer-based models. The models used are EfficientNetB4 and Vision Transformer (ViT-B-16), leveraging PyTorch and torchvision for robust image classification. The dataset used is the Real_AI_SD_LD_Dataset, containing real and AI-generated images, with preprocessing techniques like resizing, normalization, and data augmentation to enhance model performance.

Project Overview

The goal is to distinguish between real and AI-generated images to support applications in content authenticity verification, digital forensics, and media integrity. The pipeline includes data preprocessing, model training, evaluation, and visualization of training/validation accuracy curves.

Key Features





Dataset: Real_AI_SD_LD_Dataset with real and AI-generated images.



Preprocessing:





Image resizing to 224x224.



Normalization using mean [0.5] and std [0.5].



Data augmentation: Random horizontal flips for training.



Models:





EfficientNetB4: Pre-trained on ImageNet, fine-tuned for binary classification.



Vision Transformer (ViT-B-16): Pre-trained on ImageNet, fine-tuned for binary classification.



Evaluation: Training and validation accuracy tracked over epochs.



Visualizations: Plots of training and validation accuracy curves for both models.

Dataset

The Real_AI_SD_LD_Dataset contains images categorized as:





Real: Authentic photographs or non-AI-generated images.



AI-generated: Images created by AI models (e.g., Stable Diffusion, Latent Diffusion).

Dataset Analysis





Location:





Training: /kaggle/input/real-ai-art/Real_AI_SD_LD_Dataset/train



Testing: /kaggle/input/real-ai-art/Real_AI_SD_LD_Dataset/test



Preprocessing:





Images resized to 224x224 to match model input requirements.



Normalized with mean [0.5] and std [0.5] for consistency.



Training augmentation: Random horizontal flips to improve generalization.



Data Loading: Uses torchvision.datasets.ImageFolder for automatic class labeling and DataLoader for efficient batch processing.



Split: Pre-split into train and test sets by the dataset structure.

Methodology

Preprocessing





Transforms:





Training:





Resize to 224x224.



Random horizontal flip (p=0.5).



Convert to tensor and normalize.



Testing:





Resize to 224x224.



Convert to tensor and normalize.



Data Loading:





Batch size: 64.



Number of workers: 2 for parallel data loading.



Shuffling enabled for training to ensure randomness.

Models





EfficientNetB4:





Pre-trained on ImageNet with IMAGENET1K_V1 weights.



Modified classifier: Final linear layer changed to output 2 classes.



Parameters: ~19.3M.



Vision Transformer (ViT-B-16):





Pre-trained on ImageNet with IMAGENET1K_V1 weights.



Modified head: Final linear layer changed to output 2 classes.



Parameters: ~86M.

Training





Optimizer: Adam (learning rate = 0.0001).



Loss: CrossEntropyLoss.



Epochs:





EfficientNetB4: 10 epochs.



ViT-B-16: 5 epochs.



Batch Size: 64.



Hardware: GPU (CUDA) if available, else CPU.



Metrics: Training and validation accuracy computed per epoch.

Evaluation





Metrics: Accuracy on training and validation sets.



Visualizations: Training and validation accuracy curves plotted using Matplotlib.

Results





EfficientNetB4: Achieves high accuracy on both training and validation sets over 10 epochs, with stable convergence.



ViT-B-16: Shows competitive accuracy over 5 epochs, with potential for further improvement with more epochs.



Challenges:





Limited epochs for ViT may restrict performance.



Potential class imbalance in the dataset could affect generalization.



Computational cost of ViT is higher than EfficientNetB4.

Installation





Clone the repository:

git clone <repository-url>
cd real-ai-image-classification



Install dependencies:

pip install -r requirements.txt



Ensure the dataset is available at:





/kaggle/input/real-ai-art/Real_AI_SD_LD_Dataset/train



/kaggle/input/real-ai-art/Real_AI_SD_LD_Dataset/test

Usage





Run Training and Evaluation:





Train and evaluate both models (EfficientNetB4 and ViT-B-16):

python train_models.py



Alternatively, run individual models:

python train_efficientnet.py
python train_vit.py



Outputs:





Models: Saved as efficientnet_b4.pth and vit_b_16.pth (if implemented).



Visualizations: Accuracy curves saved as PNGs.



Metrics: Training and validation accuracy printed per epoch.

Requirements

See requirements.txt for the full list of dependencies. Key libraries include:





PyTorch



torchvision



NumPy



Matplotlib



tqdm

Project Structure

real-ai-image-classification/
├── train_efficientnet.py        # EfficientNetB4 training and evaluation
├── train_vit.py                 # ViT-B-16 training and evaluation
├── train_models.py              # Combined script for both models
├── efficientnet_b4.pth          # Saved EfficientNetB4 model (optional)
├── vit_b_16.pth                 # Saved ViT-B-16 model (optional)
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation

Future Improvements





More Epochs for ViT: Train ViT for 10+ epochs to match EfficientNetB4.



Advanced Augmentations: Add rotations, color jitter, or CutMix for better generalization.



Model Ensemble: Combine predictions from EfficientNetB4 and ViT for improved accuracy.



Hyperparameter Tuning: Experiment with learning rates, batch sizes, and optimizers.



Evaluation Metrics: Include precision, recall, F1-score, and confusion matrices.



Model Saving: Implement checkpointing to save best models based on validation accuracy.
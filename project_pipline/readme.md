Project: Wireless Signal Classification

This project implements a deep learning pipeline for classifying 11 modulation types (e.g., QPSK, QAM16, WBFM) using the RadioML 2016.10A dataset. The pipeline leverages PyTorch to build and train Fully Connected Networks (FCN) and Convolutional Neural Networks (CNN), with improvements over a baseline model through SNR integration, hyperparameter tuning, and robustness testing. The project aligns with concepts from Understanding Deep Learning (UDL) by Simon J.D. Prince.

Project Overview

The goal is to classify wireless signal modulations from I/Q samples under varying Signal-to-Noise Ratios (SNRs, -20 to 18 dB). The dataset contains 220,000 samples (2×128 I/Q pairs) across 11 modulation types. The project is divided into:





Phase 1: Dataset exploration, baseline FCN implementation, and I/Q visualizations.



Phase 2: SNR-aware preprocessing, hyperparameter tuning, detailed evaluation, and robustness testing.



Bonus Phase: CNN-based classifier for improved performance.

Key Features





Data Preprocessing: Normalizes I/Q samples, integrates SNR as a feature for FCN, and retains 2×128 structure for CNN.



Models:





FCN: 257 input features (256 I/Q + 1 SNR), 3 hidden layers (512, 256, 128), ~297K parameters.



CNN: 2×128 input, 3 Conv1D layers (64, 128, 256) with max pooling, ~2.2M parameters.



Evaluation: Per-class F1 scores, confusion matrices, and robustness under Gaussian noise (σ = 0.05, 0.1, 0.2).



Visualizations: I/Q plots, learning curves, and confusion matrices.



Improvements: SNR integration (+5.65% accuracy), CNN architecture (+9.60% accuracy).

Dataset

The RadioML 2016.10A dataset is sourced from a Dropbox link and contains I/Q samples stored in a pickled dictionary. Each sample is a 2×128 array (I and Q components) labeled with one of 11 modulation types and an SNR value.

Dataset Analysis





Exploration: Visualizes I/Q signals for QPSK, QAM16, and WBFM to understand data patterns.



Preprocessing:





FCN: Flattens I/Q to 256 features, adds normalized SNR (257 total).



CNN: Retains 2×128 structure with per-sample normalization.



Split: 60% train, 20% validation, 20% test with stratification to preserve label distribution.

Methodology

Preprocessing





Normalizes I/Q data per sample to stabilize training.



Integrates normalized SNR for FCN (Phase 2B).



Uses LabelEncoder for modulation labels and stratifies splits (UDL Ch. 2, 3).

Models





FCN (Phase 1 & 2):





Input: 257 (256 I/Q + 1 SNR).



Architecture: Linear(512 → 256 → 128), ReLU, Dropout, Linear(11).



Parameters: ~297K.



CNN (Bonus Phase):





Input: 2×128.



Architecture: Conv1D(64 → 128 → 256), MaxPool, Flatten, Linear(512 → 11).



Parameters: ~2.2M.

Training





Loss: CrossEntropyLoss.



Optimizer: Adam with weight decay (0.0001).



Hyperparameter Tuning (Phase 2A):





Learning rates: [0.01, 0.001, 0.0001].



Batch sizes: [64, 128, 256].



Dropout rates: [0.3, 0.5, 0.7].



Best: lr=0.0001, batch=64, dropout=0.3.



Epochs: 20.

Evaluation





Metrics: Accuracy, per-class F1 scores, confusion matrices (Phase 2C).



Robustness: Tests under Gaussian noise (σ = 0.05, 0.1, 0.2) (Phase 2D).



Visualizations: Learning curves, I/Q plots, and confusion matrices saved as PNGs.

Results







Model



Validation Accuracy



Improvement Over Baseline





Baseline FCN



45.43%



—





Optimized FCN



51.08%



+5.65%





CNN



55.03%



+9.60%





F1 Scores (CNN, Phase 2C):





CPFSK: 0.6901



QAM16: 0.2042



Robustness (CNN, Phase 2D):





σ = 0.05: 53.88%



σ = 0.2: 53.50%



Challenges: WBFM misclassification, CNN computational cost.

Installation





Clone the repository:

git clone <repository-url>
cd wireless-signal-classification



Install dependencies:

pip install -r requirements.txt



Download the dataset:





The script automatically downloads the RadioML 2016.10A dataset from the provided Dropbox URL.



Ensure the dataset is extracted to ./radioml_2016_data/.

Usage





Run the Notebook:





Execute the Jupyter notebook (wireless_signal_classification.ipynb) to run the full pipeline:

jupyter notebook wireless_signal_classification.ipynb



The notebook handles data loading, preprocessing, model training, evaluation, and visualization.



Outputs:





Models: Saved as baseline_fcn.pth, best_fcn.pth, and best_cnn.pth.



Visualizations: I/Q plots (iq_plot_*.png), learning curves (*_curves.png), confusion matrices (*_confusion_matrix.png).



Results: Printed accuracies, F1 scores, and robustness metrics.



Network Diagrams:





Create diagrams using draw.io:





FCN: Input(257) → Linear(512) → ReLU → Dropout → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Dropout → Linear(11).



CNN: Input(2, 128) → Conv1D(64) → ReLU → MaxPool → Conv1D(128) → ReLU → MaxPool → Conv1D(256) → ReLU → MaxPool → Flatten → Linear(512) → ReLU → Dropout → Linear(11).



Include in the project report.

Requirements

See requirements.txt for the full list of dependencies. Key libraries include:





PyTorch



NumPy



Matplotlib



Seaborn



Scikit-learn

Project Structure

wireless-signal-classification/
├── wireless_signal_classification.ipynb # Main notebook with full pipeline
├── radioml_2016_data/                  # Dataset directory (auto-downloaded)
├── baseline_fcn.pth                    # Saved baseline FCN model
├── best_fcn.pth                        # Saved optimized FCN model
├── best_cnn.pth                        # Saved CNN model
├── iq_plot_*.png                       # I/Q visualization plots
├── *_curves.png                        # Learning curve plots
├── *_confusion_matrix.png              # Confusion matrix plots
├── requirements.txt                    # List of dependencies
└── README.md                           # Project documentation

Future Improvements





SNR Filtering: Preprocess data to focus on specific SNR ranges for improved accuracy.



Advanced Architectures: Explore ResNet or LSTM-based models for temporal dependencies.



Transfer Learning: Use pre-trained CNNs for faster convergence.



Data Augmentation: Add synthetic noise or rotations to improve robustness.



Cross-Validation: Implement k-fold cross-validation for more reliable metrics.
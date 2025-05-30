Lung Cancer Histopathological Image Classification

Overview
This notebook implements a Convolutional Neural Network (CNN) and a pre-trained ResNet50 model for classifying lung histopathological images into three categories: Benign, Adenocarcinoma, and Squamous Cell Carcinoma. The dataset used is the Lung Cancer Histopathological Images dataset from Kaggle, which contains 15,000 high-resolution (768x768) images, with 5,000 images per class.



The notebook includes:
Data_set Source
https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images

Data Preprocessing and Augmentation: Preparing the dataset with resizing, normalization, and augmentation techniques to improve model generalization.

Custom CNN Model: A convolutional neural network designed for feature extraction and classification.

ResNet50 Transfer Learning: A pre-trained ResNet50 model fine-tuned for lung cancer classification.

Model Training and Evaluation: Training both models, evaluating performance using accuracy, loss, confusion matrices, and classification reports.
Visualization: Displaying sample images, loss/accuracy curves, and confusion matrices for model performance analysis.
Model Comparison: Comparing the performance of the custom CNN and an EfficientNet model based on accuracy and loss metrics.
Dataset
The dataset is sourced from Kaggle: Lung Cancer Histopathological Images dataset, a subset of the LC25000 dataset (Borkowski et al., 2019). It includes:

15,000 images (768x768 pixels) in JPEG format.
Three classes:
Lung Benign Tissue (5,000 images)
Lung Adenocarcinoma (5,000 images)
Lung Squamous Cell Carcinoma (5,000 images)
Fully de-identified and HIPAA compliant.
Generated from 750 histopathological samples with data augmentation for robustness.
Dataset Link: Lung Cancer Histopathological Images on Kaggle

Prerequisites
To run this notebook, ensure you have the following:

Python 3.x environment (preferably a Kaggle kernel or a similar setup).
Required Libraries:
bash

Copy
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
Install them using:
bash

Copy
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
Dataset: Download the dataset from Kaggle and place it in the /kaggle/input/lung-cancer-histopathological-images directory or adjust the base_dir variable in the code to point to your dataset location.
Hardware: A GPU is recommended for faster training due to the high-resolution images and computational complexity.
Notebook Structure
The notebook is organized into the following sections:

Dataset Exploration:
Visualizes sample images from each class (Benign, Adenocarcinoma, Squamous Cell Carcinoma) using matplotlib.
Displays augmented images to demonstrate the effect of data augmentation.

Data Preprocessing:
Resizes images to 768x768 pixels.
Applies data augmentation using ImageDataGenerator (rotation, zoom, flips, etc.).
Splits the dataset into 80% training and 20% validation sets.

Custom CNN Model:
Defines a CNN with multiple convolutional layers, batch normalization, max pooling, and dense layers.
Uses Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
Implements EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model.

ResNet50 Transfer Learning:
Loads a pre-trained ResNet50 model (ImageNet weights) with frozen convolutional layers.
Adds custom classification layers (GlobalAveragePooling, Dense, Dropout, Softmax).
Trains the model with early stopping for efficiency.

Model Evaluation:
Evaluates both models using validation loss and accuracy.
Generates classification reports (precision, recall, F1-score) and confusion matrices.
Plots training/validation loss and accuracy curves for performance analysis.
Model Comparison:
Compares the custom CNN model with an EfficientNet model based on training/validation accuracy and loss.
Usage

Clone or Download:
Clone this repository or download the notebook to your local machine or Kaggle environment.
bash

usage
Set Up the Dataset:
Ensure the dataset is placed in the correct directory (/kaggle/input/lung-cancer-histopathological-images) or update the base_dir variable in the notebook to point to your dataset location.

Run the Notebook:
Open the notebook in a Jupyter environment (e.g., Kaggle, Google Colab, or local Jupyter Notebook).
Execute the cells sequentially to preprocess data, train models, and visualize results.

Note: Training may take significant time due to the high-resolution images (768x768) and model complexity. A GPU is highly recommended.

Expected Outputs:
Visualizations of sample and augmented images.
Training/validation loss and accuracy curves.
Confusion matrices and classification reports for both models.
Comparison plots between the CNN and EfficientNet models.
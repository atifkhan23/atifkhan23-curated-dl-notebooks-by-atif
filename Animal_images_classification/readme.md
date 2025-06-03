mage Classification Project

Overview

This project implements an image classification pipeline using both a traditional machine learning approach (Support Vector Machine) and a deep learning approach (MobileNetV2 with transfer learning). The goal is to classify images from a given dataset into predefined categories, achieving high accuracy on training, validation, and test datasets.

Dataset



 



Images are expected to be organized in subdirectories by class labels (e.g., cat, dog, etc.).



Test image filenames should follow the format test_[label]_[number].jpg (e.g., test_cat_001.jpg).

Requirements

To run this project, install dependencies listed in requirements.txt. The key libraries include:





Python 3.8 or higher



NumPy



Pandas



OpenCV



Seaborn



Matplotlib



TensorFlow



Scikit-learn



Joblib

Install dependencies using:

pip install -r requirements.txt

Setup Instructions





Clone or Download: Clone this repository or download the project files.



Install Dependencies: Run the following command to install required libraries:

pip install -r requirements.txt



Dataset Preparation:





Ensure the training dataset is placed in /kaggle/input/dataset-task-02/Task_02_dataset or update the data_dir path in the scripts.



Ensure the test dataset is placed in /kaggle/input/test-datset/Test_dataset or update the test_data_dir path.



Hardware:





A GPU is recommended for faster preprocessing and training with TensorFlow. The code automatically detects and uses a GPU if available.



If no GPU is available, the code will run on CPU.

Project Structure





SVM Classifier (ml_model(1).pkl):





Uses flattened image pixel values as features.



Employs a linear kernel SVM for classification.



Saves the trained model as ml_model(1).pkl.



MobileNetV2 Model (final_model.keras):





Uses transfer learning with MobileNetV2 pre-trained on ImageNet.



Custom layers added for the specific classification task.



Saves the trained model as final_model.keras.



Test Script:





Evaluates the MobileNetV2 model on the test dataset.



Saves predictions to test_predictions_with_labels.csv.



Visualizes results with a confusion matrix and sample classified images.

Usage





Training the SVM Model:





Run the SVM training script to preprocess images, train the model, and evaluate performance.



Outputs: Model accuracy, classification report, confusion matrix, and saved model (ml_model(1).pkl).



Training the MobileNetV2 Model:





Run the MobileNetV2 training script to preprocess images, fine-tune the model, and evaluate performance.



Outputs: Training/validation accuracy and loss plots, and saved model (final_model.keras).



Testing on New Data:





Run the test script to load the MobileNetV2 model and predict on the test dataset.



Outputs: Test accuracy, confusion matrix, sample classified images, and predictions saved to test_predictions_with_labels.csv.

Results





SVM Model: Achieves high accuracy on the training dataset with a linear kernel, suitable for moderate-sized datasets.



MobileNetV2 Model:





Training Accuracy: 100%



Validation Accuracy: 96%



Test Accuracy: 100% (on 15 test images)



The MobileNetV2 model generalizes well to unseen data, with no misclassifications observed in the test set.

Visualizations





Confusion Matrix: Visualizes model performance on the test dataset.



Sample Classified Images: Displays 9 test images with true and predicted labels, color-coded for correctness.



Training Plots: Shows accuracy and loss curves for training and validation datasets.

Notes





Ensure the dataset paths are correctly set in the scripts.



The test script assumes filenames contain class labels in the format test_[label]_[number].jpg.



For large datasets, consider increasing the batch_size or epochs for better performance.



The code is optimized for both CPU and GPU execution, with GPU preferred for faster training.


# CIFAR-10 Neural Network: Weight Initializer Comparison

This project evaluates the performance of different weight initialization techniques on an Artificial Neural Network (ANN) using the CIFAR-10 image classification dataset.

 
## Objective

To compare the impact of various weight initializers—**He**, **LeCun**, and **Xavier (Glorot)**—on training performance in terms of accuracy and loss.

## Dataset

- **CIFAR-10** dataset: Consists of 60,000 32x32 color images across 10 classes (6,000 images per class).  
- Split into 50,000 training and 10,000 testing samples.  
- Each image is labeled with one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Model Architecture

The model is a fully connected feedforward neural network (ANN) with the following architecture:

- Input: 32x32x3 images flattened into a vector  
- Dense Layer 1: 512 units, ReLU activation  
- Dense Layer 2: 256 units, ReLU activation  
- Dense Layer 3: 128 units, ReLU activation  
- Output Layer: 10 units, Softmax activation  

Each model is compiled with:

- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam (learning rate = 0.001)  
- **Metric**: Accuracy  

## Weight Initializers Compared

- **He Initialization** (`he_uniform`)
- **LeCun Initialization** (`lecun_uniform`)
- **Xavier Initialization** (`glorot_uniform`)

## Training Setup

- **Epochs**: 10  
- **Batch Size**: 128  
- **Validation Data**: CIFAR-10 test set

## Visualizations

For each initializer, the following are plotted:

- Training and Validation Accuracy
- Training and Validation Loss  
- A comparison of validation accuracy and loss across all initializers

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

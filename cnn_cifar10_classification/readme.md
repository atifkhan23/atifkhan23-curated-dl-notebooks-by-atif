# CIFAR-10 Image Classification Using CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. It includes architecture tuning with configurable parameters like filter size, kernel size, learning rate, and dropout.

 

## Dataset: CIFAR-10

- 60,000 color images (32x32x3) in 10 classes
- Training: 50,000 images  
- Testing: 10,000 images  
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## CNN Architecture

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense (128) → Dropout (0.5) → BatchNormalization
- Output: Dense (10) with softmax

---

## Training Configuration

| Parameter       | Value           |
|----------------|------------------|
| Filters         | 64               |
| Kernel Size     | (3, 3)           |
| Learning Rate   | 0.0005           |
| Epochs          | 15               |
| Batch Size      | 32               |
| Loss Function   | Sparse Categorical Crossentropy |
| Optimizer       | Adam             |

---

## Results

- **Test Accuracy**: ~`<insert value from final model>`%
- Random predictions displayed with corresponding predicted class label.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

# MNIST Neural Network Optimization Analysis

This project investigates the performance of different optimization strategies, learning rates, and architectural components (Dropout and Batch Normalization) on the MNIST handwritten digits classification task using TensorFlow/Keras.

 

---

## Project Objectives

1. Compare popular optimizers: **SGD**, **Adam**, **RMSprop**.
2. Study the impact of varying **learning rates** on Adam optimizer.
3. Analyze how different **batch sizes** affect training performance.
4. Evaluate the impact of **Dropout** and **Batch Normalization** on generalization.

---

## Dataset: MNIST

- 70,000 grayscale images of handwritten digits (0–9).
- Training set: 60,000 images  
- Test set: 10,000 images  
- Each image is 28x28 pixels.

---

## Neural Network Architecture

- **Input**: Flattened 28x28 image
- **Dense Layers**: 512, 256, 128 units (ReLU)
- **Output Layer**: 10 units (Softmax)
- **Extras**: Dropout (30%), Batch Normalization (optional)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## Experiments Conducted

### 1. Optimizer Comparison
- **SGD**
- **Adam**
- **RMSprop**

Each model trained for 10 epochs with batch size = 64.

### 2. Learning Rate Comparison (Adam)
- Learning rates tested: `0.0001`, `0.001`, `0.01`

### 3. Batch Size Comparison (Adam)
- Batch sizes tested: `32`, `128`

### 4. Dropout & Batch Normalization
- Compared performance of model **with** vs. **without** Dropout & BatchNorm.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

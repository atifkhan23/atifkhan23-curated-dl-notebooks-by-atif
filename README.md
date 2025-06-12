# Deep Learning Projects & Experiments Repository

This repository contains a curated collection of deep learning projects, lab tasks, and experimental notebooks. It is designed to support hands-on learning and research across various deep learning domains, using real-world datasets and modern frameworks such as TensorFlow and PyTorch.

## Key Features

- Real-world deep learning projects
- Lab tasks and experiments on diverse datasets
- Modular and reproducible code structure
- Implementations using Jupyter Notebooks
- Models built with TensorFlow and PyTorch

## Repository Structure

Each directory represents a standalone project or experiment. These include datasets (or links), preprocessing scripts, model definitions, training routines, and evaluation notebooks.

### Included Projects

- **Animal Image Classification**  
  A convolutional neural network (CNN) trained to classify various animal species from images.

- **Palm Tree Disease Detection**  
  Image-based detection of palm leaf diseases using labeled plant pathology datasets.

- **Damaged vs Non-Damaged Structure Detection**  
  Classification of damaged infrastructure (e.g., buildings or roads) from aerial imagery.

- **AI vs Real Art Classification**  
  A binary classifier that distinguishes between AI-generated and human-created artwork.

- **DeepFake Detection**  
  A model to identify synthetically generated videos based on spatial and temporal features.

- **Oil Spill Detection (No Spill vs Spill)**  
  A classifier to detect the presence of oil spills in marine aerial imagery.

- **Pipeline Automation for Deep Learning Projects**  
  A reusable pipeline template to standardize preprocessing, training, and evaluation for experiments.

- **Lab Tasks and Experiments on Datasets**  
  Additional exploratory notebooks for:
  - Optimizer analysis on MNIST
  - Activation function comparison
  - Time series forecasting with LSTM
  - CIFAR-10 weight initialization studies
  - Transfer learning with ResNet
  - RNN-based airline passenger prediction
  - Visualization and analysis of the Iris dataset

## Requirements

Each project may include its own `requirements.txt` or `environment.yaml`. However, common dependencies across projects include:

- Python 3.8+
- TensorFlow and/or PyTorch
- NumPy, Pandas, Matplotlib, Scikit-learn
- OpenCV (for image processing)
- Jupyter Notebook

To install the general dependencies, run:

```bash
pip install -r requirements.txt

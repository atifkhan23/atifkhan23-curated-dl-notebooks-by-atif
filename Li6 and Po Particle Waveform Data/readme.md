Waveform Feature Analysis: Li6 and Po
This notebook presents a comprehensive analysis of waveform signals collected from nuclear physics experiments involving Lithium-6 (Li6) and Polonium (Po) isotopes. Using multi-channel time-series data, the goal is to explore the distinguishing characteristics of these particle signals through visualization, feature extraction, and preliminary classification insights.

Dataset
The waveform dataset used in this notebook is publicly available on Kaggle:
Waveform Feature Analysis: Li6 and Po Dataset

Data Source
The data originates from real-time experimental measurements conducted in laboratory settings by multiple students as part of nuclear physics research. The experiments captured multi-channel sensor signals representing particle interaction events. The raw signals were preprocessed and organized into compressed NumPy .npz files for efficient analysis.

Contents
Loading and inspecting the Li6 and Po waveform datasets

Visualizing multi-channel time-series signals

Exploring signal characteristics and differences between isotopes

Feature engineering ideas for classification models

GPU usage setup for accelerating model training

Requirements
To run this notebook, the following Python packages are required:

numpy
pandas
matplotlib
torch

You can install dependencies using:

pip install -r requirements.txt
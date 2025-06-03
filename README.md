# Deep Learning Projects Repository

This repository contains a curated collection of deep learning projects focused on real-world applications. The projects span various domains, including image classification, object detection, damage assessment, and anomaly detection, and utilize modern deep learning frameworks such as TensorFlow and PyTorch.

## Repository Structure

Each subdirectory in this repository represents a standalone project with its own dataset, preprocessing scripts, model architecture, training logic, and evaluation results. Projects are organized to promote clarity, reproducibility, and scalability.

### Included Projects

- **Animal Image Classification**  
  A convolutional neural network (CNN) trained to classify various species of animals based on photographic images.

- **Palm Tree Disease Detection**  
  Image-based detection system that identifies diseases affecting palm leaves using a CNN model trained on labeled plant pathology data.

- **Damaged vs Non-Damaged Structure Detection**  
  A deep learning model to detect and classify damaged infrastructure (e.g., buildings or roads) from aerial or satellite imagery.

- **AI vs Real Art Classification**  
  A binary classifier designed to distinguish between AI-generated art and artwork created by human artists.

- **DeepFake Detection**  
  A model to detect synthetically generated facial videos using temporal and spatial cues from video frames.

- **Oil Spill Detection (No Spill vs Spill)**  
  A classification system designed to identify the presence of oil spills in aerial marine imagery.

- **Pipeline Automation for Deep Learning Projects**  
  An end-to-end pipeline template for standardizing preprocessing, training, and evaluation workflows across deep learning experiments.

## Requirements

Each project includes its own `requirements.txt` or `environment.yaml` file. General dependencies across projects include:

- Python 3.8+
- TensorFlow or PyTorch
- NumPy, Pandas, Matplotlib, Scikit-learn
- OpenCV (for image processing)
- Jupyter Notebook (for exploratory analysis and visualization)

Use the following command to install general dependencies:

```bash
pip install -r requirements.txt

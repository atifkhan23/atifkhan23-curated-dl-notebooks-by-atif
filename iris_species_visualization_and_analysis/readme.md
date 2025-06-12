# Iris Species Visualization and Analysis

This project explores the Iris dataset through data visualization techniques to understand the distribution and variation of species based on key floral features.

 

## Project Objectives

- To visualize and analyze the target class (species) in the Iris dataset.
- To compare species based on features such as petal length, sepal width, etc.
- To identify key patterns and variations across Setosa, Versicolor, and Virginica using effective data visualization techniques.

## Dataset Description

The Iris dataset contains 150 samples equally distributed among three species:

1. Setosa  
2. Versicolor  
3. Virginica  

Each sample includes the following features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

This dataset is widely used in machine learning classification tasks.

## Visualizations Included

### 1. Countplot of Species Distribution

- Displays the number of samples per species.
- Shows that the dataset is balanced (50 samples per class).

### 2. Boxplot of Petal Length by Species

- Highlights the distribution and interquartile range (IQR) of petal length for each species.
- Setosa has the smallest petal lengths; Virginica has the largest.

### 3. Violin Plot of Sepal Width by Species

- Shows both distribution shape and summary statistics.
- Setosa shows higher density in certain sepal width ranges.

## Key Insights

- The dataset is balanced.
- Petal length is a key distinguishing feature.
- Sepal width shows significant variation across species.
- Setosa is easily separable, while Versicolor and Virginica exhibit some overlap.

## How to Run

1. Install the required dependencies:
```bash
pip install -r requirements.txt

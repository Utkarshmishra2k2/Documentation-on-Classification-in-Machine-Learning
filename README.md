# Iris Dataset Analysis and Classification

This repository contains a comprehensive analysis and classification pipeline for the Iris dataset. The project demonstrates various data exploration techniques, visualizations, and machine learning models to classify iris species based on their features.

## Project Overview

The Iris dataset is a classic dataset used in machine learning and statistics. It contains measurements of iris flowers and is commonly used for classification tasks. This repository explores the dataset through various methods, including:

- **Data Exploration and Visualization**
- **Feature Engineering and Preprocessing**
- **Model Training and Evaluation**
- **Advanced Visualization Techniques**

## Dataset

The dataset used is the Iris dataset, which includes the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)
- Species (target variable)

## Code Structure

The code in this repository is organized as follows:

1. **Data Exploration and Visualization**
   - Initial exploration and cleanup of the dataset.
   - Generation of correlation heatmaps, scatter plots, box plots, and advanced visualizations (Andrews Curves, Parallel Coordinates, RadViz).

2. **Feature Engineering and Preprocessing**
   - Standardization of features using `StandardScaler`.
   - Splitting the data into training and testing sets.

3. **Model Training and Evaluation**
   - Logistic Regression (using statsmodels).
   - K-Nearest Neighbors (KNN).
   - Support Vector Machine (SVM).
   - Gaussian Naive Bayes.
   - Bernoulli Naive Bayes.
   - Decision Tree (with visualization).
   - Random Forest (with feature importance and decision tree visualization).

4. **Confusion Matrix Visualization**
   - Visualization of confusion matrices for various classifiers.

## Requirements

To run the code, you'll need the following Python packages:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `statsmodels`

2. **Run the Analysis**

   Ensure that all required packages are installed, and then execute the code in a Python environment. The code will perform data exploration, visualize various aspects of the dataset, preprocess the data, train multiple classifiers, and evaluate their performance.

3. **Review Results**

   The code will generate various visualizations and print classification results to the console. Review the outputs to analyze the performance of different models.

## Results

The code includes visualizations and results for:

- **Correlation Heatmap:** Shows the correlation between different features.
- **Scatter Plots:** Visualizes the relationship between features.
- **Box Plots:** Displays feature distributions by species.
- **Confusion Matrices:** Evaluates the performance of classifiers.
- **Model Performance:** Includes accuracy, classification reports, and feature importances.

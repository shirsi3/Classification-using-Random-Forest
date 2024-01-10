# Lab 3: Decision Tree Classifier and Performance Analysis

This script focuses on implementing a Decision Tree Classifier for a dataset, along with preprocessing, feature selection, and performance evaluation.

## Overview of `lab3.py`

- **Functionality:**
  - **Data Loading and Preprocessing:** Loads a dataset and performs preprocessing, which includes dropping specific columns.
  - **Feature Selection:** Uses Variance Threshold for feature selection.
  - **Model Training and Evaluation:** Trains a Decision Tree Classifier and evaluates its performance using confusion matrix and classification report.
  - **Visualization:** Generates plots (potentially saved as PDFs) to visualize certain aspects of data or model performance.

- **Libraries Used:** pandas, sklearn (for metrics, model selection, classifiers, and feature selection), matplotlib (for plotting), and matplotlib.backends.backend_pdf (for PDF generation).

- **Key Functions:**
  - `load_dataset(file_path)`: Function to load the dataset.
  - `preprocess_dataset(dataset)`: Function for dataset preprocessing.
  - Additional functions for model training, evaluation, and visualization.

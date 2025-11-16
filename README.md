## USPS Handwritten Digit Recognition Project

This project aims to perform handwritten digit recognition using the USPS dataset and explore and compare the performance of various algorithms (from basic mathematical statistical methods to deep learning models). In addition, the project includes preprocessing and prediction validation for self-made handwritten images (real-world applications).

## Project Overview

This notebook implements the complete workflow from data download, preprocessing, model building to final evaluation. The main goal is to recognize 16x16 pixel grayscale handwritten digit images. In addition to using the standard test set, self-made handwritten digit images (handwrite/dr and handwrite/mine) are specifically included to test the model's generalization ability on real-world data.

## Techniques and Models Used

This project implements and compares the following six methods:

1. Mean Template (2-Norm):

Calculates the average image of each digit in the training set.

**Euclidean Distance (L2 Norm):** Classifies test samples to the nearest mean template.

2. Singular Value Decomposition (SVD):**

Creates an SVD subspace for each digit category.

Calculates the projection residuals of the test image in each subspace for classification.

3. Higher-Order Singular Value Decomposition (HOSVD):**

Applies tensor decomposition techniques for feature extraction and classification.

4. Random Forest:**

Uses ensemble learning methods from sklearn for classification.

5. XGBoost:**

Uses gradient boosting for efficient classification.

6 SVM - Support Vector Machine(SVM):**
Uses sklearn.svm.SVC for classification.

Suitable for classification problems in high-dimensional feature spaces.

7. Convolutional Neural Network (CNN):**

Constructs a deep learning model using TensorFlow/Keras, including convolutional layers, pooling layers, and fully connected layers.

## Dataset

Main data source: Kaggle (USPS Dataset)

Image specifications: 16x16 pixels, grayscale.

Includes train (training set) and test (test set).

## Custom test data:

handwrite/dr: Additional collected images of handwritten digits.

handwrite/mine: Images of your own handwritten digits.

## Environment Requirements

Project execution requires the following Python packages:

Python 3.x

TensorFlow / Keras

Scikit-learn

XGBoost

OpenCV (cv2)

NumPy, Pandas

Matplotlib, Seaborn

SciPy, h5py

Kaggle API (for downloading data)

## Project Flow

Data preparation:

Set up the Kaggle API and download the USPS dataset (usps.h5).

Read and parse HDF5 format data.

Data Preprocessing:

Color Inversion: Convert the raw USPS data to a "white background, black text" format to conform to common handwriting habits.

Custom Image Processing: Implement the `load_handwrite_enhanced_ROI` function, including binarization, ROI (Region of Interest) cropping, scaling to 16x16, stroke thickening (dilation), and normalization.

Model Training and Prediction:

Implement the above six methods respectively.

Calculate the accuracy on the USPS test set for each method.

Result Visualization:

Draw a confusion matrix to analyze error patterns.

Visualize the residual distribution of SVD/HOSVD.

Demonstrate the effect of custom handwritten images at each preprocessing stage (original image -> invert -> mask -> ROI -> scaling -> thickening).

## Experimental Results

The Notebook includes detailed data analysis, including:

Accuracy comparisons of each model.

Analysis of easily confused numbers (e.g., 4 and 9, 1 and 7).

Comparison of recognition results for custom handwritten images under different models (e.g., the difference in performance between CNN and SVD on real handwritten characters).

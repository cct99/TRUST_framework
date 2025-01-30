# TRUST framework
This repository contains a collection of Jupyter notebooks implementing uncertainty quantification and anomaly detection for MNIST digit classification using LeNet architecture and One-Class Classification (OCC). 

## Repository Contents
## 1. train_test.ipynb
This file implements a data processing pipeline for MNIST dataset. It creates test and training datasets of various sizes. Monte Carlo (MC) inference with dropout for uncertainty estimation will be performed generating these datasets. Training dataset is only used once during offline training and testing dataset will serve as an input for the One Class-Support Vector Machine (OC-SVM) during OC Classification inference. The datasets will have 24 features used by the OCC: 1) top1 and top2 mean, 2) top1 and top2 std_dev, 3) number of times a class appears throughout the Monte Carlo inference (the column header numbers should sum up to the number of MC samples set in the code). Test data will have noise injected. As of now, it is only a simple Gaussian noise. We will further explore other adversarial attacks soon.

## 2. OCC_train.ipynb
This file implements the OC-SVM training with various hyperparameters. Grid search is performed over different configurations of nu- and gamma-values. It generates and saves trained models for each configuration. Then, records performance metrics for hyperparameter analysis to determine the best model.

## 3. OCC_inference.ipynb
This file implements the OC-SVM inference using the trained OC-SVM models with two-stage filtering approach. Prior to OC-SVM predictions, we perform a filter to filter out clean normal data based on a threshold (first quartile value of top1_mean of the training dataset). After filtering, the remaining uncertain images (could be either normal or anomalous) will be sent to the OC-SVM for anomaly detection. 

## Setup and Dependencies
Required Python packages:

tensorflow
numpy
pandas
scikit-learn
matplotlib
tqdm
joblib

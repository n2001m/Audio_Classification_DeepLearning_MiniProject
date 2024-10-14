# Audio Classification Project

## Overview

This repository contains a Jupyter notebook for an audio classification project using the UrbanSound8K dataset. The project focuses on classifying urban sounds into three main categories: siren, traffic, and others, with a particular emphasis on critical sounds like sirens and traffic noises.

## Project Description

The notebook demonstrates the following key steps in audio processing and machine learning:

1. Data Loading and Preprocessing:
   - Loads the UrbanSound8K dataset
   - Performs initial data exploration and class mapping

2. Feature Extraction:
   - Extracts audio features using librosa, including MFCCs, chroma, mel spectrograms, and spectral features

3. Data Augmentation:
   - Implements silence removal and time stretching for data augmentation

4. Model Building:
   - Builds a deep neural network using Keras
   - Implements a Balanced Random Forest Classifier for comparison

5. Training and Evaluation:
   - Trains the models on the processed data
   - Evaluates models using classification reports and confusion matrices

6. Data Handling:
   - Saves extracted features to a CSV file
   - Implements options for handling imbalanced data, including SMOTE and class weighting

7. Model Saving:
   - Saves the trained model for future use

The project showcases skills in audio processing, feature engineering, deep learning, and handling imbalanced datasets in a real-world audio classification task.

## Additional Components

- The repository includes a database file for saving data to MongoDB.
- Other supporting files are present in the repository.

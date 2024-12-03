# EM-ID_MODEL

End-to-End Pipeline for Appliance Classification using SCPI Server

## Project Overview

This project utilizes Electromagnetic Interference (EMI) data for appliance classification. It includes scripts for data collection, preprocessing, model training, and live classification, designed to integrate seamlessly with the SCPI server.

## Key Features

- **Data Collection**: Scripts for collecting live EMI data via the SCPI server.
- **Data Preprocessing**: Feature extraction and transformation of EMI signals.
- **Model Training**:
  - KNN model for interpretable classification.
  - Neural Network for multi-label classification.
- **Live Classification**: Real-time appliance classification from live EMI data.

## Dependencies

- Python 3.10
- TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib
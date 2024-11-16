import os
import json
import pandas as pd
import numpy as np
import redpitaya_scpi as scpi
import time
import datetime
import sys
from scipy import stats
from scipy.signal import welch
import requests
import joblib
from tensorflow.keras.models import load_model

# ---------------------------
# Configuration and Setup
# ---------------------------

def load_config(config_path='config.json'):
    """
    Load configuration parameters from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Validate required keys
    required_keys = [
        "ip_address", "decimation_factor", "base_dir",
        "session_delay_seconds", "aws_api_endpoint", "aws_api_key",
        "model_path", "scaler_path", "mlb_path"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: '{key}'")
    
    return config

# ---------------------------
# Statistical Feature Computation
# ---------------------------

def compute_statistical_features(signal, decimation_factor):
    """
    Compute statistical features from the raw signal data.
    """
    try:
        # Remove DC component
        signal = signal - np.mean(signal)
        
        # Apply Welch's method to estimate power spectral density
        sampling_rate = 125e6 / decimation_factor  # Adjust based on decimation factor
        frequencies, power_spectrum = welch(signal, fs=sampling_rate, nperseg=1024)
        
        # Convert power spectrum to dBm assuming 50 ohms impedance
        Z_ref = 50  # Reference impedance in ohms
        power_dbm = 10 * np.log10(power_spectrum / Z_ref)
        
        # Filter frequencies (10kHz to 100kHz)
        mask = (frequencies >= 10e3) & (frequencies <= 100e3)
        filtered_power_dbm = power_dbm[mask]
        
        # Calculate statistical features
        features = {}
        features['entropy'] = stats.entropy(np.abs(filtered_power_dbm))
        features['skewness'] = stats.skew(filtered_power_dbm)
        features['interquartile_range'] = np.percentile(filtered_power_dbm, 75) - np.percentile(filtered_power_dbm, 25)
        features['kurtosis'] = stats.kurtosis(filtered_power_dbm)
        features['percentile_75'] = np.percentile(filtered_power_dbm, 75)
        features['range'] = np.ptp(filtered_power_dbm)
        features['maximum'] = np.max(filtered_power_dbm)
        features['median'] = np.median(filtered_power_dbm)
        features['percentile_90'] = np.percentile(filtered_power_dbm, 90)
        features['mean_absolute_deviation'] = np.mean(np.abs(filtered_power_dbm - np.mean(filtered_power_dbm)))
        
        return features
    except Exception as e:
        print(f"Error computing statistical features: {e}")
        # Return NaN for all features in case of error
        features = {key: np.nan for key in [
            'entropy', 'skewness', 'interquartile_range', 'kurtosis',
            'percentile_75', 'range', 'maximum', 'median',
            'percentile_90', 'mean_absolute_deviation'
        ]}
        return features

# ---------------------------
# Prediction Function
# ---------------------------

def predict_labels(features, model, scaler, mlb):
    """
    Takes a dictionary of features, uses the trained model to predict labels and their probabilities.
    
    Parameters:
    features (dict): A dictionary containing the feature names and their corresponding values.
                     The keys should match the feature columns used during training.
    model: Loaded Keras model.
    scaler: Loaded scaler object.
    mlb: Loaded MultiLabelBinarizer object.
    
    Returns:
    tuple: A tuple containing:
        - predicted_labels (list): A list of predicted labels.
        - label_probabilities (dict): A dictionary mapping label names to their predicted probabilities.
    """
    # Define the feature columns (should match the columns used during training)
    feature_columns = ['ch1_entropy', 'ch1_skewness', 'ch1_interquartile_range',
                       'ch1_kurtosis', 'ch1_percentile_75', 'ch1_range', 'ch1_maximum',
                       'ch1_median', 'ch1_percentile_90', 'ch1_mean_absolute_deviation',
                       'ch2_entropy', 'ch2_skewness', 'ch2_interquartile_range', 'ch2_kurtosis',
                       'ch2_percentile_75', 'ch2_range', 'ch2_maximum', 'ch2_median',
                       'ch2_percentile_90', 'ch2_mean_absolute_deviation']

    # Ensure all required features are provided
    missing_features = set(feature_columns) - set(features.keys())
    if missing_features:
        raise ValueError(f"The following features are missing: {missing_features}")

    # Create a DataFrame with the feature columns
    X = pd.DataFrame([features], columns=feature_columns)

    # Standardize the features
    X_scaled = scaler.transform(X)

    # Make predictions
    probabilities = model.predict(X_scaled)[0]

    # Apply threshold to get binary predictions
    threshold = 0.5
    predictions_binary = (probabilities >= threshold).astype(int)

    # Ensure predictions_binary is 2D
    predictions_binary = predictions_binary.reshape(1, -1)

    # Get the predicted labels
    predicted_labels = mlb.inverse_transform(predictions_binary)[0]

    # Map label names to probabilities
    label_probabilities = dict(zip(mlb.classes_, probabilities.astype(float)))

    return predicted_labels, label_probabilities

# ---------------------------
# Data Acquisition Functions
# ---------------------------

def send_features_to_aws(api_endpoint, api_key, timestamp, features, label_probabilities):
    """
    Send the features and label probabilities to AWS API Gateway endpoint.
    """
    try:
        # Prepare the payload wrapped in 'body' key
        payload = {
            "body": json.dumps({
                'timestamp': timestamp,
                'features': features,
                'label_probabilities': label_probabilities
            })
        }
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key  # Include API Key if required
        }
        print(payload)
        # Send POST request with the payload
        response = requests.post(api_endpoint, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print(f"{response.text}")
        else:
            print(f"Failed to send data to AWS. Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Error sending data to AWS: {e}")

def data_acquisition(config):
    """
    Main data acquisition function.
    """
    # Extract configurations
    IP = config['ip_address']
    decimation_factor = config['decimation_factor']
    base_dir = config['base_dir']
    session_delay = config['session_delay_seconds']
    api_endpoint = config['aws_api_endpoint']
    api_key = config['aws_api_key']  # API Key for AWS API Gateway
    model_path = config['model_path']
    scaler_path = config['scaler_path']
    mlb_path = config['mlb_path']

    # Load the model, scaler, and mlb once
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Scaler file not found at {scaler_path}")
        sys.exit(1)
    if not os.path.exists(mlb_path):
        print(f"MLB file not found at {mlb_path}")
        sys.exit(1)

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    mlb = joblib.load(mlb_path)

    # Initialize Red Pitaya connection
    try:
        rp_s = scpi.scpi(IP)
        print(f"Connected to Red Pitaya at {IP}")
    except Exception as e:
        print(f"Failed to connect to Red Pitaya at {IP}: {e}")
        sys.exit(1)

    try:
        while True:
            print("Starting data acquisition cycle.")

            # Configure Red Pitaya for acquisition
            try:
                rp_s.tx_txt('ACQ:RST')  # Reset acquisition
                rp_s.tx_txt(f'ACQ:DEC {decimation_factor}')  # Set decimation factor
                rp_s.tx_txt('ACQ:TRIG:LEV -1')  # Set trigger level
                rp_s.tx_txt('ACQ:START')  # Start acquisition
                print("Red Pitaya configured successfully.")
            except Exception as e:
                print(f"Error configuring Red Pitaya: {e}")
                time.sleep(session_delay)
                continue  # Skip to next iteration

            # Trigger acquisition
            try:
                rp_s.tx_txt('ACQ:TRIG CH1_PE')
                rp_s.tx_txt('ACQ:TRIG CH2_PE')
                time.sleep(1)  # Wait for trigger and acquisition
                print("Acquisition triggered successfully.")
            except Exception as e:
                print(f"Error triggering acquisition: {e}")
                time.sleep(session_delay)
                continue  # Skip to next iteration

            # Capture data from both channels
            channel_data = {}
            data_acquired = False  # Flag to check if any data was acquired
            for channel in ['CH1', 'CH2']:
                try:
                    rp_s.tx_txt(f'ACQ:SOUR{1 if channel == "CH1" else 2}:DATA?')
                    raw_data = rp_s.rx_txt().strip('{}\n\r').split(',')
                    signal = np.array([float(x) for x in raw_data])

                    if signal.size > 0:
                        channel_data[channel] = signal
                        data_acquired = True
                        print(f"Data acquired from {channel}")
                    else:
                        print(f"No data captured for {channel}")
                        channel_data[channel] = np.array([])
                except Exception as e:
                    print(f"Error acquiring data from {channel}: {e}")
                    channel_data[channel] = np.array([])

            if data_acquired:
                # Compute features for CH1 and CH2
                features = {}
                for channel in ['CH1', 'CH2']:
                    if channel_data[channel].size > 0:
                        channel_features = compute_statistical_features(channel_data[channel], decimation_factor)
                        # Prefix feature names with channel identifier
                        for key, value in channel_features.items():
                            features[f'{channel.lower()}_{key}'] = value
                    else:
                        # Assign NaN to all features if no data
                        for key in [
                            'entropy', 'skewness', 'interquartile_range', 'kurtosis',
                            'percentile_75', 'range', 'maximum', 'median',
                            'percentile_90', 'mean_absolute_deviation'
                        ]:
                            features[f'{channel.lower()}_{key}'] = np.nan

                # Get current timestamp
                timestamp = datetime.datetime.now().isoformat()

                # Prepare input data for prediction
                input_data = {
                    'timestamp': timestamp,
                    'features': features
                }

                # Perform prediction
                try:
                    predicted_labels, label_probabilities = predict_labels(features, model, scaler, mlb)
                except ValueError as ve:
                    print(f"Prediction Error: {ve}")
                    label_probabilities = {}
                    predicted_labels = []
                except Exception as e:
                    print(f"An error occurred during prediction: {e}")
                    label_probabilities = {}
                    predicted_labels = []

                # Send features and label probabilities to AWS
                send_features_to_aws(api_endpoint, api_key, timestamp, features, label_probabilities)

            # Delay before next acquisition cycle
            time.sleep(session_delay)
    except KeyboardInterrupt:
        print("\nData acquisition stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Data acquisition terminated.")

# ---------------------------
# Entry Point
# ---------------------------

if __name__ == "__main__":
    try:
        config = load_config('config.json')
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    data_acquisition(config)

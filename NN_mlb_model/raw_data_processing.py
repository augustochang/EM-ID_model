import os
import json
import pandas as pd
import numpy as np
import re
from scipy import stats
from scipy.signal import welch
import matplotlib.pyplot as plt  # For histogram computation (optional)

# ---------------------------
# Configuration and Setup
# ---------------------------

def load_config(config_path='config.json'):
    """
    Load configuration parameters from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_keys = ['base_dir']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: '{key}'")
    
    return config

# ---------------------------
# Function to Convert Binary Data to DataFrame
# ---------------------------

def binary_to_dataframe(binary_filepath, input_range=1.0, adc_bits=16):
    """
    Convert a binary file containing interleaved CH1 and CH2 data into a DataFrame.

    Parameters:
        binary_filepath (str): Path to the binary file.
        input_range (float): Input range of the ADC (1:1 divider).
        adc_bits (int): Number of ADC bits

    Returns:
        pd.DataFrame: DataFrame containing voltage_ch1 and voltage_ch2.
    """
    # Read the binary file
    with open(binary_filepath, 'rb') as f:
        raw_data = f.read()
    
    # Convert to 16-bit integers
    samples = np.frombuffer(raw_data, dtype=np.int16)
    
    # Split interleaved data
    ch1_samples = samples[::2]  # Every other sample for CH1
    ch2_samples = samples[1::2]  # Every other sample for CH2

    # Convert raw ADC values to voltages
    voltage_ch1 = np.array(ch1_samples) * (input_range / (2**(adc_bits - 1)))
    voltage_ch2 = np.array(ch2_samples) * (input_range / (2**(adc_bits - 1)))
    
    # Create a DataFrame
    df = pd.DataFrame({
        'voltage_ch1': voltage_ch1,
        'voltage_ch2': voltage_ch2
    })
    
    return df

# ---------------------------
# Statistical Feature Computation
# ---------------------------

def compute_statistical_features(data_array):
    """
    Compute statistical features from a data array (e.g., power spectrum or histogram).
    """
    features = {}
    try:
        features['entropy'] = stats.entropy(np.abs(data_array))
        features['skewness'] = stats.skew(data_array)
        features['interquartile_range'] = np.percentile(data_array, 75) - np.percentile(data_array, 25)
        features['kurtosis'] = stats.kurtosis(data_array)
        features['percentile_75'] = np.percentile(data_array, 75)
        features['range'] = np.ptp(data_array)
        features['maximum'] = np.max(data_array)
        features['median'] = np.median(data_array)
        features['percentile_90'] = np.percentile(data_array, 90)
        features['mean_absolute_deviation'] = np.mean(np.abs(data_array - np.mean(data_array)))
    except Exception as e:
        print(f"Error computing statistical features: {e}")
        features = {key: np.nan for key in ['entropy', 'skewness', 'interquartile_range', 'kurtosis',
                                            'percentile_75', 'range', 'maximum', 'median',
                                            'percentile_90', 'mean_absolute_deviation']}
    return features

# ---------------------------
# Welch Processing Function
# ---------------------------

def process_welch(signal, sampling_rate):
    """
    Process the signal using Welch's method and compute the power spectrum.

    Parameters:
        signal (np.ndarray): The input signal.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        np.ndarray: Filtered power spectrum in dBm.
    """
    frequencies, power_spectrum = welch(signal, fs=sampling_rate, nperseg=1024)
    # Avoid log of zero by replacing zeros with a very small number
    power_spectrum = np.where(power_spectrum == 0, 1e-12, power_spectrum)
    power_dbm = 10 * np.log10(power_spectrum / 50)  # Assuming 50 ohm impedance
    mask = (frequencies >= 10e3) & (frequencies <= 100e3)
    filtered_data = power_dbm[mask]
    return filtered_data

# ---------------------------
# Histogram Processing Function
# ---------------------------

def process_histogram(signal):
    """
    Compute the histogram of the signal.

    Parameters:
        signal (np.ndarray): The input signal.

    Returns:
        np.ndarray: Histogram counts for feature computation.
    """
    hist_counts, bin_edges = np.histogram(signal, bins=50, density=True)
    data_to_analyze = hist_counts  # Use histogram counts for feature computation
    return data_to_analyze

# ---------------------------
# Channel Processing
# ---------------------------

def process_channel_old(signal, channel_label, decimation_factor, processing_method):
    """
    Process a single channel and return computed features.
    """
    try:
        if signal.size == 0:
            print(f"No data in {channel_label}")
            return None
        
        # Remove DC component
        signal = signal - np.mean(signal)

        sampling_rate = 125e6 / decimation_factor

        if processing_method.lower() == 'welch':
            filtered_data = process_welch(signal, sampling_rate)
        elif processing_method.lower() == 'hist':
            filtered_data = process_histogram(signal)
        else:
            raise ValueError(f"Invalid processing method: {processing_method}")

        # Compute statistical features
        features = compute_statistical_features(filtered_data)
        
        # Prefix feature names with channel label
        features = {f"{channel_label}_{key}": value for key, value in features.items()}
        
        return features

    except Exception as e:
        print(f"Error processing {channel_label}: {e}")
        return None
    

def process_channel(signal, channel_label, decimation_factor, processing_method):
    """
    Process a single channel and return computed features.
    """
    try:
        if signal.size == 0:
            print(f"No data in {channel_label}")
            return None
        
        # Remove DC component
        signal = signal - np.mean(signal)

        sampling_rate = 125e6 / decimation_factor

        if processing_method.lower() == 'welch':
            filtered_data = process_welch(signal, sampling_rate)
        elif processing_method.lower() == 'hist':
            filtered_data = process_histogram(signal)
        else:
            raise ValueError(f"Invalid processing method: {processing_method}")

        # Compute statistical features
        features = compute_statistical_features(filtered_data)
        
        # Prefix feature names with channel label and processing method
        features = {f"{channel_label}_{key}_{processing_method}": value for key, value in features.items()}
        
        return features

    except Exception as e:
        print(f"Error processing {channel_label}: {e}")
        return None


# ---------------------------
# Data Augmentation Function
# ---------------------------

def augment_signal(signal, augmentation_method='overlap', overlap_ratio=0.5):
    """
    Augment the signal using specified method.

    Parameters:
        signal (np.ndarray): The input signal.
        augmentation_method (str): The augmentation method to apply.
        overlap_ratio (float): The ratio of overlap between chunks.

    Returns:
        List[np.ndarray]: List of augmented signal chunks.
    """
    augmented_signals = []
    if augmentation_method == 'overlap':
        chunk_size = 16384  # Same as batch_size
        step_size = int(chunk_size * (1 - overlap_ratio))
        for start_idx in range(0, len(signal) - chunk_size + 1, step_size):
            end_idx = start_idx + chunk_size
            augmented_signals.append(signal[start_idx:end_idx])
    else:
        # Implement other augmentation methods if needed
        augmented_signals.append(signal)
    return augmented_signals

# ---------------------------
# Session Processing
# ---------------------------

def process_session(binary_filepath, decimation_factor, batch_size=16384, augment=False):
    """
    Process a single session by computing features for CH1 and CH2 in batches.

    Parameters:
        binary_filepath (str): Path to the binary file.
        decimation_factor (int): Decimation factor for signal processing.
        batch_size (int): Number of samples per batch.
        augment (bool): Whether to augment data using overlapping chunks.

    Returns:
        List[Dict]: List of dictionaries containing features for each batch.
    """
    session_data_list = []
    try:
        # Extract session number from filename
        session_match = re.search(r'session_(\d{3})\.bin', os.path.basename(binary_filepath))
        if not session_match:
            print(f"Invalid session filename format: {binary_filepath}")
            return None
        session_number = int(session_match.group(1))
        
        # Extract date from filepath
        date_str = os.path.basename(os.path.dirname(binary_filepath))
        
        # Read binary file into DataFrame
        df = binary_to_dataframe(binary_filepath)
        
        # Total number of samples
        total_samples = len(df)
        
        # If total_samples is less than batch_size, skip into the next session
        if total_samples < batch_size:
            print(f"Session {session_number} has less than {batch_size} samples. Skipping to next session.")
            return None
        
        # Determine the number of batches
        if augment:
            step_size = int(batch_size * 0.5)  # 50% overlap
            num_batches = (total_samples - batch_size) // step_size + 1
        else:
            num_batches = total_samples // batch_size
            step_size = batch_size

        for batch_index in range(num_batches):
            start_idx = batch_index * step_size
            end_idx = start_idx + batch_size

            # Skip if we don't have enough samples
            if end_idx > total_samples:
                break

            batch_df = df.iloc[start_idx:end_idx]

            batch_data = {}
            batch_data['session'] = session_number
            batch_data['date'] = date_str
            batch_data['batch'] = batch_index

            # Prepare data for CH1 and CH2
            ch1_signal = batch_df['voltage_ch1'].values
            ch2_signal = batch_df['voltage_ch2'].values

            # For each processing method (Welch and Histogram)
            for processing_method in ['welch']:

                # Process CH1
                ch1_features = process_channel(ch1_signal, 'ch1', decimation_factor, processing_method)
                if ch1_features is None:
                    print(f"Failed to process CH1 for batch {batch_index} in session {session_number}")
                    continue

                # # Process CH2
                # ch2_features = process_channel(ch2_signal, 'ch2', decimation_factor, processing_method)
                # if ch2_features is None:
                #     print(f"Failed to process CH2 for batch {batch_index} in session {session_number}")
                #     continue

                # Add features to batch_data with processing method suffix
                for key, value in ch1_features.items():
                    batch_data[f"{key}_{processing_method}"] = value

                # for key, value in ch2_features.items():
                #     batch_data[f"{key}_{processing_method}"] = value

            # Append batch_data to session_data_list
            session_data_list.append(batch_data)

    except Exception as e:
        print(f"Error processing session {binary_filepath}: {e}")
        return None

    return session_data_list
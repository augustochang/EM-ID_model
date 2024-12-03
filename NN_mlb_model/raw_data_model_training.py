# raw_data_model_training.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

def load_data(file_path='master_features.csv'):
    """
    Load and preprocess the master_features.csv file.
    """
    data = pd.read_csv(file_path)
    return data

def filter_features(data, channels=None, preprocessing=None):
    """
    Filter the dataset based on selected channels and preprocessing methods.
    Args:
        data: The DataFrame containing all features.
        channels: List of channels to include (e.g., ['ch1', 'ch2']).
        preprocessing: List of preprocessing methods to include (e.g., ['welch', 'hist']).
    Returns:
        A DataFrame containing only the selected features.
    """
    # Select columns based on channels and preprocessing
    selected_columns = []
    if channels:
        for channel in channels:
            for col in data.columns:
                if channel in col:
                    selected_columns.append(col)
    if preprocessing:
        selected_columns = [col for col in selected_columns if any(pre in col for pre in preprocessing)]
    if not channels and not preprocessing:
        selected_columns = data.columns.tolist()

    # Ensure `device_label` is included
    selected_columns = [col for col in selected_columns if col != "device_label"] + ["device_label"]
    return data[selected_columns]


def prepare_data(data, mlb_filename, scaler_filename):
    """
    Prepare the features and labels for training.
    Args:
        data: DataFrame containing filtered features and labels.
        mlb_filename: Filename to save the MultiLabelBinarizer.
        scaler_filename: Filename to save the Scaler.
    Returns:
        Scaled features, binary label matrix, and additional objects for reuse.
    """
    # Features
    X = data.drop(columns=['device_label'])

    # Labels (multi-label handling)
    y = data['device_label'].str.split('|')  # Split multi-labels by the separator "|"
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    # Save the MultiLabelBinarizer for future use
    joblib.dump(mlb, mlb_filename)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the fitted scaler for future use
    joblib.dump(scaler, scaler_filename)

    return X_scaled, y, mlb, scaler

def build_model(input_shape, output_shape):
    """
    Build and compile the neural network model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_shape, activation='sigmoid')  # Multi-label output with sigmoid activation
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy()])

    return model

def train_model(X_train, X_test, y_train, y_test, model_filename='model.h5'):
    """
    Train the neural network model.
    Args:
        X_train, X_test: Training and testing features.
        y_train, y_test: Training and testing labels.
        model_filename: Filename to save the trained model.
    """
    model = build_model(X_train.shape[1], y_train.shape[1])

    # Early stopping for performance improvement
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=32,
                        callbacks=[early_stopping])

    # Save the trained model
    model.save(model_filename)

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    return model

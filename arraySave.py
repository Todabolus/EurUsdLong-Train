import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(data_dir):
    data = []
    labels = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(label_dir, filename)
                df = pd.read_csv(file_path)
                data.append(df.values)
                labels.append(int(label))
    return data, labels

def flatten_data(data):
    # Flatten the 3D data (samples, timesteps, features) into 2D (samples, features)
    flattened_data = [sample.flatten() for sample in data]
    return np.array(flattened_data)

def save_npy(data, labels, prefix):
    np.save(f'{prefix}_data.npy', data)
    np.save(f'{prefix}_labels.npy', labels)

if __name__ == "__main__":
    # Load training data
    train_data, train_labels = load_data('train_data')
    val_data, val_labels = load_data('val_data')
    test_data, test_labels = load_data('test_data')

    # Flatten the data
    train_data_flat = flatten_data(train_data)
    val_data_flat = flatten_data(val_data)
    test_data_flat = flatten_data(test_data)

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(train_data_flat)

    # Save the scaler
    joblib.dump(scaler, 'data_scaler.joblib')
    print("Scaler saved as 'data_scaler.joblib'.")

    # Scale the data
    train_data_scaled = scaler.transform(train_data_flat)
    val_data_scaled = scaler.transform(val_data_flat)
    test_data_scaled = scaler.transform(test_data_flat)

    # Convert labels to NumPy arrays
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # Save the scaled data and labels
    save_npy(train_data_scaled, train_labels, 'train')
    save_npy(val_data_scaled, val_labels, 'val')
    save_npy(test_data_scaled, test_labels, 'test')

    print("Data scaling and saving completed successfully.")

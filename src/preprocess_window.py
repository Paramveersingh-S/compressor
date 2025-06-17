# src/preprocess_window.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# PARAMETERS
DATA_PATH = os.path.join('data', 'AirQualityUCI.csv')
WINDOW_SIZE = 128
FEATURES = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]

def load_and_clean_data(path, features):
    # Read CSV (semicolon separator, comma decimal)
    df = pd.read_csv(path, sep=';', decimal=',')
    df = df.dropna(axis=1, how='all')  # Drop empty columns
    df = df.dropna()  # Drop rows with missing values
    df = df[features]  # Keep only selected features
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(df.values)
    return norm_data, scaler

def create_windows(data, window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)

def split_data(X, train_ratio=0.7, val_ratio=0.15):
    n = X.shape[0]
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    return X_train, X_val, X_test

if __name__ == "__main__":
    # Load and clean
    df = load_and_clean_data(DATA_PATH, FEATURES)
    print(f"Loaded data shape: {df.shape}")

    # Normalize
    data_norm, scaler = normalize_data(df)
    print(f"Normalized data shape: {data_norm.shape}")

    # Windowing
    X = create_windows(data_norm, WINDOW_SIZE)
    print(f"Windowed data shape: {X.shape} (samples, {WINDOW_SIZE}, features)")

    # Split
    X_train, X_val, X_test = split_data(X)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Save to disk (optional)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/X_test.npy', X_test)
    print("Saved windowed and split data as .npy files in data/")

    # (Optional) Quick plot for EDA
    try:
        import matplotlib.pyplot as plt
        plt.plot(df[FEATURES[0]].values[:1000])
        plt.title(f"First 1000 values of {FEATURES[0]}")
        plt.xlabel("Time Step")
        plt.ylabel("Raw Value")
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping EDA plot.")

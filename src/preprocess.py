import joblib
import numpy as np


def preprocess_input(data, scaler_path='models/scaler.pkl'):
    scaler = joblib.load(scaler_path)

    feature_order = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']

    features = np.array([data[feature] for feature in feature_order]).reshape(1, -1)

    scaled_features = scaler.transform(features)

    return scaled_features
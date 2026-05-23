import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(data_path='data/processed', model_path='models'):
    print("Loading preprocessed data...")
    X_train = joblib.load(f'{data_path}/X_train.pkl')
    y_train = joblib.load(f'{data_path}/y_train.pkl')
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
    }

    os.makedirs(model_path, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        save_path = f'{model_path}/{name}.pkl'
        joblib.dump(model, save_path)
        print(f"Saved to {save_path}")

    print("\nAll models trained and saved successfully.")


if __name__ == '__main__':
    train_models()
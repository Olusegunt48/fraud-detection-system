import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TransactionInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float


app = FastAPI(
    title='Fraud Detection API',
    description='Real time credit card fraud detection using XGBoost',
    version='1.0.0'
)


def load_artifacts():
    model_path  = os.path.join(BASE_DIR, 'models', 'xgboost.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


model, scaler = load_artifacts()

FEATURE_ORDER = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']


@app.get('/')
def health_check():
    return {
        'status': 'online',
        'message': 'Fraud Detection API is running'
    }


@app.get('/model-info')
def model_info():
    return {
        'model': 'XGBoost',
        'features': FEATURE_ORDER,
        'total_features': len(FEATURE_ORDER),
        'output_classes': {
            '0': 'Legitimate',
            '1': 'Fraud'
        }
    }


@app.post('/predict')
def predict(transaction: TransactionInput):
    try:
        features = np.array(
            [getattr(transaction, feature) for feature in FEATURE_ORDER]
        ).reshape(1, -1)

        scaled_features = scaler.transform(features)

        prediction    = model.predict(scaled_features)[0]
        probability   = model.predict_proba(scaled_features)[0][1]

        return {
            'prediction': int(prediction),
            'label': 'Fraud' if prediction == 1 else 'Legitimate',
            'fraud_probability': round(float(probability), 4),
            'status': 'success'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
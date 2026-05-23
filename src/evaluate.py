import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score)


def evaluate_model(model_name, model_path='models', data_path='data/processed'):
    model  = joblib.load(f'{model_path}/{model_name}.pkl')
    X_test = joblib.load(f'{data_path}/X_test.pkl')
    y_test = joblib.load(f'{data_path}/y_test.pkl')

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, preds,
        target_names=['Legitimate', 'Fraud'],
        output_dict=True
    )

    auc = roc_auc_score(y_test, proba)
    cm  = confusion_matrix(y_test, preds)

    print(f"\n{'='*50}")
    print(f"  {model_name.replace('_', ' ').title()}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds,
                                target_names=['Legitimate', 'Fraud']))
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    return {
        'model':     model_name,
        'precision': round(report['Fraud']['precision'], 4),
        'recall':    round(report['Fraud']['recall'], 4),
        'f1':        round(report['Fraud']['f1-score'], 4),
        'roc_auc':   round(auc, 4)
    }


if __name__ == '__main__':
    results = []

    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        result = evaluate_model(model_name)
        results.append(result)

    summary = pd.DataFrame(results).set_index('model')
    print("\nSummary Comparison:")
    print(summary)
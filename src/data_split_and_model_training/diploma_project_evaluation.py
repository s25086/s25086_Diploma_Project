"""
Ewaluacja modelu
Lokalizacja: /opt/airflow/src/diploma_project_evaluation.py
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from pathlib import Path
from datetime import datetime

def evaluate_and_save(model, X_val, y_val, X_test,
                      y_test, X_train, model_type,
                      params, cv_r2,
                      hypertuning, early_stopping_info = None):

    # Używane metryki do mierzenie precyzji modelu: r2, mae, rmse
    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print(f"Validation - R2: {val_r2:.4f} | MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}")
    print(f"Test - R2: {test_r2:.4f} | MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f}")

    # Wylistowanie 10 najistotniejszych kolumn dla każdego modelu
    top_features = []
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:10]
        top_features = list(X_train.columns[indices])
        print(f"Top 10 features: {top_features}")

    model_dir = "/opt/airflow/models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_type": model_type,
        "hypertuning": hypertuning,
        "params": params,
        "cv_r2": float(cv_r2) if cv_r2 else None,
        "early_stopping": early_stopping_info,
        "val": {
            "r2": float(val_r2),
            "mae": float(val_mae),
            "rmse": float(val_rmse)
        },
        "test": {
            "r2": float(test_r2),
            "mae": float(test_mae),
            "rmse": float(test_rmse)
        },
        "top_features": top_features
    }

    with open(f"{model_dir}/{model_type}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    _save_report(model_type, metrics, params, cv_r2, hypertuning, early_stopping_info)
    return metrics

# Zapisywanie raportów do folderu evaluation_reports
def _save_report(model_type, metrics, params,
                 cv_r2, hypertuning, early_stopping_info=None):
    report_dir = "/opt/airflow/reports/evaluation_reports"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/{model_type}_report_{timestamp}.txt"

    with open(report_path, 'w') as f:
        f.write(f"Model evaluation report\n")

        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hyperparameter Tuning: {'Yes' if hypertuning else 'No'}\n\n")

        if cv_r2:
            f.write(f"Cross-Validation R2: {cv_r2:.4f}\n\n")

        if early_stopping_info:
            f.write(f"Early stopping:\n")
            f.write(f"  Zaplanowana liczba drzew:      {early_stopping_info['requested_rounds']}\n")
            f.write(f"  Faktycznie wykorzystana:       {early_stopping_info['actual_rounds']}\n")
            f.write(f"  Cierpliwość (patience_rounds): {early_stopping_info['patience_rounds']}\n")
            f.write(f"  Zatrzymano wcześniej:          {'Tak' if early_stopping_info['stopped_early'] else 'Nie'}\n\n")

        f.write(f"Parameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n")

        f.write(f"Validation set results:\n")
        f.write(f"  R2:     {metrics['val']['r2']:.4f}\n")
        f.write(f"  MAE:    {metrics['val']['mae']:.2f}\n")
        f.write(f"  RMSE:   {metrics['val']['rmse']:.2f}\n\n")

        f.write(f"Test set results:\n")
        f.write(f"  R2:     {metrics['test']['r2']:.4f}\n")
        f.write(f"  MAE:    {metrics['test']['mae']:.2f}\n")
        f.write(f"  RMSE:   {metrics['test']['rmse']:.2f}\n\n")

        if metrics.get('top_features'):
            f.write(f"Top 10 Features:\n")
            for i, feat in enumerate(metrics['top_features'], 1):
                f.write(f"  {i:2d}. {feat}\n")

    print(f"Report saved to {report_path}")
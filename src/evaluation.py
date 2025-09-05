"""
Métricas e avaliação de modelos.
"""
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error
)

def clf_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def clf_auc(y_true, y_proba) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    except Exception:
        return float("nan")

def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse)}

def kfold_score(estimator, X, y, cv: int = 5, scoring: str = "accuracy") -> float:
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return float(scores.mean())

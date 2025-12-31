"""Evaluation metrics utilities."""

import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from typing import Dict


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE), robust to zero values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true_arr != 0
    
    if mask.sum() == 0:
        return 0.0
    
    mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    return mape


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }
    
    return metrics


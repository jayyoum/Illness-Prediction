"""Model training with Optuna optimization."""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def optimize_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> Tuple[Dict, optuna.Study]:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (None for no limit)
    
    Returns:
        Tuple of (best_params, study)
    """
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    
    def objective(trial):
        """Objective function for Optuna."""
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'n_jobs': -1,
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    
    logger.info(f"Best validation RMSE: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    return study.best_params, study


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict
) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train final model on train+val and evaluate on test.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target
        best_params: Best hyperparameters from optimization
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training final model on train+validation data...")
    
    # Combine train and validation
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Train final model
    final_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    final_model.fit(X_train_val, y_train_val)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    # Calculate MAPE (handling zeros)
    y_true_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    mask = y_true_arr != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    else:
        mape = 0.0
    metrics['mape'] = mape
    
    logger.info("Final model evaluation:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return final_model, metrics


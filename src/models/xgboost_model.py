"""XGBoost model training with early stopping and optimization."""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Optional, Tuple
from copy import deepcopy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """XGBoost model trainer with early stopping."""
    
    def __init__(
        self,
        objective: str = 'reg:squarederror',
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize XGBoost trainer.
        
        Args:
            objective: Objective function
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.best_iteration = None
        self.best_rmse = None
    
    def train_with_early_stopping(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
        max_iterations: int = 1000
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            early_stopping_rounds: Rounds without improvement before stopping
            max_iterations: Maximum number of iterations
        
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost with early stopping...")
        
        best_rmse = float('inf')
        best_iteration = 0
        no_improvement_count = 0
        best_model = None
        
        for i in range(max_iterations):
            model = xgb.XGBRegressor(
                objective=self.objective,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                n_estimators=i + 1
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            current_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            if (i + 1) % 50 == 0:
                logger.info(f"Iteration {i+1}, RMSE: {current_rmse:.4f}")
            
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_iteration = i + 1
                best_model = deepcopy(model)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= early_stopping_rounds:
                logger.info(
                    f"Early stopping at iteration {best_iteration} "
                    f"with RMSE: {best_rmse:.4f}"
                )
                break
        
        self.model = best_model
        self.best_iteration = best_iteration
        self.best_rmse = best_rmse
        
        return best_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_with_early_stopping first.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        
        return dict(zip(feature_names, importances))


def train_xgboost_simple(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    params: Optional[Dict] = None
) -> Tuple[xgb.XGBRegressor, Optional[Dict]]:
    """
    Simple XGBoost training without early stopping.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Optional test features
        y_test: Optional test target
        params: XGBoost parameters
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    metrics = None
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    return model, metrics


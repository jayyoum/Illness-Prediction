"""Feature selection utilities (forward selection, backward elimination, RFECV)."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import logging

logger = logging.getLogger(__name__)


def forward_selection_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Optional[dict] = None,
    verbose: bool = True
) -> Tuple[List[str], dict, float]:
    """
    Forward selection using XGBoost and R² score.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        param_grid: Parameter grid for XGBoost
        verbose: Whether to print progress
    
    Returns:
        Tuple of (selected_features, best_params, best_score)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [300, 600, 900],
            'learning_rate': [0.03, 0.05],
            'max_depth': [3, 5],
            'subsample': [0.8],
            'colsample_bytree': [0.75],
            'gamma': [0.1]
        }
    
    remaining = list(X.columns)
    selected = []
    best_score = -np.inf
    best_params = None
    
    from itertools import product
    from sklearn.metrics import r2_score
    
    while remaining:
        scores = []
        
        for candidate in remaining:
            trial_features = selected + [candidate]
            X_trial = X[trial_features]
            
            # Try all parameter combinations
            best_trial_score = -np.inf
            best_trial_params = None
            
            for params in product(*param_grid.values()):
                param_combo = dict(zip(param_grid.keys(), params))
                model = XGBRegressor(
                    **param_combo,
                    objective='reg:squarederror',
                    random_state=42,
                    verbosity=0
                )
                model.fit(X_trial, y)
                y_pred = model.predict(X_trial)
                score = r2_score(y, y_pred)
                
                if score > best_trial_score:
                    best_trial_score = score
                    best_trial_params = param_combo
            
            scores.append((best_trial_score, candidate, best_trial_params))
        
        scores.sort(reverse=True)
        best_new_score, best_candidate, best_candidate_params = scores[0]
        
        if best_new_score > best_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            best_score = best_new_score
            best_params = best_candidate_params
            if verbose:
                logger.info(f"Added: {best_candidate} | R² = {best_score:.4f}")
        else:
            if verbose:
                logger.info("No improvement. Stopping forward selection.")
            break
    
    return selected, best_params, best_score


def rfecv_selection(
    X: pd.DataFrame,
    y: pd.Series,
    step: int = 5,
    min_features: int = 20,
    cv_splits: int = 3,
    scoring: str = 'neg_root_mean_squared_error'
) -> Tuple[List[str], RFECV]:
    """
    Recursive Feature Elimination with Cross-Validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        step: Number of features to remove at each step
        min_features: Minimum number of features to select
        cv_splits: Number of CV splits
        scoring: Scoring metric
    
    Returns:
        Tuple of (selected_features, fitted_selector)
    """
    logger.info("Starting RFECV feature selection...")
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Base estimator
    estimator = XGBRegressor(
        objective='reg:squarederror',
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    # RFECV
    selector = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min_features,
        cv=tscv,
        scoring=scoring,
        verbose=1
    )
    
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.support_].tolist()
    logger.info(f"RFECV selected {len(selected_features)} features")
    
    return selected_features, selector


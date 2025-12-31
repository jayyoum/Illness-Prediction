#!/usr/bin/env python3
"""
Model training script.

This script trains XGBoost models with RFECV feature selection and Optuna optimization.
"""

import argparse
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.engineering import prepare_features
from src.features.selection import rfecv_selection
from src.models.training import optimize_with_optuna, train_final_model
from src.models.xgboost_model import XGBoostTrainer
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    config: dict,
    illness_name: str,
    lag_days: int = 0,
    use_rfecv: bool = True,
    use_optuna: bool = True
) -> None:
    """
    Train model for a specific illness.
    
    Args:
        config: Configuration dictionary
        illness_name: Name of illness to predict
        lag_days: Lag days used in preprocessing
        use_rfecv: Whether to use RFECV for feature selection
        use_optuna: Whether to use Optuna for hyperparameter optimization
    """
    logger.info(f"Training model for {illness_name} (lag={lag_days})")
    
    # Load merged data
    safe_illness_name = illness_name.replace(', ', '_').replace(' ', '_')
    data_path = Path(config['paths']['illness_env_output_dir']) / f"merged_data_{safe_illness_name}_lag{lag_days}.csv"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please run preprocessing first: python scripts/preprocess.py")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare features
    exclude_cols = config['illness']['excluded_vars']
    X, y = prepare_features(
        df,
        target_col='CaseCount',
        exclude_cols=exclude_cols
    )
    
    # Split data
    train_size = int(len(X) * config['training']['train_size'])
    val_size = int(len(X) * config['training']['val_size'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Feature selection with RFECV
    if use_rfecv:
        logger.info("Performing RFECV feature selection...")
        selected_features, selector = rfecv_selection(
            X_train, y_train,
            step=config['model']['rfecv']['step'],
            min_features=config['model']['rfecv']['min_features_to_select'],
            cv_splits=config['model']['rfecv']['cv_splits'],
            scoring=config['model']['rfecv']['scoring']
        )
        
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
    else:
        selected_features = list(X_train.columns)
    
    # Hyperparameter optimization with Optuna
    if use_optuna:
        logger.info("Optimizing hyperparameters with Optuna...")
        best_params, study = optimize_with_optuna(
            X_train, y_train, X_val, y_val,
            n_trials=config['model']['optuna']['n_trials'],
            timeout=config['model']['optuna'].get('timeout')
        )
    else:
        # Use default parameters
        best_params = config['model']['xgboost'].copy()
        best_params.update({
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        })
        study = None
    
    # Train final model
    final_model, metrics = train_final_model(
        X_train, y_train, X_val, y_val, X_test, y_test, best_params
    )
    
    # Save model and results
    output_dir = Path(config['paths']['regression_results_dir']) / safe_illness_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"model_{safe_illness_name}_lag{lag_days}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"Saved model to {model_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / f"metrics_{safe_illness_name}_lag{lag_days}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save selected features
    if use_rfecv:
        features_df = pd.DataFrame({'feature': selected_features})
        features_path = output_dir / f"selected_features_{safe_illness_name}_lag{lag_days}.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved selected features to {features_path}")
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train illness prediction model')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file (default: configs/config.yaml)'
    )
    parser.add_argument(
        '--illness', type=str, default=None,
        help='Illness name to train model for'
    )
    parser.add_argument(
        '--lag', type=int, default=0,
        help='Lag days (default: 0)'
    )
    parser.add_argument(
        '--no-rfecv', action='store_true',
        help='Skip RFECV feature selection'
    )
    parser.add_argument(
        '--no-optuna', action='store_true',
        help='Skip Optuna hyperparameter optimization'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    illness_name = args.illness or config['illness']['default_illness']
    lag_days = args.lag or config['illness']['default_lag_days']
    
    train_model(
        config, illness_name, lag_days,
        use_rfecv=not args.no_rfecv,
        use_optuna=not args.no_optuna
    )


if __name__ == '__main__':
    main()


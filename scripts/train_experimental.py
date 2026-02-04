#!/usr/bin/env python3
"""
Experimental model training script with comprehensive time series features.

This script trains XGBoost models using the expanded feature set:
- Lag features for days 1-14
- Rolling mean features for each lag feature
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
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_experimental_model(
    config: dict,
    illness_name: str,
    lag_days: int = 0,
    use_rfecv: bool = True,
    use_optuna: bool = True
) -> None:
    """
    Train experimental model with comprehensive time series features.
    
    Args:
        config: Configuration dictionary
        illness_name: Name of illness to predict
        lag_days: Lag days used in preprocessing
        use_rfecv: Whether to use RFECV for feature selection
        use_optuna: Whether to use Optuna for hyperparameter optimization
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENTAL MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Illness: {illness_name}")
    logger.info(f"Lag days: {lag_days}")
    logger.info(f"RFECV: {use_rfecv}")
    logger.info(f"Optuna: {use_optuna}")
    logger.info("=" * 60)
    
    # Load experimental data
    safe_illness_name = illness_name.replace(', ', '_').replace(' ', '_')
    data_path = (
        Path(config['paths']['illness_env_output_dir']) / "experimental" /
        f"merged_data_{safe_illness_name}_lag{lag_days}_comprehensive_ts.csv"
    )
    
    if not data_path.exists():
        logger.error(f"Experimental data file not found: {data_path}")
        logger.error("Please run preprocessing first: python scripts/preprocess_experimental.py")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded experimental data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare features
    exclude_cols = config['illness']['excluded_vars']
    X, y = prepare_features(
        df,
        target_col='CaseCount',
        exclude_cols=exclude_cols
    )
    
    logger.info(f"Features after preparation: {X.shape[1]} features")
    
    # Split data
    train_size = int(len(X) * config['training']['train_size'])
    val_size = int(len(X) * config['training']['val_size'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"  Val:   {X_val.shape[0]} samples")
    logger.info(f"  Test:  {X_test.shape[0]} samples")
    
    # Feature selection with RFECV
    if use_rfecv:
        logger.info("\n" + "=" * 60)
        logger.info("RFECV FEATURE SELECTION")
        logger.info("=" * 60)
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
        
        logger.info(f"Selected {len(selected_features)} features from {X.shape[1]} original features")
        logger.info(f"Feature reduction: {X.shape[1]} -> {len(selected_features)}")
    else:
        selected_features = list(X_train.columns)
        logger.info(f"Using all {len(selected_features)} features (no RFECV)")
    
    # Hyperparameter optimization with Optuna
    if use_optuna:
        logger.info("\n" + "=" * 60)
        logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
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
    logger.info("\n" + "=" * 60)
    logger.info("FINAL MODEL TRAINING")
    logger.info("=" * 60)
    final_model, metrics = train_final_model(
        X_train, y_train, X_val, y_val, X_test, y_test, best_params
    )
    
    # Save model and results
    output_dir = Path(config['paths']['regression_results_dir']) / "experimental" / safe_illness_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"model_experimental_{safe_illness_name}_lag{lag_days}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"\nSaved model to {model_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / f"metrics_experimental_{safe_illness_name}_lag{lag_days}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save selected features
    if use_rfecv:
        features_df = pd.DataFrame({'feature': selected_features})
        features_path = output_dir / f"selected_features_experimental_{safe_illness_name}_lag{lag_days}.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved selected features to {features_path}")
    
    # Save feature importance
    import xgboost as xgb
    importance_dict = dict(zip(selected_features, final_model.feature_importances_))
    importance_df = pd.DataFrame([
        {'feature': feat, 'importance': imp}
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    ])
    importance_path = output_dir / f"feature_importance_experimental_{safe_illness_name}_lag{lag_days}.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENTAL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Final metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='EXPERIMENTAL: Train model with comprehensive time series features'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config_experimental.yaml',
        help='Path to experimental config file'
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
    
    train_experimental_model(
        config, illness_name, lag_days,
        use_rfecv=not args.no_rfecv,
        use_optuna=not args.no_optuna
    )


if __name__ == '__main__':
    main()

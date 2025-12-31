#!/usr/bin/env python3
"""
Model evaluation script.

This script evaluates trained models and generates visualizations.
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

from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.plotting import (
    plot_predictions_vs_actual,
    plot_feature_importance,
    plot_optuna_optimization_history,
    plot_optuna_param_importances
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    config: dict,
    illness_name: str,
    lag_days: int = 0
) -> None:
    """
    Evaluate a trained model.
    
    Args:
        config: Configuration dictionary
        illness_name: Name of illness
        lag_days: Lag days used
    """
    logger.info(f"Evaluating model for {illness_name} (lag={lag_days})")
    
    safe_illness_name = illness_name.replace(', ', '_').replace(' ', '_')
    
    # Load model
    model_dir = Path(config['paths']['regression_results_dir']) / safe_illness_name
    model_path = model_dir / f"model_{safe_illness_name}_lag{lag_days}.pkl"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train the model first: python scripts/train.py")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    data_path = Path(config['paths']['illness_env_output_dir']) / f"merged_data_{safe_illness_name}_lag{lag_days}.csv"
    df = pd.read_csv(data_path)
    
    # Load selected features if available
    features_path = model_dir / f"selected_features_{safe_illness_name}_lag{lag_days}.csv"
    if features_path.exists():
        selected_features_df = pd.read_csv(features_path)
        selected_features = selected_features_df['feature'].tolist()
    else:
        # Use all numeric features
        exclude_cols = config['illness']['excluded_vars'] + ['CaseCount']
        selected_features = [
            col for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    # Prepare test data
    train_size = int(len(df) * config['training']['train_size'])
    val_size = int(len(df) * config['training']['val_size'])
    
    X_test = df[selected_features][train_size + val_size:]
    y_test = df['CaseCount'][train_size + val_size:]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test.values, y_pred)
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Generate plots
    plots_dir = Path(config['paths']['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Predictions vs actual
    plot_predictions_vs_actual(
        y_test, y_pred,
        title=f"Predictions vs. Actual Values - {illness_name}",
        save_path=str(plots_dir / f"predictions_{safe_illness_name}_lag{lag_days}.png")
    )
    
    # Feature importance
    plot_feature_importance(
        model,
        max_features=20,
        title=f"Feature Importance - {illness_name}",
        save_path=str(plots_dir / f"feature_importance_{safe_illness_name}_lag{lag_days}.png")
    )
    
    logger.info(f"Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file (default: configs/config.yaml)'
    )
    parser.add_argument(
        '--illness', type=str, default=None,
        help='Illness name to evaluate'
    )
    parser.add_argument(
        '--lag', type=int, default=0,
        help='Lag days (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    illness_name = args.illness or config['illness']['default_illness']
    lag_days = args.lag or config['illness']['default_lag_days']
    
    evaluate_model(config, illness_name, lag_days)


if __name__ == '__main__':
    main()


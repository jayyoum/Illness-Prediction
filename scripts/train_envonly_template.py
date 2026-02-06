#!/usr/bin/env python3
"""
Environmental-Only training template
Research Focus: Pure environmental exposure effects
Excludes: Case_Count features, Region one-hot encoding, Year, Season
Includes: Environmental lags, DayOfYear, DayOfWeek, all other environmental vars
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import yaml
import logging
import pickle
from datetime import datetime
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna

def setup_logging(config, illness_short_name):
    """Setup logging configuration"""
    log_dir = project_root / config['logging']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"train_{illness_short_name}_envonly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def prepare_data_envonly(df, config, logger):
    """
    Prepare features for ENVIRONMENTAL-ONLY model
    Excludes: Case_Count features, Region encoding, Year, Season, RegionName
    Includes: All environmental variables, their lags/rolling stats, DayOfYear, DayOfWeek
    """
    # Columns to EXCLUDE completely
    exclude_cols = [
        config['data']['target_column'],  # Case_Count (target)
        config['data']['datetime_column'],  # DateTime
        'IllnessName',
        'RegionName',  # EXCLUDE region name
        'Season',      # EXCLUDE season  
        'Year',        # EXCLUDE year (but keep Year_x if it's from merging)
        'Year_x',      # EXCLUDE year
        'Year_y',      # EXCLUDE year
        'Region'       # EXCLUDE Region column itself (we don't want one-hot encoding)
    ]
    
    # Get all column names
    all_cols = df.columns.tolist()
    
    # Identify and exclude ALL Case_Count-related features
    case_count_features = [col for col in all_cols if 'Case_Count' in col or 'CaseCount' in col]
    
    logger.info(f"Excluding {len(case_count_features)} Case_Count-related features:")
    for feat in sorted(case_count_features)[:10]:  # Show first 10
        logger.info(f"  - {feat}")
    if len(case_count_features) > 10:
        logger.info(f"  ... and {len(case_count_features) - 10} more")
    
    # Combine all exclusions
    all_exclusions = list(set(exclude_cols + case_count_features))
    
    # Select feature columns (everything NOT in exclusions)
    feature_cols = [col for col in all_cols if col not in all_exclusions]
    
    logger.info(f"\nFeature selection summary:")
    logger.info(f"  Total columns in data: {len(all_cols)}")
    logger.info(f"  Excluded columns: {len(all_exclusions)}")
    logger.info(f"  Selected feature columns: {len(feature_cols)}")
    
    X = df[feature_cols].copy()
    y = df[config['data']['target_column']].copy()
    
    # Check what features we're actually using
    env_lag_features = [col for col in X.columns if '_lag_' in col]
    temporal_features = [col for col in X.columns if col in ['DayOfYear', 'DayOfWeek', 'Month']]
    base_env_features = [col for col in X.columns if col not in env_lag_features and col not in temporal_features]
    
    logger.info(f"\nFeature breakdown:")
    logger.info(f"  Environmental lag features: {len(env_lag_features)}")
    logger.info(f"  Temporal features: {len(temporal_features)} {temporal_features}")
    logger.info(f"  Base environmental features: {len(base_env_features)}")
    
    logger.info(f"\nTarget: {config['data']['target_column']}")
    logger.info(f"Final feature count: {X.shape[1]}")
    
    return X, y

def perform_rfecv(X_train, y_train, config, logger):
    """Perform Recursive Feature Elimination with Cross-Validation"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("RFECV - FEATURE SELECTION")
    logger.info("=" * 80)
    
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    
    rfecv = RFECV(
        estimator=base_model,
        step=config['model']['rfecv']['step'],
        cv=TimeSeriesSplit(n_splits=config['model']['rfecv']['cv']),
        scoring='r2',
        min_features_to_select=config['model']['rfecv']['min_features_to_select'],
        n_jobs=-1
    )
    
    logger.info(f"Starting RFECV with {X_train.shape[1]} ENVIRONMENTAL features...")
    rfecv.fit(X_train, y_train)
    
    selected_features = X_train.columns[rfecv.support_].tolist()
    logger.info(f"RFECV selected {len(selected_features)} features from {X_train.shape[1]}")
    
    # Get optimal CV score - use max from cv_results since n_features_ indexing can be tricky with step>1
    optimal_score = np.max(rfecv.cv_results_['mean_test_score'])
    logger.info(f"Optimal CV score (R²): {optimal_score:.4f}")
    
    return selected_features, rfecv

def optimize_hyperparameters(X_train, y_train, selected_features, config, logger):
    """Optimize hyperparameters using Optuna"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTUNA - HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    
    X_train_selected = X_train[selected_features]
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'random_state': 42
        }
        
        tscv = TimeSeriesSplit(n_splits=config['model']['optuna']['cv_folds'])
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train_selected):
            X_t, X_v = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            
            y_pred = model.predict(X_v)
            scores.append(r2_score(y_v, y_pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', study_name='envonly_optimization')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    logger.info(f"Starting Optuna optimization with {config['model']['optuna']['n_trials']} trials...")
    study.optimize(objective, n_trials=config['model']['optuna']['n_trials'], show_progress_bar=False)
    
    logger.info(f"Best trial R²: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    return study.best_params

def train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                     selected_features, best_params, config, logger):
    """Train final model with best parameters"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL MODEL TRAINING")
    logger.info("=" * 80)
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    model = xgb.XGBRegressor(**best_params)
    
    logger.info("Training final ENVIRONMENTAL-ONLY model...")
    model.fit(
        X_train_selected, y_train,
        eval_set=[(X_val_selected, y_val)],
        verbose=False
    )
    
    # Evaluate
    train_pred = model.predict(X_train_selected)
    val_pred = model.predict(X_val_selected)
    test_pred = model.predict(X_test_selected)
    
    logger.info("")
    logger.info("Performance Metrics (Environmental-Only):")
    logger.info(f"  Train R²: {r2_score(y_train, train_pred):.4f}")
    logger.info(f"  Val R²:   {r2_score(y_val, val_pred):.4f}")
    logger.info(f"  Test R²:  {r2_score(y_test, test_pred):.4f}")
    logger.info(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")
    logger.info(f"  Test MAE:  {mean_absolute_error(y_test, test_pred):.2f}")
    
    return model, train_pred, val_pred, test_pred

def save_results(model, selected_features, best_params, y_test, test_pred, config, illness_short_name, logger):
    """Save model and results"""
    results_dir = project_root / config['paths']['regression_results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_file = results_dir / f"model_{illness_short_name}_envonly_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {model_file}")
    
    # Save selected features
    features_file = results_dir / f"selected_features_{illness_short_name}_envonly_{timestamp}.csv"
    pd.DataFrame({'feature': selected_features}).to_csv(features_file, index=False)
    logger.info(f"Selected features saved: {features_file}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_file = results_dir / f"feature_importance_{illness_short_name}_envonly_{timestamp}.csv"
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"Feature importance saved: {importance_file}")
    
    # Save test predictions
    predictions_file = results_dir / f"test_predictions_{illness_short_name}_envonly_{timestamp}.csv"
    pd.DataFrame({
        'actual': y_test.values,
        'predicted': test_pred
    }).to_csv(predictions_file, index=False)
    logger.info(f"Test predictions saved: {predictions_file}")
    
    # Save metadata
    metadata = {
        'illness': config['illness_name'],
        'model_type': 'environmental_only',
        'timestamp': timestamp,
        'n_features_initial': len(model.feature_names_in_),
        'n_features_selected': len(selected_features),
        'test_r2': float(r2_score(y_test, test_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
        'test_mae': float(mean_absolute_error(y_test, test_pred)),
        'best_params': best_params,
        'excluded_features': 'Case_Count lags, Region, Year, Season, RegionName'
    }
    
    metadata_file = results_dir / f"metadata_{illness_short_name}_envonly_{timestamp}.yaml"
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f)
    logger.info(f"Metadata saved: {metadata_file}")

# This function will be called by specific illness scripts
def train_envonly_model(config_filename, illness_short_name):
    """Main training function for environmental-only models"""
    # Load configuration
    config_path = project_root / "configs" / config_filename
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logging(config, illness_short_name)
    logger.info("=" * 80)
    logger.info(f"ENVIRONMENTAL-ONLY TRAINING - {illness_short_name.upper()}")
    logger.info("=" * 80)
    logger.info(f"Illness: {config['illness_name']}")
    logger.info(f"Research Focus: Pure environmental exposure (NO autocorrelation)")
    logger.info("")
    
    # Load processed data
    data_file = project_root / config['paths']['illness_env_output_dir'] / f"{config['illness_name']}_illnessenv_envonly.csv"
    
    if not data_file.exists():
        logger.error(f"Processed data not found: {data_file}")
        logger.error(f"Please run preprocessing first: python scripts/preprocess_{illness_short_name}_envonly.py")
        return
    
    logger.info(f"Loading data: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Prepare data (ENVIRONMENTAL ONLY)
    X, y = prepare_data_envonly(df, config, logger)
    
    # Time series split
    train_ratio = config['model']['train_val_test_split']['train_ratio']
    val_ratio = config['model']['train_val_test_split']['val_ratio']
    
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    logger.info("")
    logger.info("Data Split:")
    logger.info(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
    logger.info(f"  Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
    logger.info(f"  Test:  {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Feature selection with RFECV
    selected_features, rfecv = perform_rfecv(X_train, y_train, config, logger)
    
    # Hyperparameter optimization with Optuna
    best_params = optimize_hyperparameters(X_train, y_train, selected_features, config, logger)
    
    # Train final model
    model, train_pred, val_pred, test_pred = train_final_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        selected_features, best_params, config, logger
    )
    
    # Save results
    save_results(model, selected_features, best_params, y_test, test_pred, config, illness_short_name, logger)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✓ {illness_short_name.upper()} ENVIRONMENTAL-ONLY TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("Next: Analyze lag periods from PURE environmental signal")

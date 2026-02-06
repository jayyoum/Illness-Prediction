#!/usr/bin/env python3
"""
Advanced Optimization with Optuna + LightGBM
Comparing XGBoost vs LightGBM with sophisticated hyperparameter tuning
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import logging
import json

project_root = Path(__file__).parent.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === CONFIGURATION ===
input_dir = project_root / "Processed Data/Illness & Environmental/Grouped/experimental"
feature_selection_dir = project_root / "results/feature_selection_lagged"
output_base_dir = project_root / "results/advanced_optimization"
os.makedirs(output_base_dir, exist_ok=True)

# Three illnesses
illnesses = [
    "Acute laryngopharyngitis",
    "Gastritis, unspecified",
    "Chronic rhinitis"
]

# CV Splitter
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# === OPTUNA OBJECTIVE FUNCTIONS ===

def objective_xgboost(trial, X, y):
    """Optuna objective for XGBoost with wider parameter space"""
    param = {
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    model = XGBRegressor(**param)
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='r2', n_jobs=-1)
    return scores.mean()

def objective_lightgbm(trial, X, y):
    """Optuna objective for LightGBM"""
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    model = LGBMRegressor(**param)
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='r2', n_jobs=-1)
    return scores.mean()

# === MAIN LOOP ===
print("="*80)
print("ADVANCED OPTIMIZATION - XGBoost + LightGBM with Optuna")
print("="*80)

all_results = []

for illness_name in illnesses:
    print(f"\n{'='*80}")
    print(f"ILLNESS: {illness_name}")
    print(f"{'='*80}")
    
    safe_name = illness_name.replace(", ", "_").replace(" ", "_")
    
    # Load data
    data_file = input_dir / f"{illness_name}_illnessenv_envonly.csv"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data: {len(df)} rows")
    
    # Load selected variables
    forward_file = feature_selection_dir / "forward_selection" / f"forward_selection_{safe_name}_lagged.csv"
    backward_file = feature_selection_dir / "backward_elimination" / f"backward_elimination_{safe_name}_lagged.csv"
    stepwise_file = feature_selection_dir / "stepwise_selection" / f"stepwise_selection_{safe_name}_lagged.csv"
    
    selected_sets = []
    
    if forward_file.exists():
        forward_df = pd.read_csv(forward_file)
        forward_vars = forward_df["Selected_Variables"].dropna().tolist()
        selected_sets.append(set(forward_vars))
    
    if backward_file.exists():
        backward_df = pd.read_csv(backward_file)
        backward_vars = backward_df["Selected_Variables"].dropna().tolist()
        selected_sets.append(set(backward_vars))
    
    if stepwise_file.exists():
        stepwise_df = pd.read_csv(stepwise_file)
        stepwise_vars = stepwise_df["Selected_Variables"].dropna().tolist()
        selected_sets.append(set(stepwise_vars))
    
    if not selected_sets:
        logger.error(f"No feature selection results found for {illness_name}")
        continue
    
    # Use intersection
    intersected_vars = list(set.intersection(*selected_sets))
    if not intersected_vars:
        logger.warning(f"No intersecting features. Using union.")
        intersected_vars = list(set.union(*selected_sets))
    
    available_vars = [v for v in intersected_vars if v in df.columns]
    logger.info(f"Using {len(available_vars)} features")
    
    # Prepare data
    X = df[available_vars]
    y = df["Case_Count"]
    
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    logger.info(f"Valid rows: {len(X)}")
    
    if len(X) < 50:
        logger.warning(f"Skipping {illness_name}: not enough rows")
        continue
    
    # === XGBoost Optimization ===
    print("\n--- XGBoost with Optuna (50 trials) ---")
    study_xgb = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f"xgb_{safe_name}"
    )
    
    study_xgb.optimize(
        lambda trial: objective_xgboost(trial, X, y),
        n_trials=50,
        show_progress_bar=True
    )
    
    best_params_xgb = study_xgb.best_params
    best_cv_r2_xgb = study_xgb.best_value
    
    logger.info(f"XGBoost Best CV R²: {best_cv_r2_xgb:.4f}")
    logger.info(f"XGBoost Best params: {best_params_xgb}")
    
    # Train final XGBoost model
    final_xgb = XGBRegressor(
        objective='reg:squarederror',
        verbosity=0,
        random_state=42,
        **best_params_xgb
    )
    final_xgb.fit(X, y)
    y_pred_xgb = final_xgb.predict(X)
    
    xgb_r2 = r2_score(y, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y, y_pred_xgb))
    xgb_mae = mean_absolute_error(y, y_pred_xgb)
    
    logger.info(f"XGBoost Final R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}")
    
    # === LightGBM Optimization ===
    print("\n--- LightGBM with Optuna (50 trials) ---")
    study_lgb = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f"lgb_{safe_name}"
    )
    
    study_lgb.optimize(
        lambda trial: objective_lightgbm(trial, X, y),
        n_trials=50,
        show_progress_bar=True
    )
    
    best_params_lgb = study_lgb.best_params
    best_cv_r2_lgb = study_lgb.best_value
    
    logger.info(f"LightGBM Best CV R²: {best_cv_r2_lgb:.4f}")
    logger.info(f"LightGBM Best params: {best_params_lgb}")
    
    # Train final LightGBM model
    final_lgb = LGBMRegressor(
        objective='regression',
        metric='rmse',
        verbosity=-1,
        random_state=42,
        **best_params_lgb
    )
    final_lgb.fit(X, y)
    y_pred_lgb = final_lgb.predict(X)
    
    lgb_r2 = r2_score(y, y_pred_lgb)
    lgb_rmse = np.sqrt(mean_squared_error(y, y_pred_lgb))
    lgb_mae = mean_absolute_error(y, y_pred_lgb)
    
    logger.info(f"LightGBM Final R²: {lgb_r2:.4f}, RMSE: {lgb_rmse:.2f}, MAE: {lgb_mae:.2f}")
    
    # === Choose Best Model ===
    if best_cv_r2_lgb > best_cv_r2_xgb:
        best_model_name = "LightGBM"
        best_model = final_lgb
        best_cv_r2 = best_cv_r2_lgb
        best_params = best_params_lgb
        best_r2 = lgb_r2
        best_rmse = lgb_rmse
        best_mae = lgb_mae
        y_pred_best = y_pred_lgb
        importances = final_lgb.feature_importances_
    else:
        best_model_name = "XGBoost"
        best_model = final_xgb
        best_cv_r2 = best_cv_r2_xgb
        best_params = best_params_xgb
        best_r2 = xgb_r2
        best_rmse = xgb_rmse
        best_mae = xgb_mae
        y_pred_best = y_pred_xgb
        importances = final_xgb.feature_importances_
    
    logger.info(f"\n✓ Best Model: {best_model_name}")
    logger.info(f"✓ CV R²: {best_cv_r2:.4f}, Full R²: {best_r2:.4f}, RMSE: {best_rmse:.2f}")
    
    # Save results
    output_dir = output_base_dir / safe_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison
    comparison = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM'],
        'CV_R2': [best_cv_r2_xgb, best_cv_r2_lgb],
        'Full_R2': [xgb_r2, lgb_r2],
        'RMSE': [xgb_rmse, lgb_rmse],
        'MAE': [xgb_mae, lgb_mae]
    })
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Save best model summary
    result = {
        "IllnessName": illness_name,
        "Best_Model": best_model_name,
        "Num_Features": len(available_vars),
        "CV_R2": round(best_cv_r2, 4),
        "Full_R2": round(best_r2, 4),
        "RMSE": round(best_rmse, 2),
        "MAE": round(best_mae, 2),
        **{f"Param_{k}": v for k, v in best_params.items()}
    }
    
    for var, imp in zip(available_vars, importances):
        result[f"Importance_{var}"] = round(imp, 4)
    
    output_df = pd.DataFrame([result])
    output_df.to_csv(output_dir / "best_model_summary.csv", index=False)
    
    # Save feature importance
    importance_df = pd.DataFrame({
        "Feature": available_vars,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        "Actual": y.values,
        f"Predicted_{best_model_name}": y_pred_best,
        "Predicted_XGBoost": y_pred_xgb,
        "Predicted_LightGBM": y_pred_lgb
    })
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Collect summary
    all_results.append({
        'Illness': illness_name,
        'Best_Model': best_model_name,
        'CV_R2': best_cv_r2,
        'Full_R2': best_r2,
        'RMSE': best_rmse,
        'MAE': best_mae,
        'XGBoost_CV_R2': best_cv_r2_xgb,
        'LightGBM_CV_R2': best_cv_r2_lgb,
        'Improvement_vs_GridSearch': None  # Will calculate later
    })

# Save overall summary
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(output_base_dir / "ADVANCED_OPTIMIZATION_SUMMARY.csv", index=False)

print("\n" + "="*80)
print("ADVANCED OPTIMIZATION COMPLETE")
print("="*80)
print(f"\nResults saved to: {output_base_dir}")

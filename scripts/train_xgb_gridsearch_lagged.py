#!/usr/bin/env python3
"""
XGBoost GridSearch with Selected Lagged Variables
Imitates original XGB_GridSearch.py methodology
Uses features selected from forward/backward/stepwise selection
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
import json
import logging

project_root = Path(__file__).parent.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
input_dir = project_root / "Processed Data/Illness & Environmental/Grouped/experimental"
feature_selection_dir = project_root / "results/feature_selection_lagged"
output_base_dir = project_root / "results/xgb_gridsearch_lagged"
os.makedirs(output_base_dir, exist_ok=True)

# Three illnesses to process
illnesses = [
    "Acute laryngopharyngitis",
    "Gastritis, unspecified",
    "Chronic rhinitis"
]

# === HYPERPARAMETER GRID (from original XGB_GridSearch.py) ===
param_grid = {
    'n_estimators': [100, 120, 150],
    'learning_rate': [0.1, 0.15],
    'max_depth': [4, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# === CV SPLITTER (matches original) ===
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# === MAIN LOOP ===
print("="*80)
print("XGBoost GridSearch - Lagged Environmental Variables")
print("="*80)
print("\nUsing features from: Forward + Backward + Stepwise Selection")
print("Parameter Grid:", json.dumps(param_grid, indent=2))
print()

for illness_name in illnesses:
    print(f"\n{'='*80}")
    print(f"ILLNESS: {illness_name}")
    print(f"{'='*80}")
    
    safe_name = illness_name.replace(", ", "_").replace(" ", "_")
    
    # Load environmental-only data
    data_file = input_dir / f"{illness_name}_illnessenv_envonly.csv"
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data: {len(df)} rows")
    
    # Load selected variables from each method
    forward_file = feature_selection_dir / "forward_selection" / f"forward_selection_{safe_name}_lagged.csv"
    backward_file = feature_selection_dir / "backward_elimination" / f"backward_elimination_{safe_name}_lagged.csv"
    stepwise_file = feature_selection_dir / "stepwise_selection" / f"stepwise_selection_{safe_name}_lagged.csv"
    
    selected_sets = []
    
    if forward_file.exists():
        forward_df = pd.read_csv(forward_file)
        forward_vars = forward_df["Selected_Variables"].dropna().tolist()
        selected_sets.append(set(forward_vars))
        logger.info(f"Forward selection: {len(forward_vars)} features")
    
    if backward_file.exists():
        backward_df = pd.read_csv(backward_file)
        backward_vars = backward_df["Selected_Variables"].dropna().tolist()
        selected_sets.append(set(backward_vars))
        logger.info(f"Backward elimination: {len(backward_vars)} features")
    
    if stepwise_file.exists():
        stepwise_df = pd.read_csv(stepwise_file)
        stepwise_vars = stepwise_df["Selected_Variables"].dropna().tolist()
        selected_sets.append(set(stepwise_vars))
        logger.info(f"Stepwise selection: {len(stepwise_vars)} features")
    
    if not selected_sets:
        logger.error(f"No feature selection results found for {illness_name}")
        continue
    
    # Find intersection of selected features (like original methodology)
    intersected_vars = list(set.intersection(*selected_sets))
    
    if not intersected_vars:
        logger.warning(f"No intersecting features found. Using union instead.")
        intersected_vars = list(set.union(*selected_sets))
    
    logger.info(f"Using {len(intersected_vars)} intersected features")
    logger.info(f"Features: {intersected_vars}")
    
    # Check which features exist in the dataframe
    available_vars = [v for v in intersected_vars if v in df.columns]
    
    if not available_vars:
        logger.error(f"None of the selected features exist in the dataframe!")
        continue
    
    if len(available_vars) < len(intersected_vars):
        missing = set(intersected_vars) - set(available_vars)
        logger.warning(f"Missing {len(missing)} features: {missing}")
    
    # Prepare data
    X = df[available_vars]
    y = df["Case_Count"]
    
    # Drop rows with NaNs
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    logger.info(f"Valid rows after removing NaNs: {len(X)}")
    
    if len(X) < 50:
        logger.warning(f"Skipping {illness_name}: not enough valid rows")
        continue
    
    # Initialize model (matching original)
    model = XGBRegressor(
        objective='reg:squarederror',
        verbosity=0,
        random_state=42
    )
    
    # GridSearch
    logger.info("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(r2_score),
        cv=cv_splitter,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_r2_cv = grid_search.best_score_
    best_params = grid_search.best_params_
    
    logger.info(f"Best CV R²: {best_r2_cv:.4f}")
    logger.info(f"Best params: {best_params}")
    
    # Refit on full data to get importances and final metrics
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    
    # Calculate metrics
    final_r2 = r2_score(y, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y, y_pred))
    final_mae = mean_absolute_error(y, y_pred)
    
    importances = best_model.feature_importances_
    
    logger.info(f"Final R² (full data): {final_r2:.4f}")
    logger.info(f"Final RMSE: {final_rmse:.2f}")
    logger.info(f"Final MAE: {final_mae:.2f}")
    
    # Prepare results
    result = {
        "IllnessName": illness_name,
        "Num_Features": len(available_vars),
        "Best_R2_CV": round(best_r2_cv, 4),
        "Final_R2": round(final_r2, 4),
        "Final_RMSE": round(final_rmse, 2),
        "Final_MAE": round(final_mae, 2),
        **{f"Param_{k}": v for k, v in best_params.items()}
    }
    
    for var, imp in zip(available_vars, importances):
        result[f"Importance_{var}"] = round(imp, 4)
    
    # Save results
    output_dir = output_base_dir / safe_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    output_df = pd.DataFrame([result])
    summary_file = output_dir / f"xgb_gridsearch_{safe_name}_summary.csv"
    output_df.to_csv(summary_file, index=False)
    logger.info(f"✓ Saved summary: {summary_file}")
    
    # Save feature importance separately
    importance_df = pd.DataFrame({
        "Feature": available_vars,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    importance_file = output_dir / f"feature_importance_{safe_name}.csv"
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"✓ Saved feature importance: {importance_file}")
    
    # Save predictions for visualization
    predictions_df = pd.DataFrame({
        "Actual": y.values,
        "Predicted": y_pred
    })
    predictions_file = output_dir / f"predictions_{safe_name}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"✓ Saved predictions: {predictions_file}")

print("\n" + "="*80)
print("XGBoost GridSearch COMPLETE")
print("="*80)

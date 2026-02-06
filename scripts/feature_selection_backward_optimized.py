#!/usr/bin/env python3
"""
Backward Elimination with Lagged Environmental Variables (Optimized)
Pre-filters to key environmental variables to make elimination practical
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import statsmodels.api as sm

project_root = Path(__file__).parent.parent

# === CONFIGURATION ===
input_dir = project_root / "Processed Data/Illness & Environmental/Grouped/experimental"
output_base_dir = project_root / "results/feature_selection_lagged/backward_elimination"
os.makedirs(output_base_dir, exist_ok=True)

# Three illnesses to process
illnesses = [
    "Acute laryngopharyngitis",
    "Gastritis, unspecified",
    "Chronic rhinitis"
]

# Core environmental variables (matching forward selection)
CORE_ENV_VARS = [
    'AvgTemp', 'MinTemp', 'MaxTemp',
    'PM10', 'PM25',
    'AvgHumidity', 'MinHumidity',
    'Rainfall', 'CloudCover',
    'AvgWindSpeed', 'MaxWindSpeed',
    'SO2', 'CO', 'O3', 'NO2',
    'SunshineHours',
    'AvgVaporPressure',
    'AvgLocalPressure'
]

# === HELPER FUNCTIONS ===
def filter_to_core_features(all_columns):
    """Filter to core environmental variables + their lags"""
    core_features = []
    
    for col in all_columns:
        if any(col == var for var in CORE_ENV_VARS):
            core_features.append(col)
        elif any(col.startswith(f"{var}_lag_") for var in CORE_ENV_VARS):
            if "_rolling_" not in col:
                core_features.append(col)
    
    return core_features

def backward_elimination(X, y, threshold=0.05, verbose=True):
    """Backward elimination using OLS and p-values"""
    features = list(X.columns)
    pmax = 1
    
    print(f"Starting backward elimination with {len(features)} features...")
    print(f"P-value threshold: {threshold}")
    
    iteration = 0
    while pmax > threshold and len(features) > 1:
        iteration += 1
        try:
            X_with_const = sm.add_constant(X[features])
            model = sm.OLS(y, X_with_const).fit()
            p_values = model.pvalues.iloc[1:]  # skip intercept
            pmax = p_values.max()
            worst_feature = p_values.idxmax()
            
            if pmax > threshold:
                if verbose:
                    print(f"  [{iteration}] 🗑️  Removing: {worst_feature} (p = {pmax:.4f})")
                features.remove(worst_feature)
            else:
                if verbose:
                    print("  ✅ All remaining features are significant.")
                break
        except Exception as e:
            print(f"  ⚠️ Error at iteration {iteration}: {e}")
            break
    
    return features, model.rsquared if 'model' in locals() else 0

# === MAIN LOOP ===
print("="*80)
print("BACKWARD ELIMINATION - OPTIMIZED")
print("="*80)
print(f"\nFocusing on {len(CORE_ENV_VARS)} core environmental variables + their lags\n")

for illness_name in illnesses:
    print(f"\n{'='*80}")
    print(f"ILLNESS: {illness_name}")
    print(f"{'='*80}")
    
    # Load environmental-only data
    data_file = input_dir / f"{illness_name}_illnessenv_envonly.csv"
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Prepare features
    excluded_cols = {
        "Case_Count", "DateTime", "IllnessName", "RegionName",
        "Season", "Year", "Year_x", "Year_y", "Region", "Month"
    }
    
    case_count_features = [col for col in df.columns if 'Case_Count' in col or 'CaseCount' in col]
    all_exclusions = excluded_cols.union(set(case_count_features))
    
    env_features = [col for col in df.columns
                    if col not in all_exclusions
                    and pd.api.types.is_numeric_dtype(df[col])]
    
    # Filter to core features
    core_features = filter_to_core_features(env_features)
    
    print(f"Total environmental features: {len(env_features)}")
    print(f"Filtered to core features: {len(core_features)}")
    
    X = df[core_features]
    y = df["Case_Count"]
    
    # Drop rows with NaNs
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    print(f"Valid rows: {len(X)}")
    
    if len(X) < 10:
        print(f"⚠️ Skipping {illness_name}: Not enough valid rows")
        continue
    
    # Run backward elimination
    selected_vars, r2 = backward_elimination(X, y, threshold=0.05, verbose=True)
    
    if not selected_vars:
        print(f"⚠️ No variables selected for {illness_name}")
        continue
    
    print(f"\n✓ Selected {len(selected_vars)} features")
    print(f"✓ Final R² = {r2:.4f}")
    
    # Fit final model to get coefficients
    X_final = sm.add_constant(X[selected_vars])
    model_final = sm.OLS(y, X_final).fit()
    coefs = model_final.params[1:].values
    
    # Save results
    results = pd.DataFrame({
        "Selected_Variables": selected_vars,
        "Coefficient": coefs,
        "R_squared": [r2] * len(selected_vars)
    })
    
    safe_name = illness_name.replace(", ", "_").replace(" ", "_")
    output_file = output_base_dir / f"backward_elimination_{safe_name}_lagged.csv"
    results.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("BACKWARD ELIMINATION COMPLETE")
print("="*80)

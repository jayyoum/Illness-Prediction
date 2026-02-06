#!/usr/bin/env python3
"""
Backward Elimination with Lagged Environmental Variables
Imitates original backward_elimination.py but uses data with 1-14 day lags
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

# === HELPER FUNCTION ===
def backward_elimination(X, y, threshold=0.05, verbose=True):
    """
    Backward elimination using OLS and p-values
    (Imitates original methodology)
    """
    features = list(X.columns)
    pmax = 1
    
    print(f"Starting backward elimination with {len(features)} features...")
    print(f"P-value threshold: {threshold}")
    
    while pmax > threshold and len(features) > 1:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        p_values = model.pvalues.iloc[1:]  # skip intercept
        pmax = p_values.max()
        worst_feature = p_values.idxmax()
        
        if pmax > threshold:
            if verbose:
                print(f"  🗑️  Removing: {worst_feature} (p = {pmax:.4f})")
            features.remove(worst_feature)
        else:
            if verbose:
                print("  ✅ All remaining features are significant.")
            break
    
    return features, model.rsquared

# === MAIN LOOP ===
print("="*80)
print("BACKWARD ELIMINATION - LAGGED ENVIRONMENTAL VARIABLES")
print("="*80)
print("\nExcluding: Case_Count features, Region, Year, Season, RegionName")
print("Including: Environmental variables + lags (1-14 days)\n")

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
    
    # Prepare features (exclude non-environmental columns)
    excluded_cols = {
        "Case_Count", "DateTime", "IllnessName", "RegionName",
        "Season", "Year", "Year_x", "Year_y", "Region"
    }
    
    # Exclude all Case_Count-related features
    case_count_features = [col for col in df.columns if 'Case_Count' in col or 'CaseCount' in col]
    all_exclusions = excluded_cols.union(set(case_count_features))
    
    env_features = [col for col in df.columns
                    if col not in all_exclusions
                    and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[env_features]
    y = df["Case_Count"]
    
    # Drop rows with NaNs
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    print(f"Features for selection: {len(env_features)}")
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

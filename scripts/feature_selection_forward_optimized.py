#!/usr/bin/env python3
"""
Forward Selection with Lagged Environmental Variables (Optimized)
Pre-filters to key environmental variables to make greedy search practical
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

project_root = Path(__file__).parent.parent

# === CONFIGURATION ===
input_dir = project_root / "Processed Data/Illness & Environmental/Grouped/experimental"
output_base_dir = project_root / "results/feature_selection_lagged/forward_selection"
os.makedirs(output_base_dir, exist_ok=True)

# Three illnesses to process
illnesses = [
    "Acute laryngopharyngitis",
    "Gastritis, unspecified",
    "Chronic rhinitis"
]

# Core environmental variables to focus on (matching your original research)
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
        # Include base core variables
        if any(col == var for var in CORE_ENV_VARS):
            core_features.append(col)
        # Include lags of core variables (but not rolling means)
        elif any(col.startswith(f"{var}_lag_") for var in CORE_ENV_VARS):
            # Skip rolling means of lags
            if "_rolling_" not in col:
                core_features.append(col)
    
    return core_features

def adjusted_r2(r2, n, k):
    """Calculate adjusted R²"""
    if k >= n - 1:
        return -np.inf
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

def forward_selection(X, y, verbose=True):
    """Forward selection using LinearRegression and adjusted R²"""
    remaining = list(X.columns)
    selected = []
    current_score = -np.inf
    n = len(y)
    
    print(f"Starting forward selection with {len(remaining)} features...")
    
    iteration = 0
    while remaining:
        iteration += 1
        scores = []
        
        for candidate in remaining:
            trial_features = selected + [candidate]
            try:
                model = LinearRegression().fit(X[trial_features], y)
                r2 = r2_score(y, model.predict(X[trial_features]))
                adj_r2 = adjusted_r2(r2, n, len(trial_features))
                scores.append((adj_r2, candidate))
            except:
                continue
        
        if not scores:
            break
        
        scores.sort(reverse=True)
        best_new_score, best_candidate = scores[0]
        
        if best_new_score > current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            if verbose:
                print(f"  [{iteration}] ✅ Added: {best_candidate} | Adj R² = {current_score:.4f}")
        else:
            if verbose:
                print("  ⛔ No improvement. Stopping.")
            break
    
    return selected, current_score

# === MAIN LOOP ===
print("="*80)
print("FORWARD SELECTION - OPTIMIZED")
print("="*80)
print(f"\nFocusing on {len(CORE_ENV_VARS)} core environmental variables + their lags")
print("Core variables:", ", ".join(CORE_ENV_VARS[:5]), "...\n")

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
        "Season", "Year", "Year_x", "Year_y", "Region", "Month"
    }
    
    # Exclude all Case_Count-related features
    case_count_features = [col for col in df.columns if 'Case_Count' in col or 'CaseCount' in col]
    all_exclusions = excluded_cols.union(set(case_count_features))
    
    env_features = [col for col in df.columns
                    if col not in all_exclusions
                    and pd.api.types.is_numeric_dtype(df[col])]
    
    # Filter to core features + their lags
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
    
    # Run forward selection
    selected_vars, final_score = forward_selection(X, y, verbose=True)
    
    if not selected_vars:
        print(f"⚠️ No variables selected for {illness_name}")
        continue
    
    print(f"\n✓ Selected {len(selected_vars)} features")
    print(f"✓ Final Adjusted R² = {final_score:.4f}")
    
    # Fit final model to get coefficients
    model = LinearRegression().fit(X[selected_vars], y)
    coefs = model.coef_
    r2 = r2_score(y, model.predict(X[selected_vars]))
    
    # Save results
    results = pd.DataFrame({
        "Selected_Variables": selected_vars,
        "Coefficient": coefs,
        "R_squared": [r2] * len(selected_vars)
    })
    
    safe_name = illness_name.replace(", ", "_").replace(" ", "_")
    output_file = output_base_dir / f"forward_selection_{safe_name}_lagged.csv"
    results.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("FORWARD SELECTION COMPLETE")
print("="*80)

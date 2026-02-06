#!/usr/bin/env python3
"""
Stepwise Selection with Lagged Environmental Variables
Imitates original stepwise_selection.py but uses data with 1-14 day lags
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
output_base_dir = project_root / "results/feature_selection_lagged/stepwise_selection"
os.makedirs(output_base_dir, exist_ok=True)

# Three illnesses to process
illnesses = [
    "Acute laryngopharyngitis",
    "Gastritis, unspecified",
    "Chronic rhinitis"
]

# === HELPER FUNCTIONS ===
def adjusted_r2(r2, n, k):
    """Calculate adjusted R²"""
    if k >= n - 1:
        return -np.inf
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

def stepwise_selection(X, y, verbose=True):
    """
    Stepwise selection: combination of forward and backward steps
    (Imitates original methodology)
    """
    remaining = list(X.columns)
    selected = []
    n = len(y)
    current_score = -np.inf
    
    print(f"Starting stepwise selection with {len(remaining)} features...")
    
    iteration = 0
    while True:
        changed = False
        iteration += 1
        
        if verbose and iteration > 1:
            print(f"\n  Iteration {iteration}")
        
        # FORWARD STEP: Try adding features
        scores_with_candidates = []
        for candidate in remaining:
            if candidate in selected:
                continue
            trial_features = selected + [candidate]
            try:
                model = LinearRegression().fit(X[trial_features], y)
                r2 = r2_score(y, model.predict(X[trial_features]))
                adj_r2 = adjusted_r2(r2, n, len(trial_features))
                scores_with_candidates.append((adj_r2, candidate))
            except:
                continue
        
        if scores_with_candidates:
            best_new_score, best_candidate = max(scores_with_candidates)
            if best_new_score > current_score:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                current_score = best_new_score
                changed = True
                if verbose:
                    print(f"  ✅ Added: {best_candidate} | Adjusted R² = {current_score:.4f}")
        
        # BACKWARD STEP: Try removing features
        if len(selected) > 1:
            scores_with_candidates = []
            for candidate in selected:
                trial_features = [f for f in selected if f != candidate]
                try:
                    model = LinearRegression().fit(X[trial_features], y)
                    r2 = r2_score(y, model.predict(X[trial_features]))
                    adj_r2 = adjusted_r2(r2, n, len(trial_features))
                    scores_with_candidates.append((adj_r2, candidate))
                except:
                    continue
            
            if scores_with_candidates:
                best_new_score, worst_candidate = max(scores_with_candidates)
                if best_new_score > current_score:
                    selected.remove(worst_candidate)
                    current_score = best_new_score
                    changed = True
                    if verbose:
                        print(f"  ❌ Removed: {worst_candidate} | Adjusted R² = {current_score:.4f}")
        
        if not changed:
            if verbose:
                print("  ⛔ No improvement. Stopping.")
            break
    
    return selected, current_score

# === MAIN LOOP ===
print("="*80)
print("STEPWISE SELECTION - LAGGED ENVIRONMENTAL VARIABLES")
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
    
    # Run stepwise selection
    selected_vars, final_score = stepwise_selection(X, y, verbose=True)
    
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
    output_file = output_base_dir / f"stepwise_selection_{safe_name}_lagged.csv"
    results.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("STEPWISE SELECTION COMPLETE")
print("="*80)

#!/usr/bin/env python3
"""
Master Script: Run complete lagged variable pipeline
1. Forward Selection
2. Backward Elimination  
3. Stepwise Selection
4. XGBoost GridSearch with selected features
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
scripts_dir = project_root / "scripts"

def run_script(script_name):
    """Run a Python script and check for errors"""
    script_path = scripts_dir / script_name
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ {script_name} completed successfully")

# Main pipeline
print("\n" + "="*80)
print("LAGGED VARIABLE GRIDSEARCH PIPELINE")
print("="*80)
print("\nThis pipeline will:")
print("1. Run Forward Selection")
print("2. Run Backward Elimination")
print("3. Run Stepwise Selection")
print("4. Train XGBoost with GridSearch on selected features")
print("\nFor all three illnesses with 1-14 day lagged variables")
print("="*80 + "\n")

input("Press Enter to start...")

# Step 1: Forward Selection (Optimized)
run_script("feature_selection_forward_optimized.py")

# Step 2: Backward Elimination (Optimized)
run_script("feature_selection_backward_optimized.py")

# Step 3: Stepwise Selection (Optimized)
run_script("feature_selection_stepwise_optimized.py")

# Step 4: XGBoost GridSearch
run_script("train_xgb_gridsearch_lagged.py")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nResults saved to:")
print(f"  - Feature selection: {project_root}/results/feature_selection_lagged/")
print(f"  - XGBoost models: {project_root}/results/xgb_gridsearch_lagged/")

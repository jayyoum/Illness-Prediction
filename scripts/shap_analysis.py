#!/usr/bin/env python3
"""
SHAP Analysis for Illness Prediction Models
Analyzes feature contributions and directional effects (increase/decrease illness)
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

# === CONFIGURATION ===
input_dir = project_root / "Processed Data/Illness & Environmental/Grouped/experimental"
output_base_dir = project_root / "results/advanced_optimization"
shap_output_dir = project_root / "results/shap_analysis"
os.makedirs(shap_output_dir, exist_ok=True)

# Three illnesses
illnesses = [
    "Acute laryngopharyngitis",
    "Gastritis, unspecified",
    "Chronic rhinitis"
]

# Best hyperparameters from Optuna (from previous optimization)
best_params = {
    "Acute laryngopharyngitis": {
        'n_estimators': 479,
        'learning_rate': 0.061819715682712134,
        'max_depth': 10,
        'min_child_weight': 10,
        'subsample': 0.8034876554172365,
        'colsample_bytree': 0.7215959033697187,
        'gamma': 1.9094491224439119,
        'reg_alpha': 5.528038932096746,
        'reg_lambda': 7.822506485240053,
        'random_state': 42
    },
    "Gastritis, unspecified": {
        'n_estimators': 481,
        'learning_rate': 0.04905735969916903,
        'max_depth': 10,
        'min_child_weight': 9,
        'subsample': 0.7488661740154809,
        'colsample_bytree': 0.6555722520571796,
        'gamma': 2.101044992551405,
        'reg_alpha': 9.089776098629683,
        'reg_lambda': 9.955562368873672,
        'random_state': 42
    },
    "Chronic rhinitis": {
        'n_estimators': 364,
        'learning_rate': 0.04133458606012691,
        'max_depth': 9,
        'min_child_weight': 8,
        'subsample': 0.9408817090480656,
        'colsample_bytree': 0.9973639668867567,
        'gamma': 1.8278028009699595,
        'reg_alpha': 8.08628877632956,
        'reg_lambda': 4.5842816965826665,
        'random_state': 42
    }
}

def safe_filename(illness_name):
    """Convert illness name to safe filename"""
    return illness_name.replace(", ", "_").replace(" ", "_")

def load_data_and_features(illness_name):
    """Load dataset and selected features"""
    safe_name = safe_filename(illness_name)
    
    # Load the comprehensive lag dataset
    # Try different file patterns
    possible_files = [
        input_dir / f"{illness_name}_illnessenv.csv",
        input_dir / f"merged_data_{safe_name}_lag0_comprehensive_ts.csv",
        input_dir / f"{safe_name}_enviro_illness_experimental.csv"
    ]
    
    data_file = None
    for f in possible_files:
        if f.exists():
            data_file = f
            break
    
    if data_file is None:
        raise FileNotFoundError(f"Could not find data file for {illness_name}")
    
    df = pd.read_csv(data_file)
    
    # Load selected features (intersection of forward, backward, stepwise)
    feature_files = {
        'forward': project_root / "results/feature_selection_lagged/forward_selection" / f"forward_selection_{safe_name}_lagged.csv",
        'backward': project_root / "results/feature_selection_lagged/backward_elimination" / f"backward_elimination_{safe_name}_lagged.csv",
        'stepwise': project_root / "results/feature_selection_lagged/stepwise_selection" / f"stepwise_selection_{safe_name}_lagged.csv"
    }
    
    selected_features_sets = []
    for method_name, feat_file in feature_files.items():
        if feat_file.exists():
            feat_df = pd.read_csv(feat_file)
            # Get feature column (might be 'Feature' or 'feature' or first column)
            feature_col = feat_df.columns[0] if 'Feature' not in feat_df.columns and 'feature' not in feat_df.columns else ('Feature' if 'Feature' in feat_df.columns else 'feature')
            selected_features_sets.append(set(feat_df[feature_col].tolist()))
        else:
            print(f"  Warning: {method_name} feature file not found")
    
    # Intersection of all three methods
    if len(selected_features_sets) == 3:
        selected_features = list(selected_features_sets[0].intersection(
            selected_features_sets[1], selected_features_sets[2]))
    else:
        raise ValueError(f"Not all feature selection files found for {illness_name}. Found {len(selected_features_sets)}/3")
    
    # Prepare X and y
    X = df[selected_features].copy()
    
    # Try different case count column names
    if 'Case_Count' in df.columns:
        y = df['Case_Count'].copy()
    elif 'CaseCount' in df.columns:
        y = df['CaseCount'].copy()
    elif 'case_count' in df.columns:
        y = df['case_count'].copy()
    else:
        raise KeyError(f"Could not find case count column in {data_file}")
    
    return X, y, selected_features

def train_and_save_model(illness_name, X, y, params):
    """Train model with best parameters and save it"""
    safe_name = safe_filename(illness_name)
    output_dir = output_base_dir / safe_name
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining final model for {illness_name}...")
    
    # Train on full dataset
    model = XGBRegressor(**params)
    model.fit(X, y)
    
    # Save model
    model_path = output_dir / "best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved: {model_path}")
    
    # Calculate metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"  R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    return model, r2, rmse, mae

def compute_shap_values(model, X, illness_name):
    """Compute SHAP values for the model"""
    safe_name = safe_filename(illness_name)
    
    print(f"\nComputing SHAP values for {illness_name}...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values on a sample (for computational efficiency)
    # Use max 1000 samples or all if dataset is smaller
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    shap_values = explainer.shap_values(X_sample)
    
    print(f"✓ SHAP values computed for {sample_size} samples")
    
    return explainer, shap_values, X_sample

def analyze_shap_results(shap_values, X_sample, feature_names, illness_name):
    """Analyze SHAP values and extract insights"""
    safe_name = safe_filename(illness_name)
    shap_dir = shap_output_dir / safe_name
    os.makedirs(shap_dir, exist_ok=True)
    
    print(f"\nAnalyzing SHAP results for {illness_name}...")
    
    # 1. Mean absolute SHAP values (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 2. Mean SHAP values (directional effect)
    mean_shap = shap_values.mean(axis=0)
    
    # 3. Create comprehensive dataframe
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Mean_SHAP': mean_shap,
        'Direction': ['Increase' if x > 0 else 'Decrease' for x in mean_shap],
        'Abs_Mean_SHAP': np.abs(mean_shap)
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    # 4. Compute feature value correlations
    feature_correlations = []
    for i, feature in enumerate(feature_names):
        corr = np.corrcoef(X_sample[feature].values, shap_values[:, i])[0, 1]
        feature_correlations.append(corr)
    
    shap_df['Feature_SHAP_Correlation'] = [feature_correlations[list(feature_names).index(f)] 
                                            for f in shap_df['Feature']]
    
    # Save results
    shap_df.to_csv(shap_dir / "shap_values_summary.csv", index=False)
    print(f"✓ Saved: shap_values_summary.csv")
    
    # 5. Detailed SHAP values per sample (top 30 features only for efficiency)
    top_features = shap_df.head(30)['Feature'].tolist()
    top_indices = [list(feature_names).index(f) for f in top_features]
    
    detailed_shap = pd.DataFrame(
        shap_values[:, top_indices],
        columns=top_features,
        index=X_sample.index
    )
    detailed_shap.to_csv(shap_dir / "shap_values_detailed_top30.csv")
    print(f"✓ Saved: shap_values_detailed_top30.csv (top 30 features)")
    
    # 6. Categorize features
    def categorize_feature(feature):
        if any(pol in feature for pol in ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']):
            return 'Air Quality'
        elif 'Temp' in feature:
            return 'Temperature'
        elif 'Humidity' in feature or 'Vapor' in feature:
            return 'Humidity'
        elif 'Wind' in feature:
            return 'Wind'
        elif 'Pressure' in feature:
            return 'Pressure'
        elif 'Rain' in feature or 'Cloud' in feature or 'Sunshine' in feature:
            return 'Weather'
        else:
            return 'Other'
    
    shap_df['Category'] = shap_df['Feature'].apply(categorize_feature)
    shap_df['Is_Lag'] = shap_df['Feature'].str.contains('_lag_')
    
    # Category summary
    category_summary = shap_df.groupby('Category').agg({
        'Mean_Abs_SHAP': 'sum',
        'Mean_SHAP': 'mean',
        'Feature': 'count'
    }).rename(columns={'Feature': 'Count'}).sort_values('Mean_Abs_SHAP', ascending=False)
    
    category_summary.to_csv(shap_dir / "shap_category_summary.csv")
    print(f"✓ Saved: shap_category_summary.csv")
    
    # Lag vs Base summary
    lag_summary = shap_df.groupby('Is_Lag').agg({
        'Mean_Abs_SHAP': 'sum',
        'Mean_SHAP': 'mean',
        'Feature': 'count'
    }).rename(columns={'Feature': 'Count'})
    lag_summary.index = ['Base Features', 'Lag Features']
    lag_summary.to_csv(shap_dir / "shap_lag_vs_base.csv")
    print(f"✓ Saved: shap_lag_vs_base.csv")
    
    # Directional summary
    direction_summary = shap_df.groupby('Direction').agg({
        'Mean_Abs_SHAP': 'sum',
        'Feature': 'count'
    }).rename(columns={'Feature': 'Count'})
    direction_summary.to_csv(shap_dir / "shap_direction_summary.csv")
    print(f"✓ Saved: shap_direction_summary.csv")
    
    return shap_df, category_summary, lag_summary, direction_summary

def save_shap_data_for_r(explainer, shap_values, X_sample, illness_name):
    """Save SHAP base values and expected value for R plotting"""
    safe_name = safe_filename(illness_name)
    shap_dir = shap_output_dir / safe_name
    
    # Save expected value (base value)
    expected_value = explainer.expected_value
    with open(shap_dir / "shap_expected_value.txt", 'w') as f:
        f.write(str(expected_value))
    
    # Save feature values for reference
    X_sample.to_csv(shap_dir / "shap_feature_values.csv", index=False)
    
    print(f"✓ Saved SHAP base data for R plotting")

def main():
    print("=" * 80)
    print("SHAP ANALYSIS FOR ILLNESS PREDICTION MODELS")
    print("=" * 80)
    
    all_results = []
    
    for illness_name in illnesses:
        print(f"\n{'=' * 80}")
        print(f"Processing: {illness_name}")
        print("=" * 80)
        
        try:
            # Load data
            X, y, selected_features = load_data_and_features(illness_name)
            print(f"✓ Loaded data: {len(X)} samples, {len(selected_features)} features")
            
            # Train and save model
            params = best_params[illness_name]
            model, r2, rmse, mae = train_and_save_model(illness_name, X, y, params)
            
            # Compute SHAP values
            explainer, shap_values, X_sample = compute_shap_values(model, X, illness_name)
            
            # Analyze SHAP results
            shap_df, category_summary, lag_summary, direction_summary = analyze_shap_results(
                shap_values, X_sample, selected_features, illness_name
            )
            
            # Save SHAP data for R
            save_shap_data_for_r(explainer, shap_values, X_sample, illness_name)
            
            # Collect results
            all_results.append({
                'Illness': illness_name,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Num_Features': len(selected_features),
                'Top_Feature': shap_df.iloc[0]['Feature'],
                'Top_Feature_SHAP': shap_df.iloc[0]['Mean_Abs_SHAP'],
                'Top_Feature_Direction': shap_df.iloc[0]['Direction'],
                'Increase_Features': len(shap_df[shap_df['Direction'] == 'Increase']),
                'Decrease_Features': len(shap_df[shap_df['Direction'] == 'Decrease'])
            })
            
            print(f"\n✓ Completed SHAP analysis for {illness_name}")
            
        except Exception as e:
            print(f"\n✗ Error processing {illness_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(shap_output_dir / "SHAP_ANALYSIS_SUMMARY.csv", index=False)
        print(f"\n{'=' * 80}")
        print("SHAP Analysis Complete!")
        print(f"{'=' * 80}")
        print(f"\nResults saved to: {shap_output_dir}")
        print("\nSummary:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze Lag vs Non-Lag Variable Impact
Compare importance of base environmental variables vs their lagged versions
"""

import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
results_dir = project_root / "results/advanced_optimization"
output_dir = project_root / "results/lag_vs_nonlag_analysis"
output_dir.mkdir(exist_ok=True, parents=True)

# Three illnesses
illnesses = [
    ("Acute_laryngopharyngitis", "Acute laryngopharyngitis"),
    ("Gastritis_unspecified", "Gastritis, unspecified"),
    ("Chronic_rhinitis", "Chronic rhinitis")
]

print("="*80)
print("LAG vs NON-LAG VARIABLE IMPACT ANALYSIS")
print("="*80)

all_analyses = []

for safe_name, full_name in illnesses:
    print(f"\n{'='*80}")
    print(f"ILLNESS: {full_name}")
    print(f"{'='*80}")
    
    # Load feature importance
    importance_file = results_dir / safe_name / "feature_importance.csv"
    if not importance_file.exists():
        print(f"⚠️ Feature importance file not found: {importance_file}")
        continue
    
    imp_df = pd.read_csv(importance_file)
    
    # Classify features
    lag_features = []
    base_features = []
    base_var_importances = {}  # Track base variable and all its lags
    
    for _, row in imp_df.iterrows():
        feat = row['Feature']
        imp = row['Importance']
        
        if '_lag_' in feat:
            lag_features.append((feat, imp))
            # Extract base variable name
            base_var = feat.split('_lag_')[0]
            if base_var not in base_var_importances:
                base_var_importances[base_var] = {'base': 0, 'lags': [], 'total_lag': 0}
            base_var_importances[base_var]['lags'].append((feat, imp))
            base_var_importances[base_var]['total_lag'] += imp
        else:
            base_features.append((feat, imp))
            if feat not in base_var_importances:
                base_var_importances[feat] = {'base': imp, 'lags': [], 'total_lag': 0}
            else:
                base_var_importances[feat]['base'] = imp
    
    # Calculate statistics
    total_importance = imp_df['Importance'].sum()
    base_importance = sum(imp for _, imp in base_features)
    lag_importance = sum(imp for _, imp in lag_features)
    
    base_pct = (base_importance / total_importance) * 100
    lag_pct = (lag_importance / total_importance) * 100
    
    print(f"\nFeature Count:")
    print(f"  Base features: {len(base_features)}")
    print(f"  Lag features: {len(lag_features)}")
    print(f"  Total: {len(imp_df)}")
    
    print(f"\nImportance Distribution:")
    print(f"  Base features: {base_importance:.4f} ({base_pct:.1f}%)")
    print(f"  Lag features: {lag_importance:.4f} ({lag_pct:.1f}%)")
    
    # Analyze each base variable's lag contribution
    print(f"\nBase Variable Analysis (Base vs Lag Contribution):")
    
    var_analysis = []
    for var, data in sorted(base_var_importances.items(), 
                            key=lambda x: x[1]['base'] + x[1]['total_lag'], 
                            reverse=True):
        total = data['base'] + data['total_lag']
        if total > 0:
            var_analysis.append({
                'Variable': var,
                'Base_Importance': data['base'],
                'Lag_Importance': data['total_lag'],
                'Total_Importance': total,
                'Base_Pct': (data['base'] / total * 100) if total > 0 else 0,
                'Lag_Pct': (data['total_lag'] / total * 100) if total > 0 else 0,
                'Num_Lags': len(data['lags'])
            })
            
            if data['base'] > 0 or len(data['lags']) > 0:
                print(f"  {var:20s} | Base: {data['base']:.4f} ({data['base']/total*100:5.1f}%) | "
                      f"Lags: {data['total_lag']:.4f} ({data['total_lag']/total*100:5.1f}%) | "
                      f"# Lags: {len(data['lags'])}")
    
    # Save analysis
    var_analysis_df = pd.DataFrame(var_analysis)
    var_analysis_df.to_csv(output_dir / f"{safe_name}_variable_analysis.csv", index=False)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Category': ['Base Features', 'Lag Features'],
        'Count': [len(base_features), len(lag_features)],
        'Total_Importance': [base_importance, lag_importance],
        'Pct_Importance': [base_pct, lag_pct],
        'Avg_Importance': [
            base_importance / len(base_features) if base_features else 0,
            lag_importance / len(lag_features) if lag_features else 0
        ]
    })
    comparison.to_csv(output_dir / f"{safe_name}_base_vs_lag_summary.csv", index=False)
    
    # Top lag features by lag day
    lag_day_importance = {}
    for feat, imp in lag_features:
        lag_num = int(feat.split('_lag_')[1])
        if lag_num not in lag_day_importance:
            lag_day_importance[lag_num] = 0
        lag_day_importance[lag_num] += imp
    
    print(f"\nImportance by Lag Day:")
    for lag_day in sorted(lag_day_importance.keys()):
        print(f"  Lag {lag_day:2d}: {lag_day_importance[lag_day]:.4f}")
    
    lag_day_df = pd.DataFrame([
        {'Lag_Day': k, 'Total_Importance': v} 
        for k, v in sorted(lag_day_importance.items())
    ])
    lag_day_df.to_csv(output_dir / f"{safe_name}_lag_day_importance.csv", index=False)
    
    # Collect for overall summary
    all_analyses.append({
        'Illness': full_name,
        'Base_Features': len(base_features),
        'Lag_Features': len(lag_features),
        'Base_Importance_Pct': base_pct,
        'Lag_Importance_Pct': lag_pct,
        'Most_Important_Base': base_features[0][0] if base_features else 'N/A',
        'Most_Important_Lag': lag_features[0][0] if lag_features else 'N/A'
    })

# Save overall summary
summary_df = pd.DataFrame(all_analyses)
summary_df.to_csv(output_dir / "LAG_VS_NONLAG_SUMMARY.csv", index=False)

print("\n" + "="*80)
print("SUMMARY ACROSS ALL ILLNESSES")
print("="*80)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

avg_base_pct = summary_df['Base_Importance_Pct'].mean()
avg_lag_pct = summary_df['Lag_Importance_Pct'].mean()

print(f"\nAverage Importance Distribution:")
print(f"  Base features: {avg_base_pct:.1f}%")
print(f"  Lag features: {avg_lag_pct:.1f}%")

print(f"\nInterpretation:")
if avg_lag_pct > avg_base_pct:
    print(f"  → Lag features are MORE important ({avg_lag_pct:.1f}% vs {avg_base_pct:.1f}%)")
    print(f"  → Historical environmental patterns matter more than current conditions")
elif avg_base_pct > avg_lag_pct:
    print(f"  → Base features are MORE important ({avg_base_pct:.1f}% vs {avg_lag_pct:.1f}%)")
    print(f"  → Current environmental conditions matter more than historical patterns")
else:
    print(f"  → Base and lag features are equally important")

print(f"\n" + "="*80)
print(f"Results saved to: {output_dir}")
print("="*80)

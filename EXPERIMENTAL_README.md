# Experimental Branch: Comprehensive Time Series Features

This branch contains experimental work on expanding time series features for the illness prediction model.

## What's Different?

### Original Model Features:
- **Lag features:** 7, 14, 21 days
- **Rolling statistics:** 7 and 14-day windows for base features only
- **Total lag features:** 4 features × 3 lags = 12 features

### Experimental Model Features:
- **Lag features:** **1-14 days** (comprehensive daily lags)
- **Rolling statistics:** 7 and 14-day windows for base features
- **NEW: Rolling means for lag features:** 3 and 7-day windows for each lag feature
- **Total lag features:** 4 features × 14 lags = **56 features**
- **Total rolling features for lags:** 4 features × 14 lags × 2 windows = **112 features**

## Feature Count Comparison

### Original Model:
- Lag features: 12 (4 features × 3 lags)
- Rolling features (base): 16 (4 features × 2 windows × 2 stats)
- **Total time series features: ~28**

### Experimental Model:
- Lag features: **56** (4 features × 14 lags)
- Rolling features (base): 16 (4 features × 2 windows × 2 stats)
- Rolling features (lags): **112** (4 features × 14 lags × 2 windows)
- **Total time series features: ~184**

**Increase:** ~6.5x more time series features

## Usage

### 1. Preprocess Data with Comprehensive Features

```bash
python scripts/preprocess_experimental.py \
    --illness "Acute_upper_respiratory_infections" \
    --lag 0
```

This will create a file with comprehensive time series features:
- `data/Processed Data/Illness & Environmental/experimental/merged_data_Acute_upper_respiratory_infections_lag0_comprehensive_ts.csv`

### 2. Train Experimental Model

```bash
python scripts/train_experimental.py \
    --illness "Acute_upper_respiratory_infections" \
    --lag 0
```

This will:
- Load the comprehensive feature dataset
- Apply RFECV feature selection (reduces from ~184+ features)
- Optimize hyperparameters with Optuna
- Train final model
- Save results to `results/experimental/models/`

## Expected Benefits

1. **More Granular Temporal Patterns:** Daily lags (1-14) capture short-term patterns missed by weekly lags
2. **Smoother Trends:** Rolling means on lag features capture trends in historical patterns
3. **Better Short-Term Predictions:** 1-3 day lags may improve immediate predictions
4. **Richer Feature Space:** More features for RFECV to select from

## Potential Challenges

1. **Computational Cost:** More features = longer training time
2. **Overfitting Risk:** More features may increase overfitting (mitigated by RFECV)
3. **Feature Redundancy:** Some lag features may be highly correlated
4. **Memory Usage:** Larger datasets require more memory

## Files Created

### Configuration:
- `configs/config_experimental.yaml` - Experimental configuration

### Code:
- `src/features/engineering_experimental.py` - Extended feature engineering functions
- `scripts/preprocess_experimental.py` - Experimental preprocessing script
- `scripts/train_experimental.py` - Experimental training script

### Outputs:
- `data/Processed Data/Illness & Environmental/experimental/` - Preprocessed data
- `results/experimental/models/` - Trained models and results
- `results/experimental/plots/` - Evaluation plots

## Comparison with Main Branch

| Aspect | Main Branch | Experimental Branch |
|--------|-------------|---------------------|
| Lag days | [7, 14, 21] | [1, 2, ..., 14] |
| Rolling for lags | No | Yes (3, 7-day windows) |
| Total TS features | ~28 | ~184 |
| Config file | `configs/config.yaml` | `configs/config_experimental.yaml` |
| Scripts | `preprocess.py`, `train.py` | `preprocess_experimental.py`, `train_experimental.py` |

## Next Steps

1. Run preprocessing to generate comprehensive features
2. Train experimental model
3. Compare performance with main branch model
4. Analyze which new features are most important
5. Decide whether to merge improvements back to main branch

## Notes

- This is an experimental branch - all changes are isolated
- Original files are NOT modified
- Results are saved in separate `experimental/` directories
- Can safely switch back to `main` branch anytime

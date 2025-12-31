# Quick Start Guide

This guide will help you get started with the refactored codebase.

## Prerequisites

1. **Python 3.8+** installed
2. **Data files** placed in the correct directories (see Data Setup below)

## Setup Steps

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `configs/config.yaml` to match your data structure:

- Update `paths.climate_data_dir` to point to your climate data
- Update `paths.atmospheric_data_dir` to point to your atmospheric data  
- Update `paths.illness_data_dir` to point to your illness data
- Update output paths as needed

### 3. Data Setup

Ensure your data is organized as follows:

```
data/
├── Raw Data/
│   ├── Climate Data 2/
│   │   ├── 2018.csv
│   │   ├── 2019.csv
│   │   └── ...
│   ├── Atmospheric CSV/
│   │   └── (atmospheric data files)
│   └── Illness data/
│       └── (illness data files)
└── Processed Data/
    ├── Merged Data/
    │   └── merged_env.csv  # Created by merging climate + atmospheric
    └── Illness Data/
        └── final_illnesses/
            └── combined_illness.csv
```

**Note**: If `merged_env.csv` doesn't exist, you may need to run the merging scripts from the `archive/` folder first, or create a script to merge climate and atmospheric data.

### 4. Run the Pipeline

#### Step 1: Preprocess Data

```bash
python scripts/preprocess.py \
    --illness "Acute_upper_respiratory_infections" \
    --lag 0
```

This will:
- Load and process climate/atmospheric/illness data
- Apply feature engineering (lags, rolling stats)
- Save merged dataset ready for modeling

#### Step 2: Train Model

```bash
python scripts/train.py \
    --illness "Acute_upper_respiratory_infections" \
    --lag 0
```

This will:
- Load preprocessed data
- Perform RFECV feature selection
- Optimize hyperparameters with Optuna
- Train final XGBoost model
- Save model and metrics

#### Step 3: Evaluate Model

```bash
python scripts/evaluate.py \
    --illness "Acute_upper_respiratory_infections" \
    --lag 0
```

This will:
- Load trained model
- Evaluate on test set
- Generate prediction plots
- Generate feature importance plots

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running scripts from the project root:

```bash
cd "/Users/jay/Desktop/Illness Prediction"
python scripts/preprocess.py ...
```

### File Not Found Errors

1. Check that data files exist at the paths specified in `configs/config.yaml`
2. Verify file names match expected patterns (e.g., `2019.csv` for climate data)
3. Ensure output directories exist or can be created

### Missing Dependencies

If you get module not found errors:

```bash
pip install -r requirements.txt
```

### Path Issues on Windows

If you're on Windows, paths in `config.yaml` use forward slashes which should work, but if you encounter issues, you can use Windows-style paths or use `Path` objects in the code.

## Next Steps

- Experiment with different lag values (0, 3, 7 days)
- Try different illnesses
- Adjust hyperparameters in `configs/config.yaml`
- Explore the code in `src/` to understand the implementation

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review code comments in `src/` modules
- Check `archive/` folder for original scripts if you need reference


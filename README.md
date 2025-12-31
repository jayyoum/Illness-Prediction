# Illness Prediction Using Environmental Data

A machine learning project for predicting daily illness risk using environmental and meteorological data from South Korea.

## Overview

This project develops predictive models for common acute illnesses by leveraging:
- **Meteorological data**: Temperature, humidity, precipitation, wind, pressure, etc.
- **Air quality data**: PM10, PM2.5, SO2, NO2, O3, CO concentrations
- **Illness consultation data**: Daily case counts by region and illness type

The models use **XGBoost** with advanced feature engineering (lag features, rolling statistics) and optimization techniques (RFECV, Optuna) to predict illness incidence.

## Project Structure

```
illness-prediction/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── configs/
│   └── config.yaml          # Configuration file (paths, parameters)
│
├── src/                     # Source code modules
│   ├── data/                # Data loading and preprocessing
│   │   ├── loaders.py       # Load climate, atmospheric, illness data
│   │   ├── preprocessing.py # Missing value handling, aggregation
│   │   └── merging.py       # Merge illness and environmental data
│   │
│   ├── features/            # Feature engineering
│   │   ├── engineering.py  # Lag features, rolling stats, temporal features
│   │   └── selection.py    # Forward selection, RFECV
│   │
│   ├── models/             # Model training
│   │   ├── xgboost_model.py # XGBoost trainer with early stopping
│   │   └── training.py      # Optuna optimization, final model training
│   │
│   ├── evaluation/         # Evaluation and visualization
│   │   ├── metrics.py      # Evaluation metrics (R², RMSE, MAE, MAPE)
│   │   └── plotting.py     # Prediction plots, feature importance
│   │
│   └── utils/              # Utilities
│       └── config.py       # Configuration management
│
├── scripts/                 # Main entrypoint scripts
│   ├── preprocess.py        # Data preprocessing pipeline
│   ├── train.py             # Model training pipeline
│   └── evaluate.py         # Model evaluation and visualization
│
├── data/                    # Data directory (gitignored)
│   ├── Raw Data/           # Raw input data
│   └── Processed Data/     # Processed intermediate data
│
├── results/                 # Output directory (gitignored)
│   ├── plots/              # Generated plots
│   └── models/            # Trained models
│
└── archive/                # Archived old scripts
```

## Dataset Description

### Data Sources

1. **Climate Data** (KMA - Korea Meteorological Administration)
   - Daily measurements from weather stations across South Korea
   - Variables: temperature, humidity, precipitation, wind, pressure, solar radiation, etc.
   - Aggregated to regional level (17 regions)

2. **Atmospheric Data** (Air Korea)
   - Daily air quality measurements
   - Variables: PM10, PM2.5, SO2, NO2, O3, CO
   - Regional aggregation

3. **Illness Consultation Data**
   - Daily case counts by illness type and region
   - Multiple illness categories (respiratory infections, cardiovascular, etc.)

### Data Processing

- **Regional Aggregation**: Station-level climate data aggregated to 17 administrative regions
- **Missing Value Handling**: Time-based interpolation for continuous variables, zero-filling for cumulative variables
- **Feature Engineering**:
  - **Lag features**: 7, 14, 21-day lags for key variables
  - **Rolling statistics**: 7 and 14-day rolling mean and standard deviation
  - **Temporal features**: Year, month, day of week, week of year, day of year

## Methods

### Feature Engineering

1. **Lag Features**: Capture delayed effects of environmental factors on illness
   - Example: `AvgTemp_lag_7`, `PM10_lag_14`

2. **Rolling Statistics**: Capture trends and volatility
   - Example: `AvgTemp_rolling_mean_7`, `PM10_rolling_std_14`

3. **Temporal Features**: Capture seasonal and weekly patterns

### Model Architecture

- **Algorithm**: XGBoost (Gradient Boosting)
- **Feature Selection**: RFECV (Recursive Feature Elimination with Cross-Validation)
- **Hyperparameter Optimization**: Optuna (Bayesian optimization)
- **Early Stopping**: Prevents overfitting during training
- **Cross-Validation**: Time-series cross-validation to respect temporal ordering

### Training Pipeline

1. **Data Split**: 70% train, 15% validation, 15% test (temporal split)
2. **Feature Selection**: RFECV selects optimal feature subset
3. **Hyperparameter Tuning**: Optuna optimizes XGBoost parameters
4. **Final Training**: Train on train+validation, evaluate on test

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd illness-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure paths in `configs/config.yaml`:
   - Update data paths to match your local setup
   - Adjust parameters as needed

## Usage

### Quick Start

The workflow consists of three main steps:

#### 1. Preprocess Data

Process and merge illness and environmental data:

```bash
python scripts/preprocess.py --illness "Acute_upper_respiratory_infections" --lag 0
```

Options:
- `--illness`: Illness name to process (default: from config)
- `--lag`: Lag days for illness-environment merge (default: 0)
- `--config`: Path to config file (default: configs/config.yaml)

#### 2. Train Model

Train XGBoost model with RFECV and Optuna:

```bash
python scripts/train.py --illness "Acute_upper_respiratory_infections" --lag 0
```

Options:
- `--illness`: Illness name to train for
- `--lag`: Lag days used in preprocessing
- `--no-rfecv`: Skip RFECV feature selection
- `--no-optuna`: Skip Optuna optimization

#### 3. Evaluate Model

Evaluate trained model and generate plots:

```bash
python scripts/evaluate.py --illness "Acute_upper_respiratory_infections" --lag 0
```

### Advanced Usage

#### Process Multiple Illnesses

```bash
# Process different illnesses
for illness in "Acute_upper_respiratory_infections" "Chronic_kidney_disease"; do
    python scripts/preprocess.py --illness "$illness" --lag 0
    python scripts/train.py --illness "$illness" --lag 0
    python scripts/evaluate.py --illness "$illness" --lag 0
done
```

#### Experiment with Different Lags

```bash
# Test different lag days
for lag in 0 3 7; do
    python scripts/preprocess.py --illness "Acute_upper_respiratory_infections" --lag $lag
    python scripts/train.py --illness "Acute_upper_respiratory_infections" --lag $lag
done
```

## Outputs

### Model Artifacts

- **Trained Models**: Saved as `.pkl` files in `results/models/`
- **Metrics**: CSV files with R², RMSE, MAE, MAPE
- **Selected Features**: List of features selected by RFECV

### Visualizations

- **Predictions vs. Actual**: Time series plot comparing predictions to actual values
- **Feature Importance**: Bar plot showing top features by importance
- **Optuna Optimization History**: Optimization progress over trials
- **Parameter Importances**: Relative importance of hyperparameters

All outputs are saved in the `results/` directory.

## Configuration

Edit `configs/config.yaml` to customize:

- **Data paths**: Update paths to your data directories
- **Feature engineering**: Adjust lag days, rolling windows, features to use
- **Model parameters**: XGBoost defaults, RFECV settings, Optuna trials
- **Training parameters**: Train/val/test split ratios, early stopping

## Project Status

**Current Status**: Paper drafting in progress

This repository contains the complete codebase for the research project. The models have been trained and evaluated, and results are being prepared for publication.

### Future Work

- [ ] Extend to additional illness types
- [ ] Incorporate more environmental variables
- [ ] Develop ensemble methods
- [ ] Create web-based prediction interface
- [ ] Add real-time data integration

## Citation

If you use this code in your research, please cite:

```bibtex
@article{illness_prediction_2024,
  title={Optimizing illness prediction models using environmental data and machine learning},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is a research codebase. For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Korea Meteorological Administration (KMA) for climate data
- Air Korea for air quality data
- Healthcare institutions for illness consultation data

## Contact

For questions about this project, please contact jayyoum21@gmail.com

---

**Note**: This repository does not include raw data files. Please obtain data from the respective sources and place them in the `data/Raw Data/` directory following the structure described in the configuration file.


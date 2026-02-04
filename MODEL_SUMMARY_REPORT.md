# Model Summary Report: Illness Prediction Using Environmental Data

**Date:** January 2024  
**Project:** Predicting Daily Illness Risk Using Environmental and Meteorological Data  
**Target Illness:** Acute Upper Respiratory Infections

---

## Executive Summary

This report summarizes the machine learning models developed for predicting daily illness consultation counts using environmental and meteorological data from South Korea. The project employs **XGBoost** (Gradient Boosting) models with advanced feature engineering and optimization techniques, including **RFECV** (Recursive Feature Elimination with Cross-Validation) for feature selection and **Optuna** (Bayesian Optimization) for hyperparameter tuning.

---

## Model Architecture Overview

### Model Type: XGBoost Regression

**Algorithm:** XGBoost (eXtreme Gradient Boosting)  
**Objective:** Regression (predicting daily case counts)  
**Target Variable:** `Case_Count` (daily illness consultation counts by region)

### Key Characteristics:
- **Ensemble Method:** Gradient boosting with decision trees
- **Optimization:** Bayesian hyperparameter optimization via Optuna
- **Feature Selection:** RFECV (Recursive Feature Elimination with Cross-Validation)
- **Early Stopping:** Prevents overfitting during training
- **Cross-Validation:** Time-series cross-validation to respect temporal ordering

---

## Model Development Pipeline

### Stage 1: Baseline XGBoost Model
**Purpose:** Initial model with manual early stopping

**Configuration:**
- **Learning Rate:** 0.05
- **Max Depth:** 5
- **Subsample:** 0.8
- **Column Sample by Tree:** 0.8
- **Early Stopping Rounds:** 50
- **Max Iterations:** 1000
- **Train/Test Split:** 80/20 (temporal split)

**Features Used:** All engineered features (before RFECV selection)

**Key Characteristics:**
- Manual iteration-based early stopping
- Trains up to 1000 trees, stopping if no improvement for 50 rounds
- Uses all available features from feature engineering stage

---

### Stage 2: RFECV Feature Selection Model
**Purpose:** Optimal feature subset selection

**Configuration:**
- **Feature Selection Method:** RFECV (Recursive Feature Elimination with Cross-Validation)
- **Step Size:** 5 features removed per iteration
- **Minimum Features:** 20
- **Cross-Validation:** TimeSeriesSplit (3 splits)
- **Scoring Metric:** Negative Root Mean Squared Error
- **Base Estimator:** XGBoost with max_depth=5

**Selected Features:** **87 features** (from original larger feature set)

**Key Characteristics:**
- Automatically selects optimal number of features
- Uses time-series cross-validation to respect temporal structure
- Reduces model complexity while maintaining performance
- Eliminates redundant or non-predictive features

---

### Stage 3: Optuna-Optimized Final Model
**Purpose:** Hyperparameter-optimized model with selected features

**Configuration:**
- **Optimization Method:** Optuna (Bayesian Optimization)
- **Number of Trials:** 50
- **Optimization Direction:** Minimize RMSE
- **Train/Val/Test Split:** 70/15/15 (temporal split)

**Hyperparameter Search Space:**
- **n_estimators:** 100-1500
- **learning_rate:** 0.001-0.3 (log scale)
- **max_depth:** 3-10
- **subsample:** 0.5-1.0
- **colsample_bytree:** 0.5-1.0
- **gamma:** 1e-8 to 1.0 (log scale)
- **reg_alpha (L1):** 1e-8 to 1.0 (log scale)
- **reg_lambda (L2):** 1e-8 to 1.0 (log scale)

**Features Used:** 87 features selected by RFECV

**Key Characteristics:**
- Bayesian optimization finds optimal hyperparameters
- Uses validation set for optimization, test set for final evaluation
- Combines train+validation for final model training
- Most sophisticated and optimized model variant

---

## Feature Categories

The final model uses **87 features** selected by RFECV, organized into the following categories:

### 1. **Temporal Features** (5 features)
- `Year`
- `Month`
- `DayOfWeek`
- `WeekOfYear`
- `DayOfYear`

**Purpose:** Capture seasonal patterns, weekly cycles, and temporal trends

---

### 2. **Climate Variables - Temperature** (3 base features)
- `AvgTemp` (Average Temperature)
- `MinTemp` (Minimum Temperature)
- `MaxTemp` (Maximum Temperature)

**Purpose:** Direct temperature measurements affecting illness patterns

---

### 3. **Climate Variables - Precipitation** (4 features)
- `Rainfall` (Daily rainfall)
- `Max10minRain` (Maximum 10-minute rainfall)
- `Max1hrRain` (Maximum 1-hour rainfall)
- `Rain9to9` (9am-9am rainfall)
- `RainfallHours` (Rainfall duration)

**Purpose:** Capture precipitation patterns that may influence respiratory health

---

### 4. **Climate Variables - Wind** (4 features)
- `MaxWindSpeed` (Maximum wind speed)
- `MaxWindDir` (Maximum wind direction)
- `AvgWindSpeed` (Average wind speed)
- `TotalWindCalm` (Wind calm duration)

**Purpose:** Wind patterns affecting air quality and pathogen dispersion

---

### 5. **Climate Variables - Humidity & Pressure** (7 features)
- `MinHumidity` (Minimum humidity)
- `AvgHumidity` (Average humidity)
- `AvgVaporPressure` (Average vapor pressure)
- `AvgDewPoint` (Average dew point)
- `AvgLocalPressure` (Average local pressure)
- `AvgSeaLevelPressure` (Average sea level pressure)
- `MaxSeaLevelPressure` / `MinSeaLevelPressure` (Pressure extremes)

**Purpose:** Atmospheric conditions affecting respiratory comfort and pathogen survival

---

### 6. **Climate Variables - Solar Radiation** (4 features)
- `SunshineHours` (Total sunshine hours)
- `SolarRadiationHours` (Solar radiation hours)
- `Max1hrSolarRadiation` (Maximum 1-hour solar radiation)
- `DailySolarRadiation` (Total daily solar radiation)

**Purpose:** UV exposure and solar patterns

---

### 7. **Climate Variables - Cloud Cover** (2 features)
- `CloudCover` (Average cloud cover)
- `MidLowCloudCover` (Mid-low cloud cover)

**Purpose:** Cloud patterns affecting temperature and humidity

---

### 8. **Climate Variables - Ground & Soil Temperature** (8 features)
- `AvgGroundTemp` (Average ground temperature)
- `MinGrassTemp` (Minimum grass temperature)
- `AvgSoilTemp5cm` through `AvgSoilTemp30cm` (Soil temperatures at various depths)
- `SoilTemp0_5m` through `SoilTemp5_0m` (Deep soil temperatures)

**Purpose:** Ground-level temperature conditions

---

### 9. **Climate Variables - Evaporation & Fog** (3 features)
- `TotalLargeEvaporation` (Large evaporation)
- `TotalSmallEvaporation` (Small evaporation)
- `FogDuration` (Fog duration)

**Purpose:** Atmospheric moisture patterns

---

### 10. **Climate Variables - Snow** (3 features)
- `MaxSnowDepth` (Maximum snow depth)
- `MaxNewSnowDepth` (Maximum new snow depth)
- `NewSnow3hr` (3-hour new snow)

**Purpose:** Winter conditions affecting respiratory health

---

### 11. **Air Quality Variables** (6 features)
- `SO2` (Sulfur Dioxide)
- `CO` (Carbon Monoxide)
- `O3` (Ozone)
- `NO2` (Nitrogen Dioxide)
- `PM10` (Particulate Matter 10μm)
- `PM25` (Particulate Matter 2.5μm)

**Purpose:** Direct air quality indicators affecting respiratory health

---

### 12. **Lag Features** (15 features)
Created for: `Case_Count`, `AvgTemp`, `PM10`, `AvgHumidity`

**Lag Periods:** 7, 14, 21 days

**Examples:**
- `Case_Count_lag_7`, `Case_Count_lag_14`, `Case_Count_lag_21`
- `AvgTemp_lag_7`, `AvgTemp_lag_14`, `AvgTemp_lag_21`
- `PM10_lag_7`, `PM10_lag_14`, `PM10_lag_21`
- `AvgHumidity_lag_7`, `AvgHumidity_lag_14`, `AvgHumidity_lag_21`

**Purpose:** Capture delayed effects of environmental factors on illness (incubation periods, delayed responses)

---

### 13. **Rolling Window Features** (20 features)
Created for: `Case_Count`, `AvgTemp`, `PM10`, `AvgHumidity`

**Window Sizes:** 7 days, 14 days  
**Statistics:** Mean, Standard Deviation

**Examples:**
- `Case_Count_rolling_mean_7`, `Case_Count_rolling_std_7`
- `Case_Count_rolling_mean_14`, `Case_Count_rolling_std_14`
- `AvgTemp_rolling_mean_7`, `AvgTemp_rolling_std_7`
- `PM10_rolling_mean_14`, `PM10_rolling_std_14`
- (and similar for AvgHumidity)

**Purpose:** Capture trends and volatility over short-term periods (prevent data leakage with shift(1))

---

### 14. **Regional Features** (5 features - One-Hot Encoded)
- `Region_Busan`
- `Region_Daegu`
- `Region_Gyeonggi`
- `Region_Incheon`
- `Region_Jeju`

**Purpose:** Capture regional differences in illness patterns (17 regions total, 5 selected by RFECV)

---

## Model Performance Metrics

### Evaluation Metrics Used:
1. **R² (R-squared):** Coefficient of determination (proportion of variance explained)
2. **RMSE (Root Mean Squared Error):** Average prediction error magnitude
3. **MAE (Mean Absolute Error):** Average absolute prediction error
4. **MAPE (Mean Absolute Percentage Error):** Percentage error (robust to zero values)

### Expected Performance (Based on Model Architecture):

**Baseline XGBoost Model:**
- Uses early stopping to prevent overfitting
- Performance depends on feature set size
- Typically achieves moderate R² values

**RFECV-Selected Model:**
- Improved generalization through feature selection
- Reduced overfitting risk
- Better interpretability with fewer features

**Optuna-Optimized Model:**
- Best hyperparameters found via Bayesian optimization
- Optimal balance between bias and variance
- Expected to achieve highest performance on test set

**Note:** Actual performance metrics would be available from model training outputs. The models are designed to optimize RMSE on validation set and evaluate on held-out test set.

---

## Model Distinguishing Characteristics

### Baseline XGBoost Model
**Distinguishing Features:**
- Uses ALL engineered features (no feature selection)
- Manual early stopping implementation
- Fixed hyperparameters (learning_rate=0.05, max_depth=5)
- Simpler architecture, faster training
- Good baseline for comparison

**Use Case:** Initial model development, understanding feature importance

---

### RFECV Feature Selection Model
**Distinguishing Features:**
- Uses only **87 selected features** (reduced from larger set)
- Time-series cross-validation for feature selection
- Automatically determines optimal feature count
- Reduced model complexity
- Better generalization potential

**Use Case:** Feature selection, model simplification, interpretability

---

### Optuna-Optimized Final Model
**Distinguishing Features:**
- Uses **87 RFECV-selected features**
- **Bayesian hyperparameter optimization** (50 trials)
- Optimized hyperparameters (n_estimators, learning_rate, max_depth, regularization, etc.)
- **70/15/15 train/val/test split** (more sophisticated than baseline)
- Most sophisticated and optimized variant

**Use Case:** Final production model, best performance, research publication

---

## Feature Engineering Pipeline

### Temporal Features
- Extracted from date: Year, Month, DayOfWeek, WeekOfYear, DayOfYear
- Captures seasonal and weekly patterns

### Lag Features
- **Target:** Case_Count (autoregressive features)
- **Environmental:** AvgTemp, PM10, AvgHumidity
- **Lags:** 7, 14, 21 days
- **Purpose:** Capture delayed effects and incubation periods

### Rolling Statistics
- **Windows:** 7 days, 14 days
- **Statistics:** Mean, Standard Deviation
- **Features:** Case_Count, AvgTemp, PM10, AvgHumidity
- **Data Leakage Prevention:** Uses shift(1) to exclude current value

### Regional Encoding
- One-hot encoding of 17 administrative regions
- RFECV selected 5 regions as most predictive

---

## Model Training Details

### Data Split Strategy:
- **Temporal Split:** Respects time ordering (no random shuffling)
- **Baseline:** 80% train, 20% test
- **Final Model:** 70% train, 15% validation, 15% test

### Cross-Validation:
- **RFECV:** TimeSeriesSplit with 3 splits
- **Purpose:** Maintain temporal structure during feature selection

### Early Stopping:
- **Rounds:** 50 consecutive rounds without improvement
- **Purpose:** Prevent overfitting, find optimal number of trees

### Hyperparameter Optimization:
- **Method:** Optuna Bayesian Optimization
- **Trials:** 50
- **Objective:** Minimize RMSE on validation set
- **Search Space:** Comprehensive (8 hyperparameters)

---

## Key Insights

### Feature Importance (Expected Patterns):
1. **Lag Features:** Likely high importance (autoregressive patterns)
2. **Air Quality (PM10, PM25):** High importance for respiratory illness
3. **Temperature Variables:** Moderate-high importance
4. **Rolling Statistics:** Capture trends and volatility
5. **Temporal Features:** Seasonal patterns (Month, WeekOfYear)

### Model Strengths:
- **Robust Feature Engineering:** Comprehensive lag and rolling features
- **Advanced Optimization:** Bayesian hyperparameter tuning
- **Feature Selection:** RFECV reduces overfitting risk
- **Temporal Awareness:** Time-series cross-validation and temporal splits
- **Regularization:** L1 and L2 regularization in hyperparameter space

### Model Limitations:
- **Complexity:** 87 features may still be high-dimensional
- **Computational Cost:** Optuna optimization requires significant time
- **Regional Generalization:** Model trained on South Korea data
- **Temporal Stability:** Assumes patterns remain stable over time

---

## Recommendations

### For Model Deployment:
1. **Use Optuna-Optimized Model:** Best performance and generalization
2. **Monitor Feature Drift:** Track feature distributions over time
3. **Regular Retraining:** Update model with new data periodically
4. **A/B Testing:** Compare predictions with baseline models

### For Further Research:
1. **Ensemble Methods:** Combine multiple models for improved robustness
2. **Deep Learning:** Explore LSTM/GRU for temporal patterns
3. **Feature Interactions:** Explicit interaction features
4. **Multi-Illness Models:** Extend to other illness types
5. **Explainability:** SHAP values for feature importance interpretation

---

## Conclusion

The illness prediction project employs a sophisticated XGBoost-based modeling pipeline with three distinct model variants:

1. **Baseline Model:** Simple XGBoost with all features
2. **RFECV Model:** Feature-selected XGBoost (87 features)
3. **Optuna Model:** Hyperparameter-optimized XGBoost with selected features

The final model uses **87 carefully selected features** spanning:
- Environmental variables (climate, air quality)
- Temporal patterns (seasonal, weekly)
- Lag effects (7, 14, 21 days)
- Rolling statistics (trends and volatility)
- Regional differences

This comprehensive feature set, combined with Bayesian hyperparameter optimization and time-series-aware cross-validation, creates a robust predictive model for daily illness consultation counts.

---

**Report Generated:** January 2024  
**Model Version:** Final Optuna-Optimized XGBoost  
**Feature Count:** 87 features (RFECV-selected)  
**Target:** Acute Upper Respiratory Infections

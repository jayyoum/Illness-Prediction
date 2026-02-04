"""Experimental feature engineering with comprehensive time series features.

This module extends the base feature engineering with:
- Lag features for days 1-14 (instead of just 7, 14, 21)
- Rolling mean features for each lag feature
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_lag_features_comprehensive(
    df: pd.DataFrame,
    features: List[str],
    lag_days: List[int],
    group_col: Optional[str] = 'Region'
) -> pd.DataFrame:
    """
    Create comprehensive lag features for specified columns (1-14 days).
    
    Args:
        df: DataFrame
        features: List of feature names to create lags for
        lag_days: List of lag days (e.g., [1, 2, 3, ..., 14])
        group_col: Column to group by (typically region)
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found. Skipping lag creation.")
            continue
        
        for lag in lag_days:
            col_name = f'{feature}_lag_{lag}'
            if group_col and group_col in df.columns:
                df[col_name] = df.groupby(group_col)[feature].shift(lag)
            else:
                df[col_name] = df[feature].shift(lag)
    
    logger.info(
        f"Created comprehensive lag features for {len(features)} features "
        f"with lags {lag_days} (total: {len(features) * len(lag_days)} lag features)"
    )
    return df


def create_rolling_mean_for_lags(
    df: pd.DataFrame,
    base_features: List[str],
    lag_days: List[int],
    rolling_windows: List[int],
    group_col: Optional[str] = 'Region'
) -> pd.DataFrame:
    """
    Create rolling mean features for each lag feature.
    
    For each base feature and each lag day, creates rolling mean features.
    Example: For AvgTemp_lag_7, creates AvgTemp_lag_7_rolling_mean_3, etc.
    
    Args:
        df: DataFrame with lag features already created
        base_features: Base feature names (e.g., ['Case_Count', 'AvgTemp'])
        lag_days: List of lag days used
        rolling_windows: List of rolling window sizes (e.g., [3, 7])
        group_col: Column to group by (typically region)
    
    Returns:
        DataFrame with rolling mean features for lag features added
    """
    df = df.copy()
    
    features_created = 0
    
    for base_feature in base_features:
        for lag in lag_days:
            lag_col_name = f'{base_feature}_lag_{lag}'
            
            if lag_col_name not in df.columns:
                logger.warning(
                    f"Lag feature '{lag_col_name}' not found. "
                    f"Make sure lag features are created first."
                )
                continue
            
            # Shift to avoid data leakage (use previous values)
            if group_col and group_col in df.columns:
                shifted_series = df.groupby(group_col)[lag_col_name].shift(1)
            else:
                shifted_series = df[lag_col_name].shift(1)
            
            for window in rolling_windows:
                rolling_col_name = f'{lag_col_name}_rolling_mean_{window}'
                df[rolling_col_name] = shifted_series.rolling(
                    window=window, min_periods=1
                ).mean()
                features_created += 1
    
    logger.info(
        f"Created rolling mean features for lag features: "
        f"{features_created} new features "
        f"({len(base_features)} base features × {len(lag_days)} lags × {len(rolling_windows)} windows)"
    )
    return df


def create_comprehensive_timeseries_features(
    df: pd.DataFrame,
    base_features: List[str],
    lag_days: List[int],
    rolling_windows_base: List[int],
    rolling_windows_lags: List[int],
    group_col: Optional[str] = 'Region'
) -> pd.DataFrame:
    """
    Create comprehensive time series features in one step.
    
    This function:
    1. Creates lag features for days 1-14
    2. Creates rolling statistics for base features
    3. Creates rolling means for each lag feature
    
    Args:
        df: DataFrame
        base_features: Features to create lags and rolling stats for
        lag_days: List of lag days (e.g., [1, 2, ..., 14])
        rolling_windows_base: Rolling windows for base features (e.g., [7, 14])
        rolling_windows_lags: Rolling windows for lag features (e.g., [3, 7])
        group_col: Column to group by
    
    Returns:
        DataFrame with all time series features added
    """
    df = df.copy()
    
    logger.info("Starting comprehensive time series feature engineering...")
    
    # Step 1: Create lag features (1-14 days)
    logger.info(f"Step 1: Creating lag features for days {lag_days}")
    df = create_lag_features_comprehensive(df, base_features, lag_days, group_col)
    
    # Step 2: Create rolling statistics for base features
    logger.info(f"Step 2: Creating rolling statistics for base features")
    from .engineering import create_rolling_features
    df = create_rolling_features(
        df, base_features, rolling_windows_base, 
        stats=['mean', 'std'], group_col=group_col
    )
    
    # Step 3: Create rolling means for lag features
    logger.info(f"Step 3: Creating rolling means for lag features")
    df = create_rolling_mean_for_lags(
        df, base_features, lag_days, rolling_windows_lags, group_col
    )
    
    total_lag_features = len(base_features) * len(lag_days)
    total_rolling_base = len(base_features) * len(rolling_windows_base) * 2  # mean + std
    total_rolling_lags = len(base_features) * len(lag_days) * len(rolling_windows_lags)
    
    logger.info(
        f"Feature engineering complete:\n"
        f"  - Lag features: {total_lag_features}\n"
        f"  - Rolling features (base): {total_rolling_base}\n"
        f"  - Rolling features (lags): {total_rolling_lags}\n"
        f"  - Total time series features: {total_lag_features + total_rolling_base + total_rolling_lags}"
    )
    
    return df

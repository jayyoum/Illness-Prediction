"""Feature engineering: lags, rolling statistics, temporal features."""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Create temporal features from date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
    
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found. Skipping temporal features.")
        return df
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['DayOfWeek'] = df[date_col].dt.dayofweek
    df['WeekOfYear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df[date_col].dt.dayofyear
    
    logger.info("Created temporal features: Year, Month, DayOfWeek, WeekOfYear, DayOfYear")
    return df


def create_lag_features(
    df: pd.DataFrame,
    features: List[str],
    lag_days: List[int],
    group_col: Optional[str] = 'Region'
) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        df: DataFrame
        features: List of feature names to create lags for
        lag_days: List of lag days (e.g., [7, 14, 21])
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
    
    logger.info(f"Created lag features for {len(features)} features with lags {lag_days}")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    features: List[str],
    windows: List[int],
    stats: List[str] = ['mean', 'std'],
    group_col: Optional[str] = 'Region'
) -> pd.DataFrame:
    """
    Create rolling window statistics features.
    
    Args:
        df: DataFrame
        features: List of feature names
        windows: List of window sizes (e.g., [7, 14])
        stats: List of statistics to compute (e.g., ['mean', 'std'])
        group_col: Column to group by (typically region)
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found. Skipping rolling features.")
            continue
        
        # Shift to avoid data leakage
        if group_col and group_col in df.columns:
            shifted_series = df.groupby(group_col)[feature].shift(1)
        else:
            shifted_series = df[feature].shift(1)
        
        for window in windows:
            for stat in stats:
                col_name = f'{feature}_rolling_{stat}_{window}'
                
                if stat == 'mean':
                    df[col_name] = shifted_series.rolling(
                        window=window, min_periods=1
                    ).mean()
                elif stat == 'std':
                    df[col_name] = shifted_series.rolling(
                        window=window, min_periods=1
                    ).std()
                elif stat == 'min':
                    df[col_name] = shifted_series.rolling(
                        window=window, min_periods=1
                    ).min()
                elif stat == 'max':
                    df[col_name] = shifted_series.rolling(
                        window=window, min_periods=1
                    ).max()
    
    logger.info(
        f"Created rolling features for {len(features)} features "
        f"with windows {windows} and stats {stats}"
    )
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    One-hot encode categorical features.
    
    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
    
    Returns:
        DataFrame with one-hot encoded features
    """
    df = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col)
            logger.info(f"One-hot encoded '{col}'")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Case_Count',
    lag_features: Optional[List[str]] = None,
    lag_days: Optional[List[int]] = None,
    rolling_features: Optional[List[str]] = None,
    rolling_windows: Optional[List[int]] = None,
    exclude_cols: Optional[List[str]] = None
) -> tuple:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with all features
        target_col: Name of target column
        lag_features: Features to create lags for
        lag_days: Lag days to use
        rolling_features: Features to create rolling stats for
        rolling_windows: Rolling window sizes
        exclude_cols: Columns to exclude from features
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    df = df.copy()
    
    # Default exclusions
    if exclude_cols is None:
        exclude_cols = [
            'DateTime', 'ParsedDateTime', 'Date', 'RegionCode',
            'IllnessName', 'CaseCount', 'Case_Count'
        ]
    
    # Get feature columns
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols + [target_col]
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None
    
    # Remove rows with NaN
    if y is not None:
        valid_rows = X.notna().all(axis=1) & y.notna()
        X = X[valid_rows]
        y = y[valid_rows]
    else:
        valid_rows = X.notna().all(axis=1)
        X = X[valid_rows]
    
    logger.info(f"Prepared features: {X.shape[1]} features, {X.shape[0]} samples")
    
    return X, y


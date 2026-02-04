#!/usr/bin/env python3
"""
Experimental data preprocessing script with comprehensive time series features.

This script extends the base preprocessing with:
- Lag features for days 1-14
- Rolling mean features for each lag feature
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import KMAClimateLoader, load_illness_data, load_atmospheric_data
from src.data.preprocessing import (
    handle_missing_values_climate,
    handle_missing_values_after_grouping,
    aggregate_by_region,
    clean_illness_data
)
from src.data.merging import merge_illness_environment, save_merged_data
from src.features.engineering import create_temporal_features, encode_categorical_features
from src.features.engineering_experimental import create_comprehensive_timeseries_features
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_with_comprehensive_features(
    config: dict,
    illness_name: str,
    lag_days: int = 0
) -> None:
    """Preprocess data with comprehensive time series features."""
    logger.info(f"EXPERIMENTAL: Processing '{illness_name}' with comprehensive time series features")
    
    # Load illness data
    illness_path = Path(config['paths']['processed_data_dir']) / "Illness Data" / "final_illnesses" / "combined_illness.csv"
    if not illness_path.exists():
        logger.error(f"Illness data not found: {illness_path}")
        return
    
    illness_df = load_illness_data(str(illness_path))
    illness_df = clean_illness_data(illness_df)
    
    # Load environmental data
    env_path = Path(config['paths']['merged_env_output'])
    if not env_path.exists():
        logger.error(f"Environmental data not found: {env_path}")
        return
    
    env_df = load_atmospheric_data(str(env_path))
    
    # Merge
    merged_df = merge_illness_environment(
        illness_df, env_df, lag_days=lag_days, illness_name=illness_name
    )
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    # Apply feature engineering
    logger.info("Creating temporal features...")
    merged_df = create_temporal_features(merged_df, date_col='ParsedDateTime')
    
    # Get comprehensive time series configuration
    lag_features = config['features']['lag_features']
    lag_days_list = config['features']['lag_days']  # Now 1-14 days
    rolling_windows_base = config['features']['rolling_windows']  # [7, 14]
    rolling_windows_lags = config['features'].get('rolling_mean_windows', [3, 7])
    
    logger.info("Creating comprehensive time series features...")
    logger.info(f"  - Lag days: {lag_days_list}")
    logger.info(f"  - Base features: {lag_features}")
    logger.info(f"  - Rolling windows (base): {rolling_windows_base}")
    logger.info(f"  - Rolling windows (lags): {rolling_windows_lags}")
    
    # Create comprehensive time series features
    merged_df = create_comprehensive_timeseries_features(
        merged_df,
        base_features=lag_features,
        lag_days=lag_days_list,
        rolling_windows_base=rolling_windows_base,
        rolling_windows_lags=rolling_windows_lags,
        group_col='RegionName'
    )
    
    # Encode categorical features
    if config['features']['include_season']:
        merged_df = encode_categorical_features(merged_df, ['RegionName', 'Season'])
    
    logger.info(f"Final feature-engineered data shape: {merged_df.shape}")
    logger.info(f"Total features: {len(merged_df.columns)}")
    
    # Save merged data
    output_dir = Path(config['paths']['illness_env_output_dir']) / "experimental"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_illness_name = illness_name.replace(', ', '_').replace(' ', '_')
    filename = f"merged_data_{safe_illness_name}_lag{lag_days}_comprehensive_ts.csv"
    output_path = output_dir / filename
    
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved experimental data to: {output_path}")
    logger.info(f"Rows: {len(merged_df)}, Columns: {len(merged_df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description='EXPERIMENTAL: Preprocess data with comprehensive time series features'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config_experimental.yaml',
        help='Path to experimental config file'
    )
    parser.add_argument(
        '--illness', type=str, default=None,
        help='Illness name to process'
    )
    parser.add_argument(
        '--lag', type=int, default=0,
        help='Lag days for illness-environment merge'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    illness_name = args.illness or config['illness']['default_illness']
    lag_days = args.lag or config['illness']['default_lag_days']
    
    preprocess_with_comprehensive_features(config, illness_name, lag_days)


if __name__ == '__main__':
    main()

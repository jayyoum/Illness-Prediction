#!/usr/bin/env python3
"""
Data preprocessing script.

This script processes raw climate, atmospheric, and illness data,
applies feature engineering, and prepares datasets for modeling.
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
from src.features.engineering import (
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
    encode_categorical_features
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_climate_data(config: dict, year: int) -> None:
    """Preprocess climate data for a given year."""
    logger.info(f"Preprocessing climate data for year {year}")
    
    climate_dir = config['paths']['climate_data_dir']
    file_path = Path(climate_dir) / f"{year}.csv"
    
    if not file_path.exists():
        logger.warning(f"Climate data file not found: {file_path}")
        return
    
    loader = KMAClimateLoader()
    df = loader.load_climate_data(str(file_path))
    
    # Handle missing values
    df = handle_missing_values_climate(df)
    
    # Aggregate by region
    df = aggregate_by_region(df)
    
    # Handle missing values after grouping
    df = handle_missing_values_after_grouping(df)
    
    # Save processed data
    output_dir = Path(config['paths']['processed_data_dir']) / "Climate Data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{year}_processed_climate.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed climate data to {output_path}")


def merge_illness_and_environment(config: dict, illness_name: str, lag_days: int = 0) -> None:
    """Merge illness and environmental data."""
    logger.info(f"Merging illness '{illness_name}' with environmental data (lag={lag_days})")
    
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
    
    # Apply feature engineering
    merged_df = create_temporal_features(merged_df, date_col='ParsedDateTime')
    
    # Create lag features
    lag_features = config['features']['lag_features']
    lag_days_list = config['features']['lag_days']
    merged_df = create_lag_features(
        merged_df, lag_features, lag_days_list, group_col='RegionName'
    )
    
    # Create rolling features
    rolling_features = config['features']['rolling_features']
    rolling_windows = config['features']['rolling_windows']
    merged_df = create_rolling_features(
        merged_df, rolling_features, rolling_windows, 
        stats=config['features']['rolling_stats'], group_col='RegionName'
    )
    
    # Encode categorical features
    if config['features']['include_season']:
        merged_df = encode_categorical_features(merged_df, ['RegionName', 'Season'])
    
    # Save merged data
    output_dir = Path(config['paths']['illness_env_output_dir'])
    save_merged_data(merged_df, str(output_dir), illness_name, lag_days)
    
    logger.info(f"Preprocessing complete for {illness_name} (lag={lag_days})")


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for illness prediction')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file (default: configs/config.yaml)'
    )
    parser.add_argument(
        '--illness', type=str, default=None,
        help='Illness name to process (default: from config)'
    )
    parser.add_argument(
        '--lag', type=int, default=0,
        help='Lag days for illness-environment merge (default: 0)'
    )
    parser.add_argument(
        '--year', type=int, default=None,
        help='Year to process climate data for'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.year:
        preprocess_climate_data(config, args.year)
    else:
        # Process illness-environment merge
        illness_name = args.illness or config['illness']['default_illness']
        lag_days = args.lag or config['illness']['default_lag_days']
        merge_illness_and_environment(config, illness_name, lag_days)


if __name__ == '__main__':
    main()


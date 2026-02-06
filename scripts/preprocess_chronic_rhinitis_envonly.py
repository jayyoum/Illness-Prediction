#!/usr/bin/env python3
"""
Environmental-Only preprocessing for Chronic Rhinitis
Research Focus: Pure environmental exposure effects (NO illness autocorrelation)
Excludes: Case_Count lags, RegionName, Year, Season
Includes: Environmental lags, DayOfYear, DayOfWeek
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import yaml
import logging
from datetime import datetime

from features.engineering_experimental import create_comprehensive_timeseries_features

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = project_root / config['logging']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"preprocess_chronic_rhinitis_envonly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Load configuration
    config_path = project_root / "configs" / "config_chronic_rhinitis_envonly.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("ENVIRONMENTAL-ONLY PREPROCESSING - ACUTE CHRONIC RHINITIS")
    logger.info("=" * 80)
    logger.info(f"Illness: {config['illness_name']}")
    logger.info(f"Research Focus: Pure environmental exposure (NO illness autocorrelation)")
    logger.info(f"Excluded: Case_Count lags, RegionName, Year, Season")
    logger.info("")
    
    # Load environmental data
    logger.info("Loading environmental data...")
    env_file = project_root / config['paths']['merged_env_output']
    env_df = pd.read_csv(env_file)
    logger.info(f"Environmental data loaded: {len(env_df)} rows")
    
    # Load illness data
    logger.info(f"Loading illness data: {config['illness_name']}")
    illness_file = project_root / "Processed Data/Illness Data/final_illnesses/combined_illness.csv"
    illness_data = pd.read_csv(illness_file)
    
    # Filter for specific illness
    illness_df = illness_data[illness_data['IllnessName'] == config['illness_name']].copy()
    logger.info(f"Illness data loaded: {len(illness_df)} rows for {config['illness_name']}")
    
    if len(illness_df) == 0:
        logger.error(f"No data found for illness: {config['illness_name']}")
        logger.info(f"Available illnesses: {illness_data['IllnessName'].unique().tolist()}")
        return
    
    # Merge data
    logger.info("Merging illness and environmental data...")
    illness_df['ParsedDateTime'] = pd.to_datetime(illness_df['ParsedDateTime'])
    env_df['DateTime'] = pd.to_datetime(env_df['DateTime'])
    
    illness_df = illness_df.rename(columns={'ParsedDateTime': 'DateTime', 'CaseCount': 'Case_Count'})
    
    merged_df = pd.merge(
        illness_df,
        env_df,
        on=['DateTime', 'RegionCode'],
        how='inner'
    )
    logger.info(f"Merged data: {len(merged_df)} rows")
    
    # Rename RegionCode to Region for consistency
    merged_df = merged_df.rename(columns={'RegionCode': 'Region'})
    
    # Create comprehensive time series features (ENVIRONMENTAL ONLY - NO Case_Count!)
    logger.info("")
    logger.info("=" * 80)
    logger.info("CREATING ENVIRONMENTAL-ONLY TIME SERIES FEATURES")
    logger.info("=" * 80)
    logger.info(f"Lag days: {config['features']['lag_days']}")
    logger.info(f"Environmental variables: {config['features']['lag_features']}")
    logger.info(f"NOTE: Case_Count is EXCLUDED from lag features!")
    logger.info(f"Rolling mean windows for lags: {config['features']['rolling_mean_windows']}")
    
    processed_df = create_comprehensive_timeseries_features(
        merged_df,
        base_features=config['features']['lag_features'],  # NO Case_Count here!
        lag_days=config['features']['lag_days'],
        rolling_windows_base=config['features']['rolling_windows'],
        rolling_windows_lags=config['features']['rolling_mean_windows'],
        group_col='Region'
    )
    
    logger.info(f"Final feature count: {len(processed_df.columns)}")
    logger.info(f"Final row count: {len(processed_df)}")
    
    # Save processed data
    output_dir = project_root / config['paths']['illness_env_output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{config['illness_name']}_illnessenv_envonly.csv"
    processed_df.to_csv(output_file, index=False)
    logger.info(f"Saved processed data: {output_file}")
    
    # Summary statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Illness: {config['illness_name']}")
    logger.info(f"Date Range: {processed_df['DateTime'].min()} to {processed_df['DateTime'].max()}")
    logger.info(f"Total Records: {len(processed_df):,}")
    logger.info(f"Total Features: {len(processed_df.columns):,}")
    logger.info(f"Regions: {processed_df['Region'].nunique()}")
    logger.info(f"Output: {output_file}")
    logger.info("")
    logger.info("✓ Environmental-only preprocessing complete!")
    logger.info("✓ Ready for training (pure environmental signal)")
    
if __name__ == "__main__":
    main()

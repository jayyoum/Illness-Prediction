#!/usr/bin/env python3
"""
Environmental-Only training for Chronic Rhinitis
Pure environmental exposure analysis
"""

import sys
from pathlib import Path

# Import the template training function
sys.path.insert(0, str(Path(__file__).parent))
from train_envonly_template import train_envonly_model

if __name__ == "__main__":
    train_envonly_model(
        config_filename="config_chronic_rhinitis_envonly.yaml",
        illness_short_name="chronic_rhinitis"
    )

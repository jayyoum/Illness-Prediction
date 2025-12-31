"""Plotting utilities for model evaluation."""

import matplotlib.pyplot as plt
import xgboost as xgb
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


def plot_predictions_vs_actual(
    y_true,
    y_pred,
    title: str = "Predictions vs. Actual Values",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 7))
    plt.plot(y_true.values if hasattr(y_true, 'values') else y_true,
             label='Actual Values', color='blue', alpha=0.8)
    plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Time Points (Test Set)', fontsize=12)
    plt.ylabel('Case Count', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved prediction plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    model: xgb.XGBRegressor,
    max_features: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance from XGBoost model.
    
    Args:
        model: Trained XGBoost model
        max_features: Maximum number of features to show
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    xgb.plot_importance(model, max_num_features=max_features, height=0.8)
    plt.title(title, fontsize=15)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_optuna_optimization_history(
    study,
    save_path: Optional[str] = None
) -> None:
    """
    Plot Optuna optimization history.
    
    Args:
        study: Optuna study object
        save_path: Path to save the plot
    """
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Saved Optuna optimization history to {save_path}")
        else:
            fig.show()
    except ImportError:
        logger.warning("Plotly not installed. Cannot generate Optuna plots.")
    except Exception as e:
        logger.warning(f"Could not generate Optuna plot: {e}")


def plot_optuna_param_importances(
    study,
    save_path: Optional[str] = None
) -> None:
    """
    Plot Optuna parameter importances.
    
    Args:
        study: Optuna study object
        save_path: Path to save the plot
    """
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Saved Optuna parameter importances to {save_path}")
        else:
            fig.show()
    except ImportError:
        logger.warning("Plotly not installed. Cannot generate Optuna plots.")
    except Exception as e:
        logger.warning(f"Could not generate Optuna plot: {e}")


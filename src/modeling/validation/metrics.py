"""
Performance metrics for SPX forecasting models.

Implements:
- Classification metrics (accuracy, precision, recall, F1)
- Regression metrics (RMSE, MAE, R²)
- Trading metrics (Sharpe, win rate, max drawdown)
- Directional accuracy
- Confusion matrix analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.core import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """Calculate and track model performance metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        logger.info("PerformanceMetrics initialized")
    
    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[list] = None,
    ) -> Dict[str, any]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            labels: Class labels (e.g., [0, 1, 2] for down/neutral/up)
            
        Returns:
            Dictionary of metrics
        """
        if labels is None:
            labels = list(range(len(np.unique(y_true))))
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # Per-class metrics
        for i, label in enumerate(labels):
            mask = y_true == label
            if mask.sum() > 0:
                metrics[f"accuracy_class_{label}"] = accuracy_score(y_true[mask], y_pred[mask])
                metrics[f"precision_class_{label}"] = precision_score(
                    y_true, y_pred, labels=[label], average=None, zero_division=0
                )[0]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = cm
        
        logger.debug("Classification metrics calculated", accuracy=metrics["accuracy"])
        
        return metrics
    
    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,  # Mean Absolute Percentage Error
        }
        
        # Directional accuracy (did we predict the sign correctly?)
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        metrics["directional_accuracy"] = direction_correct.mean()
        
        logger.debug("Regression metrics calculated", rmse=metrics["rmse"])
        
        return metrics
    
    def trading_metrics(
        self,
        returns: pd.Series,
        predictions: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        
        Args:
            returns: Series of returns (strategy or actual)
            predictions: Predicted signals (1 = long, -1 = short, 0 = neutral)
            risk_free_rate: Annual risk-free rate (default 0.0)
            
        Returns:
            Dictionary of trading metrics
        """
        # If predictions provided, calculate strategy returns
        if predictions is not None:
            strategy_returns = returns * predictions
        else:
            strategy_returns = returns
        
        # Basic stats
        total_return = (1 + strategy_returns).prod() - 1
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        
        # Sharpe ratio (annualized)
        # Assuming daily returns, multiply by sqrt(252)
        sharpe_ratio = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf
        
        metrics = {
            "total_return": total_return,
            "mean_return": mean_return,
            "std_return": std_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": total_trades,
            "num_winning": winning_trades,
            "num_losing": total_trades - winning_trades,
        }
        
        logger.debug(
            "Trading metrics calculated",
            sharpe=sharpe_ratio,
            total_return=total_return,
            win_rate=win_rate,
        )
        
        return metrics
    
    def directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate directional accuracy (did we predict up/down correctly?).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Threshold for neutral zone (default 0.0)
            
        Returns:
            Dictionary with directional metrics
        """
        # Convert to directional signals
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        # Overall directional accuracy
        direction_correct = true_direction == pred_direction
        overall_accuracy = direction_correct.mean()
        
        # Accuracy for up moves
        up_mask = true_direction > threshold
        up_accuracy = direction_correct[up_mask].mean() if up_mask.sum() > 0 else 0
        
        # Accuracy for down moves
        down_mask = true_direction < -threshold
        down_accuracy = direction_correct[down_mask].mean() if down_mask.sum() > 0 else 0
        
        # Accuracy for neutral
        neutral_mask = np.abs(true_direction) <= threshold
        neutral_accuracy = direction_correct[neutral_mask].mean() if neutral_mask.sum() > 0 else 0
        
        metrics = {
            "directional_accuracy": overall_accuracy,
            "up_accuracy": up_accuracy,
            "down_accuracy": down_accuracy,
            "neutral_accuracy": neutral_accuracy,
            "n_up": up_mask.sum(),
            "n_down": down_mask.sum(),
            "n_neutral": neutral_mask.sum(),
        }
        
        return metrics
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None,
    ):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Class names (e.g., ['down', 'neutral', 'up'])
        """
        if target_names is None:
            target_names = [f"class_{i}" for i in range(len(np.unique(y_true)))]
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=target_names, columns=target_names))


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Annualized Sharpe ratio
    """
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

"""
Evaluation metrics for SPX forecasting models.

Financial ML metrics:
- Classification metrics (accuracy, precision, recall, F1)
- Directional accuracy
- Profit & Loss simulation
- Sharpe ratio, max drawdown
- Confusion matrix analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    log_loss,
)

from src.core import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for financial forecasting."""
    
    def __init__(self, class_names: list = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: Names for classes (default: ['down', 'neutral', 'up'])
        """
        self.class_names = class_names or ['down', 'neutral', 'up']
        logger.info("ModelEvaluator initialized")
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = int(support[i])
        
        # Macro averages
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Log loss (if probabilities provided)
        if y_pred_proba is not None:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        logger.info(
            "Classification evaluation complete",
            accuracy=f"{metrics['accuracy']:.4f}",
            f1_macro=f"{metrics['f1_macro']:.4f}",
        )
        
        return metrics
    
    def evaluate_directional(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ignore_neutral: bool = True,
    ) -> Dict:
        """
        Directional accuracy (critical for trading).
        
        Measures: When we predict UP or DOWN, are we correct?
        Ignores NEUTRAL predictions by default.
        
        Args:
            y_true: True labels (0=down, 1=neutral, 2=up)
            y_pred: Predicted labels
            ignore_neutral: If True, ignore neutral predictions
            
        Returns:
            Directional accuracy metrics
        """
        metrics = {}
        
        if ignore_neutral:
            # Only consider non-neutral predictions
            mask = y_pred != 1  # Neutral is class 1
            y_true_dir = y_true[mask]
            y_pred_dir = y_pred[mask]
            
            if len(y_pred_dir) > 0:
                metrics['directional_accuracy'] = accuracy_score(y_true_dir, y_pred_dir)
                
                # Up/down specific
                up_mask = y_pred_dir == 2
                down_mask = y_pred_dir == 0
                
                if up_mask.sum() > 0:
                    metrics['up_accuracy'] = accuracy_score(
                        y_true_dir[up_mask], 
                        y_pred_dir[up_mask]
                    )
                    metrics['up_predictions'] = int(up_mask.sum())
                
                if down_mask.sum() > 0:
                    metrics['down_accuracy'] = accuracy_score(
                        y_true_dir[down_mask], 
                        y_pred_dir[down_mask]
                    )
                    metrics['down_predictions'] = int(down_mask.sum())
                
                metrics['non_neutral_predictions'] = int(len(y_pred_dir))
                metrics['neutral_predictions'] = int((y_pred == 1).sum())
            else:
                metrics['directional_accuracy'] = 0.0
                metrics['non_neutral_predictions'] = 0
        else:
            metrics['directional_accuracy'] = accuracy_score(y_true, y_pred)
        
        logger.info(
            "Directional evaluation complete",
            dir_accuracy=f"{metrics.get('directional_accuracy', 0):.4f}",
        )
        
        return metrics
    
    def evaluate_pnl(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray,
        position_size: float = 1.0,
        transaction_cost: float = 0.0001,
    ) -> Dict:
        """
        Simulate P&L from predictions.
        
        Simple strategy:
        - Predict UP (2): Long position
        - Predict DOWN (0): Short position  
        - Predict NEUTRAL (1): No position
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            returns: Actual forward returns
            position_size: Position size (default 1.0 = 100%)
            transaction_cost: Transaction cost per trade (default 1bp)
            
        Returns:
            P&L metrics
        """
        # Generate positions based on predictions
        # 2 (up) -> +1, 0 (down) -> -1, 1 (neutral) -> 0
        positions = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))
        positions = positions * position_size
        
        # Calculate returns
        strategy_returns = positions * returns
        
        # Subtract transaction costs
        trades = np.abs(np.diff(positions, prepend=0))
        costs = trades * transaction_cost
        strategy_returns = strategy_returns - costs
        
        # Cumulative returns
        cumulative_returns = np.cumsum(strategy_returns)
        
        # Metrics
        metrics = {
            'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            'mean_return': np.mean(strategy_returns),
            'std_return': np.std(strategy_returns),
            'sharpe_ratio': self._calculate_sharpe(strategy_returns),
            'max_drawdown': self._calculate_max_drawdown(cumulative_returns),
            'win_rate': (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0,
            'total_trades': int(trades.sum()),
            'cumulative_returns': cumulative_returns,
        }
        
        logger.info(
            "P&L evaluation complete",
            total_return=f"{metrics['total_return']:.4f}",
            sharpe=f"{metrics['sharpe_ratio']:.2f}",
        )
        
        return metrics
    
    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return float(sharpe)
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_dd = np.min(drawdown)
        
        return float(max_dd)
    
    def print_report(self, metrics: Dict):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)
        
        # Classification metrics
        if 'accuracy' in metrics:
            print("\n📊 CLASSIFICATION METRICS:")
            print(f"  Overall Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Macro F1 Score:      {metrics.get('f1_macro', 0):.4f}")
            
            if 'log_loss' in metrics:
                print(f"  Log Loss:            {metrics['log_loss']:.4f}")
            
            print("\n  Per-Class Performance:")
            for class_name in self.class_names:
                if f'f1_{class_name}' in metrics:
                    print(f"    {class_name.upper():8s} - " +
                          f"F1: {metrics[f'f1_{class_name}']:.4f}, " +
                          f"Precision: {metrics[f'precision_{class_name}']:.4f}, " +
                          f"Recall: {metrics[f'recall_{class_name}']:.4f}")
        
        # Directional metrics
        if 'directional_accuracy' in metrics:
            print("\n📈 DIRECTIONAL METRICS:")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.4f}")
            
            if 'up_accuracy' in metrics:
                print(f"  UP Predictions:       {metrics['up_predictions']} " +
                      f"(accuracy: {metrics['up_accuracy']:.4f})")
            
            if 'down_accuracy' in metrics:
                print(f"  DOWN Predictions:     {metrics['down_predictions']} " +
                      f"(accuracy: {metrics['down_accuracy']:.4f})")
        
        # P&L metrics
        if 'total_return' in metrics:
            print("\n💰 P&L SIMULATION:")
            print(f"  Total Return:        {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
            print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:        {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
            print(f"  Win Rate:            {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.1f}%)")
            print(f"  Total Trades:        {metrics['total_trades']}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            print("\n🔍 CONFUSION MATRIX:")
            cm = metrics['confusion_matrix']
            print("              Predicted")
            print("              ", "  ".join(f"{c:>6s}" for c in self.class_names))
            for i, true_class in enumerate(self.class_names):
                print(f"  Actual {true_class:>6s}", "  ".join(f"{cm[i, j]:6d}" for j in range(len(self.class_names))))
        
        print("\n" + "=" * 70 + "\n")


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    print_report: bool = True,
) -> Dict:
    """
    Quick model evaluation with all metrics.
    
    Examples:
        >>> from src.modeling.metrics import evaluate_model
        >>> metrics = evaluate_model(y_test, y_pred, y_pred_proba, forward_returns)
    """
    evaluator = ModelEvaluator()
    
    # Classification metrics
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
    
    # Directional metrics
    dir_metrics = evaluator.evaluate_directional(y_true, y_pred)
    metrics.update(dir_metrics)
    
    # P&L metrics (if returns provided)
    if returns is not None:
        pnl_metrics = evaluator.evaluate_pnl(y_true, y_pred, returns)
        metrics.update(pnl_metrics)
    
    if print_report:
        evaluator.print_report(metrics)
    
    return metrics

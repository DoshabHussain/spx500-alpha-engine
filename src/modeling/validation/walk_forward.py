"""
Walk-forward validation pipeline.

Implements complete walk-forward testing:
- Train on historical data
- Test on out-of-sample period
- Move forward in time
- Aggregate results across all folds
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from src.modeling.validation.time_series_split import TimeSeriesSplit, Split
from src.modeling.training.lgbm_trainer import LGBMTrainer
from src.modeling.validation.metrics import PerformanceMetrics
from src.core import get_logger

logger = get_logger(__name__)


class WalkForwardValidator:
    """Run walk-forward validation on time series data."""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_days: int = 30,
        mode: str = "expanding",
        gap_days: int = 1,
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of train/test splits
            test_size_days: Days in each test period
            mode: 'expanding' (growing train) or 'rolling' (fixed window)
            gap_days: Gap between train and test
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.mode = mode
        self.gap_days = gap_days
        
        self.splitter = TimeSeriesSplit(gap_days=gap_days)
        self.metrics_calc = PerformanceMetrics()
        
        self.results = []
        self.models = []
        
        logger.info(
            "WalkForwardValidator initialized",
            n_splits=n_splits,
            test_size_days=test_size_days,
            mode=mode,
        )
    
    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        model_params: Optional[Dict] = None,
        task: str = "multiclass",
        num_class: int = 3,
        save_models: bool = False,
        models_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Run complete walk-forward validation.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Label column name
            model_params: LightGBM parameters
            task: 'multiclass', 'binary', or 'regression'
            num_class: Number of classes for multiclass
            save_models: Whether to save trained models
            models_dir: Directory to save models
            
        Returns:
            DataFrame with results for each fold
        """
        logger.info(
            "Starting walk-forward validation",
            n_features=len(feature_cols),
            n_samples=len(df),
        )
        
        # Create splits
        splits = self.splitter.walk_forward_split(
            df,
            n_splits=self.n_splits,
            test_size_days=self.test_size_days,
            mode=self.mode,
        )
        
        if len(splits) == 0:
            raise ValueError("No valid splits created. Check your data and parameters.")
        
        self.results = []
        self.models = []
        
        # Run each fold
        for i, split in enumerate(splits):
            logger.info(f"Processing fold {i+1}/{len(splits)}: {split}")
            
            # Get train/test data
            X_train, X_test, y_train, y_test = self.splitter.get_train_test_data(
                df, split, feature_cols, label_col
            )
            
            # Train model
            trainer = LGBMTrainer(task=task, num_class=num_class, params=model_params)
            trainer.train(
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                verbose_eval=0,  # Quiet training
            )
            
            # Make predictions
            y_pred = trainer.predict(X_test, return_proba=False)
            
            # Calculate metrics
            if task in ["multiclass", "binary"]:
                metrics = self.metrics_calc.classification_metrics(
                    y_test.values,
                    y_pred,
                    labels=list(range(num_class)) if task == "multiclass" else [0, 1],
                )
            else:
                metrics = self.metrics_calc.regression_metrics(y_test.values, y_pred)
            
            # Store results
            fold_result = {
                "fold": i + 1,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "n_train": len(X_train),
                "n_test": len(X_test),
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
            
            self.results.append(fold_result)
            self.models.append(trainer)
            
            # Save model if requested
            if save_models and models_dir:
                models_dir = Path(models_dir)
                models_dir.mkdir(parents=True, exist_ok=True)
                model_path = models_dir / f"model_fold_{i+1}"
                trainer.save(model_path)
                logger.info(f"Saved model to {model_path}")
            
            logger.info(f"Fold {i+1} complete", metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        
        # Aggregate results
        results_df = pd.DataFrame(self.results)
        
        logger.info(
            "Walk-forward validation complete",
            n_folds=len(splits),
            mean_accuracy=results_df.get("accuracy", pd.Series([0])).mean(),
        )
        
        return results_df
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all folds.
        
        Returns:
            Dictionary with mean/std of metrics
        """
        if not self.results:
            raise ValueError("No results available. Run validation first.")
        
        results_df = pd.DataFrame(self.results)
        
        # Calculate mean and std for numeric columns
        summary = {}
        for col in results_df.columns:
            if pd.api.types.is_numeric_dtype(results_df[col]):
                summary[f"{col}_mean"] = results_df[col].mean()
                summary[f"{col}_std"] = results_df[col].std()
        
        return summary
    
    def get_best_model(self, metric: str = "accuracy") -> LGBMTrainer:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to optimize (e.g., 'accuracy', 'f1_macro', 'rmse')
            
        Returns:
            Best LGBMTrainer model
        """
        if not self.results:
            raise ValueError("No results available. Run validation first.")
        
        results_df = pd.DataFrame(self.results)
        
        # For metrics like RMSE, lower is better
        if metric in ["rmse", "mae"]:
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()
        
        logger.info(f"Best model: fold {best_idx + 1} with {metric}={results_df.loc[best_idx, metric]:.4f}")
        
        return self.models[best_idx]
    
    def plot_results(self) -> None:
        """Plot validation results across folds."""
        try:
            import matplotlib.pyplot as plt
            
            results_df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot accuracy over time
            if "accuracy" in results_df.columns:
                ax = axes[0, 0]
                ax.plot(results_df["fold"], results_df["accuracy"], marker='o')
                ax.set_xlabel("Fold")
                ax.set_ylabel("Accuracy")
                ax.set_title("Accuracy Across Folds")
                ax.grid(True)
            
            # Plot train/test sizes
            ax = axes[0, 1]
            ax.plot(results_df["fold"], results_df["n_train"], marker='o', label="Train")
            ax.plot(results_df["fold"], results_df["n_test"], marker='s', label="Test")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Sample Size")
            ax.set_title("Train/Test Sizes")
            ax.legend()
            ax.grid(True)
            
            # Plot precision/recall
            if "precision_macro" in results_df.columns:
                ax = axes[1, 0]
                ax.plot(results_df["fold"], results_df["precision_macro"], marker='o', label="Precision")
                ax.plot(results_df["fold"], results_df["recall_macro"], marker='s', label="Recall")
                ax.plot(results_df["fold"], results_df["f1_macro"], marker='^', label="F1")
                ax.set_xlabel("Fold")
                ax.set_ylabel("Score")
                ax.set_title("Precision/Recall/F1")
                ax.legend()
                ax.grid(True)
            
            # Plot test period timeline
            ax = axes[1, 1]
            for i, row in results_df.iterrows():
                ax.barh(i, (row["test_end"] - row["test_start"]).days, 
                       left=row["test_start"], height=0.8, alpha=0.6)
            ax.set_xlabel("Date")
            ax.set_ylabel("Fold")
            ax.set_title("Test Periods Timeline")
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not installed. Skipping plots.")


# ─── CONVENIENCE FUNCTION ───────────────────────────────────────────────────

def run_walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    n_splits: int = 5,
    test_size_days: int = 30,
) -> pd.DataFrame:
    """
    Quick function to run walk-forward validation.
    
    Examples:
        >>> results = run_walk_forward_validation(
        >>>     spy_df,
        >>>     feature_cols=['spy_ret_1m', 'spy_ema_5', 'spy_rsi_14'],
        >>>     label_col='label_30m',
        >>>     n_splits=5,
        >>> )
        >>> print(results)
    """
    validator = WalkForwardValidator(n_splits=n_splits, test_size_days=test_size_days)
    return validator.run(df, feature_cols, label_col)

"""
LightGBM model trainer for SPX forecasting.

Implements:
- Multi-class classification (up/down/neutral)
- Regression (continuous returns)
- Hyperparameter optimization
- Feature importance tracking
- Model persistence
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import json

from src.core import get_logger

logger = get_logger(__name__)


class LGBMTrainer:
    """Train and manage LightGBM models for SPX forecasting."""
    
    def __init__(
        self,
        task: str = "multiclass",
        num_class: int = 3,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LightGBM trainer.
        
        Args:
            task: 'multiclass', 'binary', or 'regression'
            num_class: Number of classes for multiclass (default 3: down/neutral/up)
            params: LightGBM parameters (uses defaults if None)
        """
        self.task = task
        self.num_class = num_class
        
        # Default parameters
        if params is None:
            params = self._get_default_params()
        
        self.params = params
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        logger.info(
            "LGBMTrainer initialized",
            task=task,
            num_class=num_class if task == "multiclass" else None,
        )
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters."""
        base_params = {
            "verbosity": -1,
            "seed": 42,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": -1,
        }
        
        if self.task == "multiclass":
            params = {
                **base_params,
                "objective": "multiclass",
                "num_class": self.num_class,
                "metric": "multi_logloss",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": 20,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
            }
        elif self.task == "binary":
            params = {
                **base_params,
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": 20,
            }
        elif self.task == "regression":
            params = {
                **base_params,
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": 20,
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return params
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 100,
    ) -> lgb.Booster:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            num_boost_round: Maximum boosting rounds
            early_stopping_rounds: Early stopping patience
            verbose_eval: Print metrics every N rounds
            
        Returns:
            Trained LightGBM booster
        """
        logger.info(
            "Training LightGBM model",
            X_train_shape=X_train.shape,
            y_train_shape=y_train.shape,
            has_validation=X_val is not None,
        )
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")
        
        # Train model
        callbacks = [
            lgb.log_evaluation(period=verbose_eval),
        ]
        
        if early_stopping_rounds > 0 and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        
        logger.info(
            "Training complete",
            best_iteration=self.model.best_iteration,
            best_score=self.model.best_score,
        )
        
        return self.model
    
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            return_proba: Return probabilities (for classification)
            
        Returns:
            Predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(X)
        
        if self.task == "multiclass":
            if return_proba:
                return predictions  # Already probabilities
            else:
                return np.argmax(predictions, axis=1)
        elif self.task == "binary":
            if return_proba:
                return predictions
            else:
                return (predictions > 0.5).astype(int)
        else:  # regression
            return predictions
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test, return_proba=False)
        
        if self.task in ["multiclass", "binary"]:
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            }
            
            # Add per-class metrics for multiclass
            if self.task == "multiclass":
                for i in range(self.num_class):
                    class_mask = y_test == i
                    if class_mask.sum() > 0:
                        metrics[f"accuracy_class_{i}"] = accuracy_score(
                            y_test[class_mask],
                            predictions[class_mask],
                        )
        
        else:  # regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "mae": mean_absolute_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
            }
        
        logger.info("Model evaluation", metrics=metrics)
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = filepath.with_suffix(".txt")
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            "task": self.task,
            "num_class": self.num_class,
            "params": self.params,
            "feature_names": self.feature_names,
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
        }
        
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = filepath.with_suffix(".importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        filepath = Path(filepath)
        
        # Load model
        model_path = filepath.with_suffix(".txt")
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.task = metadata["task"]
        self.num_class = metadata["num_class"]
        self.params = metadata["params"]
        self.feature_names = metadata["feature_names"]
        
        # Load feature importance
        importance_path = filepath.with_suffix(".importance.csv")
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)
        
        logger.info(f"Model loaded from {filepath}")


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def train_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_class: int = 3,
) -> LGBMTrainer:
    """
    Quick function to train a classification model.
    
    Examples:
        >>> trainer = train_classification_model(X_train, y_train, X_val, y_val)
        >>> predictions = trainer.predict(X_test)
    """
    trainer = LGBMTrainer(task="multiclass", num_class=num_class)
    trainer.train(X_train, y_train, X_val, y_val)
    return trainer


def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> LGBMTrainer:
    """
    Quick function to train a regression model.
    
    Examples:
        >>> trainer = train_regression_model(X_train, y_train, X_val, y_val)
        >>> predictions = trainer.predict(X_test)
    """
    trainer = LGBMTrainer(task="regression")
    trainer.train(X_train, y_train, X_val, y_val)
    return trainer

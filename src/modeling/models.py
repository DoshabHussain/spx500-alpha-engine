"""
ML models for SPX forecasting.

Implements:
- LightGBM (fast, accurate, handles categorical features)
- XGBoost (robust, good for time-series)
- Sklearn interface for consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from pathlib import Path
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.core import get_logger

logger = get_logger(__name__)


class SPXForecaster:
    """
    Base class for SPX forecasting models.
    
    Provides consistent interface across different ML libraries.
    """
    
    def __init__(self, model_type: str = "lightgbm", **kwargs):
        """
        Initialize forecaster.
        
        Args:
            model_type: 'lightgbm', 'xgboost', 'rf', or 'logistic'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.params = kwargs
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initializing {model_type} forecaster", params=kwargs)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **fit_kwargs,
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            **fit_kwargs: Additional training parameters
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        raise NotImplementedError("Subclasses must implement predict_proba()")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        raise NotImplementedError("Subclasses must implement get_feature_importance()")
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'SPXForecaster':
        """Load model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


class LightGBMForecaster(SPXForecaster):
    """LightGBM classifier for SPX forecasting."""
    
    def __init__(self, **kwargs):
        """
        Initialize LightGBM model.
        
        Common parameters:
            n_estimators: Number of boosting rounds (default 100)
            learning_rate: Learning rate (default 0.05)
            max_depth: Maximum tree depth (default -1, no limit)
            num_leaves: Max number of leaves (default 31)
            min_child_samples: Min samples in leaf (default 20)
            subsample: Row subsample ratio (default 1.0)
            colsample_bytree: Column subsample ratio (default 1.0)
            class_weight: 'balanced' or None (default None)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        super().__init__(model_type="lightgbm", **kwargs)
        
        # Default parameters
        default_params = {
            'objective': 'multiclass',
            'num_class': 3,  # 0=down, 1=neutral, 2=up
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': -1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
        }
        
        # Update with user params
        default_params.update(kwargs)
        self.params = default_params
        
        self.model = lgb.LGBMClassifier(**self.params)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **fit_kwargs,
    ):
        """Train LightGBM model with optional early stopping."""
        self.feature_names = X_train.columns.tolist()
        
        # Prepare validation set for early stopping
        eval_set = None
        callbacks = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
        
        logger.info(
            "Training LightGBM",
            train_samples=len(X_train),
            features=len(self.feature_names),
            val_samples=len(X_val) if X_val is not None else 0,
        )
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            **fit_kwargs,
        )
        
        self.is_trained = True
        logger.info("LightGBM training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        
        return importance


class XGBoostForecaster(SPXForecaster):
    """XGBoost classifier for SPX forecasting."""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model.
        
        Common parameters:
            n_estimators: Number of boosting rounds (default 100)
            learning_rate: Learning rate (default 0.05)
            max_depth: Maximum tree depth (default 6)
            min_child_weight: Min sum of weights in child (default 1)
            subsample: Row subsample ratio (default 0.8)
            colsample_bytree: Column subsample ratio (default 0.8)
            gamma: Min loss reduction for split (default 0)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        super().__init__(model_type="xgboost", **kwargs)
        
        # Default parameters
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'random_state': 42,
        }
        
        # Update with user params
        default_params.update(kwargs)
        self.params = default_params
        
        self.model = xgb.XGBClassifier(**self.params)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **fit_kwargs,
    ):
        """Train XGBoost model with optional early stopping."""
        self.feature_names = X_train.columns.tolist()
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            fit_kwargs['early_stopping_rounds'] = fit_kwargs.get('early_stopping_rounds', 10)
            fit_kwargs['verbose'] = False
        
        logger.info(
            "Training XGBoost",
            train_samples=len(X_train),
            features=len(self.feature_names),
        )
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            **fit_kwargs,
        )
        
        self.is_trained = True
        logger.info("XGBoost training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        
        return importance


# ─── MODEL FACTORY ──────────────────────────────────────────────────────────

def create_model(model_type: str = "lightgbm", **kwargs) -> SPXForecaster:
    """
    Factory function to create models.
    
    Args:
        model_type: 'lightgbm', 'xgboost'
        **kwargs: Model parameters
        
    Returns:
        Initialized model
        
    Examples:
        >>> model = create_model('lightgbm', n_estimators=200, learning_rate=0.03)
        >>> model = create_model('xgboost', max_depth=8)
    """
    if model_type == "lightgbm":
        return LightGBMForecaster(**kwargs)
    elif model_type == "xgboost":
        return XGBoostForecaster(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

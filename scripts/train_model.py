"""
Complete model training pipeline for SPX forecasting.

Usage: python scripts/train_model.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage import get_raw_store
from src.processing.features import (
    calculate_all_returns_features,
    calculate_all_technical_features,
    calculate_all_regime_features,
)
from src.processing.labels import calculate_all_labels
from src.modeling.dataset import prepare_dataset, select_features
from src.modeling.models import create_model
from src.modeling.metrics import evaluate_model


def main():
    """Run complete training pipeline."""
    print("=" * 70)
    print("🤖 SPX500 Model Training - Phase 4")
    print("=" * 70)
    print()
    
    # ─── STEP 1: Load Data ──────────────────────────────────────────────────
    print("📊 Step 1: Loading data...")
    store = get_raw_store()
    
    try:
        # Try to load SPY data (adjust dates to match your data)
        spy_df = store.read_underlying("SPY", datetime(2026, 2, 1), datetime(2026, 3, 11))
        
        if spy_df.empty:
            print("❌ No SPY data found.")
            print("   Creating sample data for demonstration...")
            spy_df = create_sample_data()
        else:
            print(f"  ✓ Loaded {len(spy_df)} bars from SPY")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Creating sample data for demonstration...")
        spy_df = create_sample_data()
    
    print()
    
    # ─── STEP 2: Feature Engineering ────────────────────────────────────────
    print("🔧 Step 2: Generating features...")
    
    try:
        spy_df = calculate_all_returns_features(spy_df)
        print("  ✓ Returns features")
        
        spy_df = calculate_all_technical_features(spy_df)
        print("  ✓ Technical indicators")
        
        spy_df = calculate_all_regime_features(spy_df)
        print("  ✓ Regime features")
        
        spy_df = calculate_all_labels(spy_df)
        print("  ✓ Labels (30m, 1h, 6h)")
    except Exception as e:
        print(f"  ✗ Error generating features: {e}")
        return
    
    print()
    
    # ─── STEP 3: Dataset Preparation ────────────────────────────────────────
    print("📦 Step 3: Preparing dataset...")
    
    # Select label
    label_col = "label_30m"
    
    # Auto-select features (exclude labels, timestamps, etc.)
    feature_cols = select_features(spy_df, label_col)
    print(f"  ✓ Selected {len(feature_cols)} features")
    
    # Split data
    from src.modeling.dataset import TimeSeriesDataset
    
    dataset = TimeSeriesDataset(
        spy_df,
        feature_cols=feature_cols,
        label_col=label_col,
    )
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.get_splits(
        test_size=0.2,
        validation_size=0.1,
        gap_days=0,  # Adjust based on your data frequency
    )
    
    print(f"  ✓ Train: {len(X_train)} samples")
    print(f"  ✓ Val:   {len(X_val)} samples")
    print(f"  ✓ Test:  {len(X_test)} samples")
    print()
    
    # ─── STEP 4: Model Training ─────────────────────────────────────────────
    print("🎯 Step 4: Training model...")
    
    # Create model (try both LightGBM and XGBoost)
    models_to_try = {
        "LightGBM": create_model(
            'lightgbm',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
        ),
        # Uncomment to try XGBoost
        # "XGBoost": create_model(
        #     'xgboost',
        #     n_estimators=100,
        #     learning_rate=0.05,
        #     max_depth=5,
        # ),
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models_to_try.items():
        print(f"\n  Training {name}...")
        
        try:
            # Train
            model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)
            
            from src.modeling.metrics import ModelEvaluator
            evaluator = ModelEvaluator()
            val_metrics = evaluator.evaluate_classification(y_val, y_val_pred, y_val_proba)
            
            print(f"    Validation Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"    Validation F1:       {val_metrics['f1_macro']:.4f}")
            
            # Track best model
            if val_metrics['accuracy'] > best_score:
                best_score = val_metrics['accuracy']
                best_model = model
                best_name = name
        
        except Exception as e:
            print(f"    ✗ Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    if best_model is None:
        print("❌ No models trained successfully")
        return
    
    print(f"\n  ✓ Best model: {best_name} (accuracy: {best_score:.4f})")
    print()
    
    # ─── STEP 5: Test Set Evaluation ────────────────────────────────────────
    print("📈 Step 5: Evaluating on test set...")
    
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    
    # Get forward returns for P&L simulation
    test_indices = X_test.index
    forward_returns = spy_df.loc[test_indices, 'forward_ret_30m'].values
    
    # Comprehensive evaluation
    test_metrics = evaluate_model(
        y_test,
        y_test_pred,
        y_test_proba,
        forward_returns,
        print_report=True,
    )
    
    # ─── STEP 6: Feature Importance ─────────────────────────────────────────
    print("📊 Step 6: Feature importance (Top 10)...")
    
    importance = best_model.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    print()
    
    # ─── STEP 7: Save Model ─────────────────────────────────────────────────
    print("💾 Step 7: Saving model...")
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"spx_forecaster_{best_name.lower()}_{timestamp}.pkl"
    
    best_model.save(str(model_path))
    print(f"  ✓ Model saved to {model_path}")
    
    # Save feature importance
    importance_path = model_dir / f"feature_importance_{timestamp}.csv"
    importance.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved to {importance_path}")
    print()
    
    # ─── Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"\n📊 Final Test Metrics:")
    print(f"  Accuracy:            {test_metrics['accuracy']:.4f}")
    print(f"  Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.4f}")
    print(f"  Sharpe Ratio:        {test_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"\n💡 Next steps:")
    print(f"  1. Review feature importance in {importance_path}")
    print(f"  2. Tune hyperparameters for better performance")
    print(f"  3. Run walk-forward validation")
    print(f"  4. Deploy model for live predictions")
    print()


def create_sample_data() -> pd.DataFrame:
    """Create sample SPY data for demonstration."""
    dates = pd.bdate_range(start='2026-02-09', end='2026-03-10', freq='B')
    np.random.seed(42)
    
    spy_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'SPY',
        'open': 550 + np.random.randn(len(dates)).cumsum() * 2,
        'close': 550 + np.random.randn(len(dates)).cumsum() * 2,
        'volume': np.random.randint(50000000, 100000000, len(dates)),
        'interval': '1d'
    })
    
    spy_data['high'] = spy_data[['open', 'close']].max(axis=1) * 1.005
    spy_data['low'] = spy_data[['open', 'close']].min(axis=1) * 0.995
    
    print(f"  ✓ Created {len(spy_data)} days of sample SPY data")
    return spy_data


if __name__ == "__main__":
    main()

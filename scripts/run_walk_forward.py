"""
Run walk-forward validation on SPX forecasting model.

Usage: python scripts/run_walk_forward.py
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage import get_raw_store
from src.processing.features import (
    calculate_all_returns_features,
    calculate_all_technical_features,
    calculate_all_regime_features,
)
from src.processing.labels import calculate_all_labels
from src.modeling.validation import WalkForwardValidator


def main():
    """Run walk-forward validation."""
    print("=" * 70)
    print("🔄 SPX500 Walk-Forward Validation")
    print("=" * 70)
    print()
    
    # Load data
    print("📊 Loading data...")
    store = get_raw_store()
    spy_df = store.read_underlying("SPY", datetime(2026, 2, 1), datetime(2026, 3, 10))
    
    if spy_df.empty:
        print("❌ No data found. Run ingestion first.")
        return
    
    print(f"  ✓ Loaded {len(spy_df)} bars")
    print()
    
    # Calculate features
    print("🔧 Calculating features...")
    spy_df = calculate_all_returns_features(spy_df)
    spy_df = calculate_all_technical_features(spy_df)
    spy_df = calculate_all_regime_features(spy_df)
    spy_df = calculate_all_labels(spy_df)
    print("  ✓ Features calculated")
    print()
    
    # Select features
    feature_cols = [
        col for col in spy_df.columns
        if any(x in col for x in ['_ret_', '_ema_', '_rsi_', '_macd', '_atr_', '_bb_'])
        and 'forward' not in col and 'label' not in col
    ]
    
    # Remove features with too many NaNs
    feature_cols = [col for col in feature_cols if spy_df[col].isna().sum() / len(spy_df) < 0.5]
    
    label_col = "label_30m"
    
    print(f"Selected {len(feature_cols)} features")
    print()
    
    # Clean data
    spy_clean = spy_df.dropna(subset=[label_col] + feature_cols)
    print(f"Clean dataset: {len(spy_clean)} samples")
    print()
    
    if len(spy_clean) < 100:
        print("❌ Not enough data for walk-forward validation")
        print(f"   Need at least 100 samples, got {len(spy_clean)}")
        return
    
    # Run validation
    print("=" * 70)
    print("🚀 Running Walk-Forward Validation")
    print("=" * 70)
    print()
    
    validator = WalkForwardValidator(
        n_splits=3,
        test_size_days=7,
        mode="expanding",
        gap_days=1,
    )
    
    results_df = validator.run(
        spy_clean,
        feature_cols,
        label_col,
        task="multiclass",
        num_class=3,
        save_models=True,
        models_dir=Path("models/walk_forward"),
    )
    
    # Display results
    print()
    print("=" * 70)
    print("📊 Validation Results")
    print("=" * 70)
    print()
    
    print(results_df[['fold', 'n_train', 'n_test', 'accuracy', 'f1_weighted']].to_string(index=False))
    print()
    
    # Summary
    summary = validator.get_summary()
    
    print("=" * 70)
    print("📈 Summary Statistics")
    print("=" * 70)
    print()
    
    print(f"Mean Accuracy:  {summary.get('accuracy_mean', 0):.4f} ± {summary.get('accuracy_std', 0):.4f}")
    print(f"Mean F1 Score:  {summary.get('f1_weighted_mean', 0):.4f} ± {summary.get('f1_weighted_std', 0):.4f}")
    print(f"Mean Precision: {summary.get('precision_weighted_mean', 0):.4f} ± {summary.get('precision_weighted_std', 0):.4f}")
    print(f"Mean Recall:    {summary.get('recall_weighted_mean', 0):.4f} ± {summary.get('recall_weighted_std', 0):.4f}")
    print()
    
    # Save results
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    
    results_file = results_path / "walk_forward_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"✅ Results saved to {results_file}")
    print()
    
    print("=" * 70)
    print("✅ Walk-Forward Validation Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

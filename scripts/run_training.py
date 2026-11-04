"""
Main training pipeline for SPX Intraday Forecasting Models (Stage 4).
Uses Purged Walk-Forward Cross Validation and Multi-Model Ensemble.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage import get_raw_store
from src.processing.features import (
    calculate_all_returns_features,
    calculate_all_technical_features,
    calculate_all_regime_features,
    compute_microstructure_alpha,
)
from src.processing.labels import calculate_all_labels
from src.modeling.purged_trainer import SPXEnsembleForecaster, PurgedWalkForwardValidator


def prepare_data() -> pd.DataFrame:
    """Load canonical 5m data and calculate full feature matrix."""
    print("=" * 70)
    print("[*] Stage 2 & 3: Data Loading & Feature Pipeline")
    print("=" * 70)

    store = get_raw_store()
    spy_df = store.read_underlying("SPY", datetime(2025, 12, 1), datetime(2026, 3, 1))

    if spy_df.empty:
        print("[!] No SPY data found. Generating canonical 5m dataset...")
        from scripts.create_sample_data import main as generate_data
        generate_data()
        spy_df = store.read_underlying("SPY", datetime(2025, 12, 1), datetime(2026, 3, 1))

    print(f"  [+] Loaded {len(spy_df)} 5-minute bars from {spy_df['timestamp'].min()} to {spy_df['timestamp'].max()}")

    # Calculate returns, technicals, regime, and microstructure
    print("Calculating technical & returns alpha...")
    spy_df = calculate_all_returns_features(spy_df)
    spy_df = calculate_all_technical_features(spy_df)
    spy_df = calculate_all_regime_features(spy_df)

    print("Calculating options microstructure & Dealer GEX features...")
    spy_df = compute_microstructure_alpha(spy_df)

    # Calculate forward-looking labels (30m & 1h)
    print("Calculating forward return targets (30m, 1h, 6h)...")
    spy_df = calculate_all_labels(spy_df)

    return spy_df


def select_features(df: pd.DataFrame) -> list:
    """Select non-leaking predictive features."""
    feature_cols = [
        col for col in df.columns
        if any(x in col for x in [
            '_ret_', '_ema_', '_rsi_', '_macd', '_atr_', '_bb_',
            '_obv', 'session_', 'is_', 'gex', 'breadth', 'vix'
        ])
        and 'forward' not in col
        and 'label' not in col
    ]

    valid = []
    for c in feature_cols:
        if df[c].isna().sum() / len(df) < 0.20:
            valid.append(c)

    return sorted(valid)


def main():
    print()
    print("=" * 70)
    print("[*] Stage 4: Purged Walk-Forward ML Ensemble Training")
    print("=" * 70)
    print()

    df = prepare_data()
    feature_cols = select_features(df)
    target_col = "label_30m"

    print(f"Selected {len(feature_cols)} features for model training.")

    # Clean dataset
    df_clean = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    print(f"Clean modeling dataset: {len(df_clean)} samples.")

    X = df_clean[feature_cols].values
    y = df_clean[target_col].astype(int).values

    print("\nTarget Class Distribution (0=Down, 1=Neutral, 2=Up):")
    unique, counts = np.unique(y, return_counts=True)
    for u, cnt in zip(unique, counts):
        pct = (cnt / len(y)) * 100
        print(f"  Class {u}: {cnt} bars ({pct:.1f}%)")

    # Train with Purged Walk-Forward Cross Validation
    print("\nExecuting Purged Walk-Forward Cross-Validation (5 Folds, Embargo=2%)...")
    forecaster = SPXEnsembleForecaster(random_state=42)
    metrics = forecaster.train_with_purging(
        X, y,
        feature_names=feature_cols,
        n_splits=5,
        embargo_pct=0.02,
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] Cross-Validation Performance Summary")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:22s}: {v}")

    # Top Feature Importance
    df_imp = forecaster.get_feature_importances()
    print("\n[*] Top 10 Predictive Features:")
    for idx, row in df_imp.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:28s} : {row['importance']:.4f}")

    # Save artifact
    models_dir = Path("runs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "spx_ensemble_forecaster.joblib"
    joblib.dump(forecaster, model_path)
    print(f"\n[+] Trained ensemble saved to: {model_path}")

    # Test single-bar inference
    sample_features = X[-1]
    pred = forecaster.predict_intraday(sample_features)
    print("\n[*] Sample Intraday Live Inference:")
    print(f"  P(Down)    : {pred['p_down']*100:.1f}%")
    print(f"  P(Neutral) : {pred['p_flat']*100:.1f}%")
    print(f"  P(Up)      : {pred['p_up']*100:.1f}%")
    print(f"  Signal     : {pred['signal']}")
    print(f"  Strategy   : {pred['alpha_recommendation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

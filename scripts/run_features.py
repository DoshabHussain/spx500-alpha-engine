"""
Generate features from raw data - Phase 3 Pipeline.

Usage: python scripts/run_features.py
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage import get_raw_store
from src.processing.features import (
    calculate_all_returns_features,
    calculate_all_technical_features,
    calculate_all_volatility_features,
    calculate_all_regime_features,
)
from src.processing.labels import calculate_all_labels


def main():
    """Run complete feature engineering pipeline."""
    print("=" * 70)
    print("[*] SPX500 Feature Engineering - Phase 3 & Microstructure GEX")
    print("=" * 70)
    print()
    
    store = get_raw_store()
    
    # Read SPY data
    print("[*] Step 1: Loading SPY data...")
    try:
        spy_df = store.read_underlying("SPY", datetime(2025, 12, 1), datetime(2026, 3, 1))
        
        if spy_df.empty:
            print("[!] No SPY data found. Generating 5m dataset...")
            from scripts.create_sample_data import main as generate_data
            generate_data()
            spy_df = store.read_underlying("SPY", datetime(2025, 12, 1), datetime(2026, 3, 1))
        
        print(f"  [+] Loaded {len(spy_df)} bars from {spy_df['timestamp'].min()} to {spy_df['timestamp'].max()}")
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return
    
    print()
    
    # Calculate returns features
    print("[*] Step 2: Calculating returns features...")
    try:
        spy_df = calculate_all_returns_features(spy_df)
        print("  [+] Returns, ATR, Bollinger Bands")
    except Exception as e:
        print(f"  [x] Error: {e}")
    
    # Calculate technical features
    print("[*] Step 3: Calculating technical indicators...")
    try:
        spy_df = calculate_all_technical_features(spy_df)
        print("  [+] EMAs, MACD, RSI, OBV")
    except Exception as e:
        print(f"  [x] Error: {e}")
    
    # Calculate regime and microstructure features
    print("[*] Step 4: Calculating regime & Dealer GEX features...")
    try:
        from src.processing.features.microstructure_gex import compute_microstructure_alpha
        spy_df = calculate_all_regime_features(spy_df)
        spy_df = compute_microstructure_alpha(spy_df)
        print("  [+] Session flags, Net GEX, 0DTE flow ratio, market breadth ($TICK, $ADD)")
    except Exception as e:
        print(f"  [x] Error: {e}")
    
    # Calculate labels
    print("[*] Step 5: Calculating labels (30m, 1h, 6h horizons)...")
    try:
        spy_df = calculate_all_labels(spy_df)
        print("  [+] Forward returns, directional labels (0=Down, 1=Neutral, 2=Up)")
    except Exception as e:
        print(f"  [x] Error: {e}")
    
    print()
    print("=" * 70)
    print("[SUCCESS] Feature engineering complete!")
    print("=" * 70)
    
    # Summary
    feature_cols = [col for col in spy_df.columns if any(x in col for x in 
        ['_ret_', '_atr_', '_bb_', '_ema_', '_macd', '_rsi_', '_obv', 
         'session_', 'is_', 'label_', 'forward_', 'gex', 'breadth'])]
    
    print(f"\n[*] Summary:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Total rows: {len(spy_df)}")
    print(f"  Valid labels (30m): {spy_df['label_30m'].notna().sum() if 'label_30m' in spy_df.columns else 0}")
    
    # Show sample features
    print("\n[*] Sample features:")
    for col in sorted(feature_cols)[:15]:
        non_null = spy_df[col].notna().sum()
        print(f"  - {col:30s} : {non_null:4d} non-null values")
    
    if len(feature_cols) > 15:
        print(f"  ... and {len(feature_cols) - 15} more features")
    
    print("\n[*] Ready for Phase 4 Model Training: python scripts/run_training.py")
    print()


if __name__ == "__main__":
    main()

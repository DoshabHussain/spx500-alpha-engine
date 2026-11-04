"""
Windows-compatible data ingestion script.

Usage:
    python scripts/run_ingest.py
    
This replaces run_ingest.sh for Windows users.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.batch import ingest_yfinance_latest, ingest_fred_latest


def main():
    """Run daily data ingestion pipeline."""
    print("=" * 60)
    print("🚀 SPX500 Data Ingestion - Starting...")
    print("=" * 60)
    print()
    
    # ─── Ingest yfinance data ───────────────────────────────────────────────
    print("📊 PHASE 1: Ingesting yfinance data (SPY, QQQ, VIX)")
    print("-" * 60)
    
    try:
        yf_data = ingest_yfinance_latest()
        
        if yf_data:
            for symbol, df in yf_data.items():
                if len(df) > 0:
                    print(f"  ✓ {symbol:6s} : {len(df):4d} rows | {df.index.min()} to {df.index.max()}")
                else:
                    print(f"  ⚠ {symbol:6s} : No data returned")
            print(f"\n  Total: {len(yf_data)} symbols ingested")
        else:
            print("  ✗ No yfinance data returned")
            
    except Exception as e:
        print(f"  ✗ yfinance ingestion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # ─── Ingest FRED data ───────────────────────────────────────────────────
    print("📈 PHASE 2: Ingesting FRED macro data (DGS10, DGS2, etc.)")
    print("-" * 60)
    
    try:
        fred_data = ingest_fred_latest()
        
        if fred_data:
            for series_id, df in fred_data.items():
                if len(df) > 0:
                    latest_value = df['value'].iloc[-1] if 'value' in df.columns else 'N/A'
                    print(f"  ✓ {series_id:8s} : {len(df):4d} rows | Latest: {latest_value}")
                else:
                    print(f"  ⚠ {series_id:8s} : No data returned")
            print(f"\n  Total: {len(fred_data)} series ingested")
        else:
            print("  ✗ No FRED data returned")
            
    except Exception as e:
        print(f"  ✗ FRED ingestion failed: {e}")
        print()
        print("  💡 Common issues:")
        print("     - Missing FRED_API_KEY in .env file")
        print("     - Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("✅ Data ingestion complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Check data: dir data\\raw -Recurse -File")
    print("  2. Verify files exist in data/raw/underlying/, data/raw/indices/, data/raw/macro/")
    print()


if __name__ == "__main__":
    main()
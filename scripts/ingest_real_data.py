"""
Real Market Data Ingestion Script.
Downloads 60 days of real 5-minute SPY, ^GSPC (SPX), ^VIX, ^TNX, DX-Y.NYB bars
from Yahoo Finance Chart v8 API plus real SPX/SPY options chains from Cboe Delayed API,
caching them locally under data/real/ for leak-free training & offline execution.

Usage:
    python scripts/ingest_real_data.py
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.real_market_feed import ingest_and_cache_real_dataset


def main():
    print("=" * 72)
    print("  REAL MARKET & CBOE GEX DATA INGESTION (Yahoo Chart v8 + Cboe)")
    print("=" * 72)
    manifest = ingest_and_cache_real_dataset()
    print(f"  [+] Fetched at UTC : {manifest['fetched_at_utc']}")
    for sym, info in manifest["symbols"].items():
        print(f"  [+] {sym:10s} : {info['bars']:5d} bars | Spot={info['spot']}")
    for opt_sym, info in manifest.get("cboe_gex", {}).items():
        print(
            f"  [+] Cboe {opt_sym:5s} : Spot={info['spot']} | "
            f"Net GEX={info['net_gex_total_b']:+.3f}B | Zero-GEX={info['zero_gex_pivot']} | "
            f"Contracts={info['contracts_parsed']}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()

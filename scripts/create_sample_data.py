"""
Create comprehensive 5-minute canonical SPX/SPY intraday data, options chain flow,
and macroeconomic series for Stages 1-7.

Generates:
- 60 trading days x 79 5-min bars (09:30 - 16:00 ET) = 4,740 canonical bars
- Strike-level options chains for GEX calculation
- VIX and VIXW term structure data
- FRED macro rates (DGS10, DGS2, T10Y2Y)
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd
import pytz

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage import get_raw_store

NY_TZ = pytz.timezone("America/New_York")


def generate_intraday_5m_data(n_days: int = 60, start_date: datetime = datetime(2025, 12, 1)) -> pd.DataFrame:
    """
    Generate realistic 5-minute intraday bars (09:30 to 16:00 ET) for SPY.
    Includes intraday volatility smile (U-shape), volume clustering, and realistic drift.
    """
    np.random.seed(42)
    business_days = pd.bdate_range(start=start_date, periods=n_days, freq="B")
    
    records = []
    current_close = 502.50
    
    for day in business_days:
        day_date = day.date()
        # Daily overnight gap
        overnight_return = np.random.normal(0.0003, 0.0035)
        current_close = current_close * (1 + overnight_return)
        
        # 5-minute time intervals from 09:30 to 16:00 inclusive (79 bars)
        times = [
            datetime.combine(day_date, time(9, 30)) + timedelta(minutes=5 * i)
            for i in range(79)
        ]
        
        cum_vol = 0
        cum_val = 0.0
        
        for idx, bar_dt in enumerate(times):
            # U-shaped intraday volatility multiplier
            t_norm = idx / 78.0  # 0 to 1
            vol_multiplier = 1.0 + 1.8 * ((t_norm - 0.5) ** 2)
            sigma = 0.0010 * vol_multiplier
            
            bar_ret = np.random.normal(0.00005, sigma)
            bar_open = current_close
            bar_close = bar_open * (1 + bar_ret)
            
            high_push = abs(np.random.normal(0, sigma * 0.8))
            low_pull = abs(np.random.normal(0, sigma * 0.8))
            bar_high = max(bar_open, bar_close) * (1 + high_push)
            bar_low = min(bar_open, bar_close) * (1 - low_pull)
            
            # Volume profile: heavy at open and close
            base_vol = 80000
            time_vol_factor = 1.0 + 2.5 * ((t_norm - 0.5) ** 2)
            bar_vol = int(base_vol * time_vol_factor * np.random.uniform(0.7, 1.4))
            
            typical_price = (bar_high + bar_low + bar_close) / 3.0
            cum_vol += bar_vol
            cum_val += typical_price * bar_vol
            vwap = cum_val / max(cum_vol, 1)
            
            # Timezone localized timestamp
            ts_aware = NY_TZ.localize(bar_dt)
            
            records.append({
                "timestamp": ts_aware,
                "symbol": "SPY",
                "open": round(bar_open, 2),
                "high": round(bar_high, 2),
                "low": round(bar_low, 2),
                "close": round(bar_close, 2),
                "volume": bar_vol,
                "vwap": round(vwap, 2),
                "interval": "5min",
            })
            
            current_close = bar_close
            
    df = pd.DataFrame(records)
    return df


def generate_options_chain_data(underlying_df: pd.DataFrame) -> dict:
    """
    Generate options chain snapshots for each day with strike-level Call/Put GEX parameters.
    """
    chains_by_day = {}
    
    for dt, group in underlying_df.groupby(underlying_df["timestamp"].dt.date):
        spot = group["close"].iloc[-1]
        rounded_spot = round(spot / 5.0) * 5.0
        strikes = np.arange(rounded_spot - 50, rounded_spot + 55, 5)
        
        rows = []
        for strike in strikes:
            moneyness = (strike - spot) / spot
            d1 = -moneyness / 0.15
            gamma = max(0.005, round(0.04 * np.exp(-0.5 * (d1 ** 2)), 4))
            
            # Call side
            call_oi = int(np.random.randint(500, 15000) * np.exp(-abs(moneyness) * 5))
            call_vol = int(call_oi * np.random.uniform(0.1, 0.6))
            rows.append({
                "strike": float(strike),
                "type": "CALL",
                "open_interest": call_oi,
                "volume": call_vol,
                "gamma": gamma,
                "spot": spot,
                "bid": max(0.05, round(max(0.0, spot - strike) + 1.2, 2)),
                "ask": max(0.08, round(max(0.0, spot - strike) + 1.3, 2)),
                "iv": round(0.14 + abs(moneyness) * 0.1, 3),
            })
            
            # Put side
            put_oi = int(np.random.randint(500, 15000) * np.exp(-abs(moneyness) * 5))
            put_vol = int(put_oi * np.random.uniform(0.1, 0.6))
            rows.append({
                "strike": float(strike),
                "type": "PUT",
                "open_interest": put_oi,
                "volume": put_vol,
                "gamma": gamma,
                "spot": spot,
                "bid": max(0.05, round(max(0.0, strike - spot) + 1.2, 2)),
                "ask": max(0.08, round(max(0.0, strike - spot) + 1.3, 2)),
                "iv": round(0.16 + abs(moneyness) * 0.12, 3),
            })
            
        chains_by_day[dt] = pd.DataFrame(rows)
        
    return chains_by_day


def generate_macro_indices_data(underlying_df: pd.DataFrame) -> tuple:
    """
    Generate VIX index series and FRED macro rates aligned to trading dates.
    """
    unique_dates = underlying_df["timestamp"].dt.date.unique()
    
    vix_records = []
    macro_dgs10 = []
    macro_dgs2 = []
    macro_t10y2y = []
    
    np.random.seed(101)
    base_vix = 15.5
    base_dgs10 = 4.25
    base_dgs2 = 4.40
    
    for dt in unique_dates:
        base_vix += np.random.normal(0, 0.3)
        base_vix = max(11.0, min(35.0, base_vix))
        
        base_dgs10 += np.random.normal(0, 0.02)
        base_dgs2 += np.random.normal(0, 0.02)
        spread = base_dgs10 - base_dgs2
        
        date_dt = datetime.combine(dt, time(16, 0))
        ts_aware = NY_TZ.localize(date_dt)
        
        vix_records.append({"timestamp": ts_aware, "close": round(base_vix, 2), "symbol": "VIX"})
        macro_dgs10.append({"timestamp": ts_aware, "value": round(base_dgs10, 3), "series": "DGS10"})
        macro_dgs2.append({"timestamp": ts_aware, "value": round(base_dgs2, 3), "series": "DGS2"})
        macro_t10y2y.append({"timestamp": ts_aware, "value": round(spread, 3), "series": "T10Y2Y"})
        
    return (
        pd.DataFrame(vix_records),
        pd.DataFrame(macro_dgs10),
        pd.DataFrame(macro_dgs2),
        pd.DataFrame(macro_t10y2y),
    )


def main():
    print("=" * 70)
    print("[*] Generating 5-Minute Intraday Canonical Dataset for SPX Alpha Engine")
    print("=" * 70)
    
    store = get_raw_store()
    
    # 1. 5-minute bars
    print("Generating 60 days of 5-minute SPY bars...")
    spy_df = generate_intraday_5m_data(n_days=60)
    print(f"  [+] Generated {len(spy_df)} total 5-minute bars across 60 trading days")
    print(f"  [+] Date range: {spy_df['timestamp'].min()} to {spy_df['timestamp'].max()}")
    print(f"  [+] Price range: ${spy_df['close'].min():.2f} to ${spy_df['close'].max():.2f}")
    
    # Save underlying bars
    print("Saving 5-minute bars to partitioned Parquet storage...")
    saved_count = 0
    for day_date, group in spy_df.groupby(spy_df["timestamp"].dt.date):
        dt_val = datetime.combine(day_date, datetime.min.time())
        store.write_underlying(group.reset_index(drop=True), "SPY", dt_val)
        saved_count += 1
    print(f"  [+] Saved underlying SPY 5m bars across {saved_count} partitions.")
    
    # 2. Options chain
    print("Generating strike-level options chains for Net GEX calculation...")
    options_chains = generate_options_chain_data(spy_df)
    for day_date, chain_df in options_chains.items():
        dt_val = datetime.combine(day_date, datetime.min.time())
        store.write_options(chain_df, "SPY", dt_val)
    print(f"  [+] Saved options chains for {len(options_chains)} dates.")
    
    # 3. Macro and Indices
    print("Generating VIX and FRED macro yield curve series...")
    vix_df, dgs10_df, dgs2_df, t10y2y_df = generate_macro_indices_data(spy_df)
    
    for row in vix_df.itertuples():
        day_date = row.timestamp.date()
        dt_val = datetime.combine(day_date, datetime.min.time())
        store.write_indices(pd.DataFrame([{"timestamp": row.timestamp, "close": row.close, "symbol": "VIX"}]), "VIX", dt_val)
        
    for row in dgs10_df.itertuples():
        day_date = row.timestamp.date()
        dt_val = datetime.combine(day_date, datetime.min.time())
        store.write_macro(pd.DataFrame([{"timestamp": row.timestamp, "value": row.value, "series": "DGS10"}]), "DGS10", dt_val)
        
    for row in t10y2y_df.itertuples():
        day_date = row.timestamp.date()
        dt_val = datetime.combine(day_date, datetime.min.time())
        store.write_macro(pd.DataFrame([{"timestamp": row.timestamp, "value": row.value, "series": "T10Y2Y"}]), "T10Y2Y", dt_val)
        
    print("  [+] Saved VIX and FRED macro series partitions.")
    print("=" * 70)
    print("[SUCCESS] Canonical 5m Intraday Ingestion & Partitioning Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Market Microstructure & Scanner Agent (Adapted from day-trader-skill trader-scanner).
Scans intraday bars for setups: Opening-Range Breakouts (ORB), VWAP crosses, and volume surges.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.core import get_logger

logger = get_logger(__name__)


class ScannerAgent:
    """
    Intraday Pattern & Setup Detection Agent.
    Identifies high-probability entries:
    - 15m/30m Opening-Range Breakouts (ORB)
    - VWAP trend continuation
    - Unusual volume spikes (> 1.5x rolling average)
    """

    def __init__(self, orb_minutes: int = 15):
        self.orb_minutes = orb_minutes

    def scan_opening_range(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates Opening Range (first 15/30 mins) and identifies breakout status.
        """
        if df_day.empty or len(df_day) < 3:
            return {"setup": "WAITING_FOR_DATA", "status": "INACTIVE"}

        # First 3 bars (15 mins on 5m grid)
        bars_count = max(3, self.orb_minutes // 5)
        orb_bars = df_day.iloc[:bars_count]
        range_high = float(orb_bars["high"].max())
        range_low = float(orb_bars["low"].min())
        range_mid = round((range_high + range_low) / 2.0, 2)
        range_size = round(range_high - range_low, 2)

        last_bar = df_day.iloc[-1]
        last_close = float(last_bar["close"])

        # Detect breakout
        if len(df_day) > bars_count:
            if last_close > range_high:
                status = "ORB_BULLISH_BREAKOUT"
                entry_signal = "BUY_CALL"
                target = round(last_close + range_size, 2)
                stop = range_mid
            elif last_close < range_low:
                status = "ORB_BEARISH_BREAKOUT"
                entry_signal = "BUY_PUT"
                target = round(last_close - range_size, 2)
                stop = range_mid
            else:
                status = "INSIDE_OPENING_RANGE"
                entry_signal = "NEUTRAL"
                target = range_high
                stop = range_low
        else:
            status = "FORMING_OPENING_RANGE"
            entry_signal = "WAITING"
            target = range_high
            stop = range_low

        return {
            "setup": "OPENING_RANGE_BREAKOUT",
            "status": status,
            "entry_signal": entry_signal,
            "range_high": range_high,
            "range_low": range_low,
            "range_mid": range_mid,
            "range_size": range_size,
            "current_price": last_close,
            "target": target,
            "stop": stop,
        }

    def check_volume_spike(self, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """
        Checks if current bar volume is an abnormal institutional surge (> 1.5x rolling mean).
        """
        if "volume" not in df.columns or len(df) < window:
            return {"is_spike": False, "volume_ratio": 1.0}

        avg_vol = df["volume"].iloc[-window:-1].mean()
        cur_vol = df["volume"].iloc[-1]
        ratio = cur_vol / max(avg_vol, 1.0)

        return {
            "is_spike": bool(ratio >= 1.5),
            "volume_ratio": round(float(ratio), 2),
            "current_volume": int(cur_vol),
            "average_volume": int(avg_vol),
        }


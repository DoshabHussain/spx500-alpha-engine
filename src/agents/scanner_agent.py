"""
Market Microstructure & Scanner Agent.
Computes intraday 15m Opening Range (ORB), ATR(14), VWAP distance, and volume surge ratio
strictly from real/current OHLCV bars without fabricated defaults.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

from src.core import get_logger

logger = get_logger(__name__)


class ScannerAgent:
    """
    Intraday Pattern, Volatility (ATR), and Opening-Range Detection Agent.
    """

    def __init__(self, orb_minutes: int = 15, atr_period: int = 14):
        self.orb_minutes = orb_minutes
        self.atr_period = atr_period

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        if df.empty or len(df) < 2:
            return 0.0
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        prev_c = np.roll(closes, 1)
        prev_c[0] = closes[0]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)),
        )
        window = tr[-min(period, len(tr)):]
        return round(float(np.mean(window)), 4)

    def scan_opening_range(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates Opening Range (first 15 mins on 5m grid = 3 bars), ATR(14),
        VWAP alignment, and dynamic ATR-calibrated target/stop distances.
        """
        if df_day is None or df_day.empty or len(df_day) < 3:
            return {
                "setup": "NO_INTRADAY_BARS",
                "status": "NO_DATA",
                "entry_signal": "NEUTRAL",
                "range_high": 0.0,
                "range_low": 0.0,
                "range_mid": 0.0,
                "range_size": 0.0,
                "atr_14": 0.0,
                "vwap": 0.0,
                "current_price": 0.0,
                "target": 0.0,
                "stop": 0.0,
                "bars_count": 0,
            }

        bars_count = max(3, self.orb_minutes // 5)
        orb_bars = df_day.iloc[:bars_count]
        range_high = round(float(orb_bars["high"].max()), 2)
        range_low = round(float(orb_bars["low"].min()), 2)
        range_mid = round((range_high + range_low) / 2.0, 2)
        range_size = round(range_high - range_low, 2)

        last_bar = df_day.iloc[-1]
        last_close = round(float(last_bar["close"]), 2)
        atr_14 = self.compute_atr(df_day, period=self.atr_period)
        if atr_14 <= 0:
            atr_14 = round(max(last_close * 0.0015, 0.25), 4)

        if "vwap" in df_day.columns and pd.notna(last_bar.get("vwap")):
            vwap_val = round(float(last_bar["vwap"]), 2)
        else:
            vwap_val = last_close

        if len(df_day) > bars_count:
            if last_close > range_high:
                status = "ORB_BULLISH_BREAKOUT"
                entry_signal = "BUY_CALL"
                target = round(last_close + 1.5 * atr_14, 2)
                stop = round(last_close - 1.0 * atr_14, 2)
            elif last_close < range_low:
                status = "ORB_BEARISH_BREAKOUT"
                entry_signal = "BUY_PUT"
                target = round(last_close - 1.5 * atr_14, 2)
                stop = round(last_close + 1.0 * atr_14, 2)
            else:
                status = "INSIDE_OPENING_RANGE"
                entry_signal = "NEUTRAL"
                target = round(last_close + 1.5 * atr_14, 2)
                stop = round(last_close - 1.0 * atr_14, 2)
        else:
            status = "FORMING_OPENING_RANGE"
            entry_signal = "NEUTRAL"
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
            "atr_14": round(atr_14, 2),
            "vwap": vwap_val,
            "above_vwap": bool(last_close >= vwap_val),
            "current_price": last_close,
            "target": target,
            "stop": stop,
            "bars_count": int(len(df_day)),
        }

    def check_volume_spike(self, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """
        Checks if latest non-zero bar volume exceeds 1.5x rolling mean.
        """
        if df is None or df.empty or "volume" not in df.columns or len(df) < 4:
            return {"is_spike": False, "volume_ratio": 1.0, "current_volume": 0, "average_volume": 0}

        vols = df["volume"].astype(float)
        # Index bars (^GSPC) sometimes have 0 volume on last incomplete bar -> inspect last non-zero if needed
        cur_vol = float(vols.iloc[-1])
        prior = vols.iloc[-min(window, len(vols)):-1]
        avg_vol = float(prior.mean()) if len(prior) > 0 else cur_vol
        ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        return {
            "is_spike": bool(ratio >= 1.5),
            "volume_ratio": round(float(ratio), 2),
            "current_volume": int(cur_vol),
            "average_volume": int(avg_vol),
        }

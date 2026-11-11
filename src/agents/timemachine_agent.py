"""
Time Machine Bootstrap Simulation Agent.
Replays simulated intraday trajectories using empirical historical 5m OHLC bar resampling
(supporting BOTH LONG and SHORT trade directions and intrabar High/Low stop-first checking)
to compute honest statistical odds and price percentile fan paths.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.core import get_logger

logger = get_logger(__name__)


class TimeMachineAgent:
    """
    10,000-Path Empirical Historical Bootstrap Trade Scenario Engine.
    - Supports both LONG (CALL) and SHORT (PUT) trade geometries.
    - Samples real historical 5m bar step-returns (zero biased Gaussian drift).
    - Checks intrabar High/Low extremes conservatively (STOP evaluated BEFORE TARGET).
    - Uses local numpy Generator (never mutates global np.random state).
    - Returns multi-step percentile fan trajectories (p5, p25, p50, p75, p95).
    """

    def __init__(self, default_paths: int = 5000, seed: int = 42):
        self.default_paths = default_paths
        self.seed = seed

    def simulate_trade(
        self,
        entry_price: float,
        target_pct: float = 0.0030,
        stop_pct: float = 0.0020,
        horizon_bars: int = 12,
        paths: Optional[int] = None,
        direction: str = "LONG",
        historical_df: Optional[pd.DataFrame] = None,
        historical_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Runs empirical bootstrap Monte Carlo simulation supporting LONG and SHORT directions.
        """
        n_paths = int(paths or self.default_paths)
        rng = np.random.default_rng(self.seed)
        dir_norm = "SHORT" if str(direction).upper() in ("SHORT", "PUT", "PUT_BUY", "BUY_PUT") else "LONG"

        if dir_norm == "LONG":
            target_price = entry_price * (1.0 + target_pct)
            stop_price = entry_price * (1.0 - stop_pct)
        else:
            target_price = entry_price * (1.0 - target_pct)
            stop_price = entry_price * (1.0 + stop_pct)

        # Build empirical bar pool: (close_ret, high_push, low_pull)
        if historical_df is not None and len(historical_df) > 30:
            closes = historical_df["close"].astype(float).values
            prev_closes = np.roll(closes, 1)
            prev_closes[0] = closes[0]
            close_rets = ((closes - prev_closes) / np.maximum(prev_closes, 1e-8))[1:]
            high_ratios = (historical_df["high"].astype(float).values / np.maximum(closes, 1e-8) - 1.0)[1:]
            low_ratios = (historical_df["low"].astype(float).values / np.maximum(closes, 1e-8) - 1.0)[1:]
            mode = "EMPIRICAL_HISTORICAL_BOOTSTRAP"
        elif historical_returns is not None and len(historical_returns) > 20:
            close_rets = np.asarray(historical_returns, dtype=float)
            high_ratios = np.abs(close_rets) * 0.35
            low_ratios = -np.abs(close_rets) * 0.35
            mode = "EMPIRICAL_RETURNS_BOOTSTRAP"
        else:
            # Zero-drift empirical-style fallback symmetric noise (zero upward drift bias)
            sigma = 0.0011
            close_rets = rng.normal(0.0, sigma, size=1000)
            high_ratios = np.abs(rng.normal(0.0, sigma * 0.4, size=1000))
            low_ratios = -np.abs(rng.normal(0.0, sigma * 0.4, size=1000))
            mode = "ZERO_DRIFT_NORMAL_FALLBACK"

        pool_len = len(close_rets)
        draw_idx = rng.integers(0, pool_len, size=(n_paths, horizon_bars))
        step_rets = close_rets[draw_idx]
        step_high_ext = high_ratios[draw_idx]
        step_low_ext = low_ratios[draw_idx]

        # Cumulative bar closes + intrabar high/low extremes
        cum_mult = np.cumprod(1.0 + step_rets, axis=1)
        sim_closes = entry_price * cum_mult
        sim_highs = sim_closes * (1.0 + np.maximum(0.0, step_high_ext))
        sim_lows = sim_closes * (1.0 + np.minimum(0.0, step_low_ext))

        if dir_norm == "LONG":
            hit_target_matrix = sim_highs >= target_price
            hit_stop_matrix = sim_lows <= stop_price
        else:
            hit_target_matrix = sim_lows <= target_price
            hit_stop_matrix = sim_highs >= stop_price

        target_hit_mask = np.any(hit_target_matrix, axis=1)
        target_hit_step = np.where(target_hit_mask, np.argmax(hit_target_matrix, axis=1), 999999)

        stop_hit_mask = np.any(hit_stop_matrix, axis=1)
        stop_hit_step = np.where(stop_hit_mask, np.argmax(hit_stop_matrix, axis=1), 999999)

        # Conservative rule: if both hit on same bar -> STOP wins (counted as loss)
        wins = int(np.sum(target_hit_mask & (target_hit_step < stop_hit_step)))
        losses = int(np.sum(stop_hit_mask & (stop_hit_step <= target_hit_step)))
        neither = int(n_paths - (wins + losses))

        win_rate = round(wins / n_paths * 100.0, 2)
        loss_rate = round(losses / n_paths * 100.0, 2)
        neither_rate = round(neither / n_paths * 100.0, 2)

        r_multiple = target_pct / max(stop_pct, 1e-6)
        expected_r = round(float((win_rate / 100.0) * r_multiple - (loss_rate / 100.0) * 1.0), 3)
        expected_roi = round(float((wins / n_paths) * target_pct * 100.0 - (losses / n_paths) * stop_pct * 100.0), 3)

        breakeven_winrate = 100.0 / (1.0 + r_multiple)
        edge = win_rate - breakeven_winrate

        if edge >= 8.0:
            verdict = "STRONGLY_TOWARD_TARGET"
            trade_verdict = "POSITIVE STATISTICAL EDGE"
        elif edge >= 2.0:
            verdict = "LEANS_TOWARD_TARGET"
            trade_verdict = "MODEST EDGE"
        elif edge <= -6.0:
            verdict = "LEANS_TOWARD_STOP"
            trade_verdict = "NEGATIVE EXPECTANCY"
        else:
            verdict = "CLOSE_TO_BREAK_EVEN"
            trade_verdict = "NEAR FAIR VALUE"

        # Multi-step percentile fan paths (prepend step 0 = entry_price)
        all_paths = np.column_stack([np.full(n_paths, entry_price), sim_closes])
        fan_series = {
            "steps": list(range(horizon_bars + 1)),
            "p5": [round(float(x), 2) for x in np.percentile(all_paths, 5, axis=0)],
            "p25": [round(float(x), 2) for x in np.percentile(all_paths, 25, axis=0)],
            "p50": [round(float(x), 2) for x in np.percentile(all_paths, 50, axis=0)],
            "p75": [round(float(x), 2) for x in np.percentile(all_paths, 75, axis=0)],
            "p95": [round(float(x), 2) for x in np.percentile(all_paths, 95, axis=0)],
        }

        terminal_prices = sim_closes[:, -1]
        return {
            "paths_simulated": n_paths,
            "bootstrap_mode": mode,
            "direction": dir_norm,
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_price": round(stop_price, 2),
            "target_pct": round(target_pct * 100.0, 3),
            "stop_pct": round(stop_pct * 100.0, 3),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "open_at_close_rate": neither_rate,
            "breakeven_winrate": round(breakeven_winrate, 2),
            "expected_r": expected_r,
            "expected_roi_pct": expected_roi,
            "reward_risk_ratio": round(r_multiple, 2),
            "verdict": verdict,
            "trade_verdict": trade_verdict,
            "terminal_price_fan": {
                "p5": round(float(np.percentile(terminal_prices, 5)), 2),
                "p50_median": round(float(np.median(terminal_prices)), 2),
                "p95": round(float(np.percentile(terminal_prices, 95)), 2),
            },
            "fan_series": fan_series,
        }

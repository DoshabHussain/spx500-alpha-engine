"""
Time Machine Simulation Agent (Adapted from day-trader-skill trader-timemachine).
Replays 10,000 simulated trade sessions before committing capital to determine honest statistical odds.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.core import get_logger

logger = get_logger(__name__)


class TimeMachineAgent:
    """
    10,000-Path Monte Carlo Trade Scenario Replay Engine.
    Simulates subsequent intraday bars drawing from empirical 5m return distributions.
    Enforces conservative execution: stop-loss checked before profit-target in every bar.
    """

    def __init__(self, default_paths: int = 10000, seed: int = 42):
        self.default_paths = default_paths
        self.seed = seed

    def simulate_trade(
        self,
        entry_price: float,
        target_pct: float = 0.0030,  # +30 bps (~$1.50 on SPY 500)
        stop_pct: float = 0.0020,    # -20 bps (~$1.00 on SPY 500)
        horizon_bars: int = 12,      # 12 bars = 1 hour on 5m grid
        paths: Optional[int] = None,
        historical_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo simulation of `paths` trajectories.
        """
        n_paths = paths or self.default_paths
        np.random.seed(self.seed)

        target_price = entry_price * (1.0 + target_pct)
        stop_price = entry_price * (1.0 - stop_pct)

        # Draw return innovations from empirical historical pool or calibrated normal
        if historical_returns is not None and len(historical_returns) > 50:
            draw_indices = np.random.randint(0, len(historical_returns), size=(n_paths, horizon_bars))
            innovations = historical_returns[draw_indices]
        else:
            # Calibrated 5-minute intraday SPX volatility: 0.12% per 5m bar with mild drift
            sigma = 0.0012
            mu = 0.00008
            innovations = np.random.normal(mu, sigma, (n_paths, horizon_bars))

        # Generate cumulative price paths
        cum_multipliers = np.cumprod(1.0 + innovations, axis=1)
        simulated_prices = entry_price * cum_multipliers

        # Evaluate path outcomes
        hit_target_matrix = simulated_prices >= target_price
        hit_stop_matrix = simulated_prices <= stop_price

        # Find first bar where target or stop was hit
        target_hit_step = np.argmax(hit_target_matrix, axis=1)
        target_hit_mask = np.any(hit_target_matrix, axis=1)
        target_hit_step[~target_hit_mask] = 999999

        stop_hit_step = np.argmax(hit_stop_matrix, axis=1)
        stop_hit_mask = np.any(hit_stop_matrix, axis=1)
        stop_hit_step[~stop_hit_mask] = 999999

        # Conservative rule: If both hit on same bar or stop hit first -> count as loss
        wins = np.sum((target_hit_mask) & (target_hit_step < stop_hit_step))
        losses = np.sum((stop_hit_mask) & (stop_hit_step <= target_hit_step))
        neither = n_paths - (wins + losses)

        win_rate = round(float(wins / n_paths) * 100.0, 2)
        loss_rate = round(float(losses / n_paths) * 100.0, 2)
        neither_rate = round(float(neither / n_paths) * 100.0, 2)

        # Expected Value in R units: Target R = target_pct / stop_pct, Loss R = -1.0
        r_multiple = target_pct / stop_pct
        expected_r = round(float((win_rate / 100.0) * r_multiple - (loss_rate / 100.0) * 1.0), 3)
        expected_roi = round(float((win_rate * target_pct - loss_rate * stop_pct)), 3)

        # Verdict logic from day-trader-skill
        breakeven_winrate = 1.0 / (1.0 + r_multiple) * 100.0
        edge = win_rate - breakeven_winrate

        if edge >= 12.0:
            verdict = "STRONGLY_TOWARD_TARGET"
            trade_verdict = "HIGH STATISTICAL EDGE"
        elif edge >= 4.0:
            verdict = "LEANS_TOWARD_TARGET"
            trade_verdict = "FAVORABLE ODDS"
        elif edge <= -10.0:
            verdict = "LEANS_TOWARD_STOP"
            trade_verdict = "NEGATIVE EXPECTANCY"
        else:
            verdict = "CLOSE_TO_BREAK_EVEN"
            trade_verdict = "NEUTRAL EDGE"

        terminal_prices = simulated_prices[:, -1]
        p5 = round(float(np.percentile(terminal_prices, 5)), 2)
        p50 = round(float(np.median(terminal_prices)), 2)
        p95 = round(float(np.percentile(terminal_prices, 95)), 2)

        return {
            "paths_simulated": n_paths,
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_price": round(stop_price, 2),
            "target_pct": round(target_pct * 100, 2),
            "stop_pct": round(stop_pct * 100, 2),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "open_at_close_rate": neither_rate,
            "expected_r": expected_r,
            "expected_roi_pct": expected_roi,
            "reward_risk_ratio": round(r_multiple, 2),
            "verdict": verdict,
            "trade_verdict": trade_verdict,
            "terminal_price_fan": {"p5": p5, "p50_median": p50, "p95": p95},
        }


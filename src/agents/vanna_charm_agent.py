"""
Second-Order Options Greeks & Conformal Bounds Agent (Vanna & Charm Profiler).

Tracks dealer dynamic re-hedging flows & calibrated Conformal Prediction intervals:
1. Vanna Flow (dDelta / dSigma):
   Dealer delta shift when Implied Volatility (VIX) changes relative to Net GEX.
2. Charm Flow (dDelta / dTime):
   0DTE time decay acceleration across the RTH session relative to Zero-GEX pivot.
3. Non-Parametric Conformal 90% Interval:
   Uses calibrated out-of-fold residual quantiles (`conformal_q90_ret`) learned from Stage 4 CV.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from src.core import get_logger

logger = get_logger(__name__)


class VannaCharmAgent:
    """
    Quantifies dealer re-hedging pressures driven by Vanna and Charm and applies
    empirical Conformal Prediction intervals.
    """

    def __init__(self):
        self.name = "VannaCharmAgent"

    def compute_second_order_flows(
        self,
        current_spot: float,
        vix_level: float = 18.0,
        vix_change_pct: float = 0.0,
        session_minute: int = 120,
        net_gex_b: float = 0.0,
        zero_gex_pivot: Optional[float] = None,
        conformal_q90_ret: float = 0.0025,
    ) -> Dict[str, Any]:
        """
        Calculates Vanna & Charm re-hedging estimates and calibrated Conformal 90% bounds.
        """
        pivot = float(zero_gex_pivot if zero_gex_pivot is not None else current_spot)

        # VIX points move: negative change % means VIX dropped -> positive delta buy demand when net GEX > 0
        # Conversely, when dealers are short gamma (net_gex_b < 0), VIX spike forces selling.
        vix_pts_chg = (vix_change_pct / 100.0) * vix_level
        # Scaled notional estimate proportional to VIX shift and active gamma magnitude
        vanna_flow_b = round((-vix_pts_chg * 0.25 * (max(abs(net_gex_b), 0.5) / 10.0)), 2)

        if vix_change_pct <= -4.0:
            vanna_impact = "BUY_IMBALANCE"
            vanna_narrative = (
                f"VIX drop ({vix_change_pct:+.1f}%) reduces put delta hedging pressure "
                f"(est. {vanna_flow_b:+.2f}B delta unwind flow)."
            )
        elif vix_change_pct >= 4.0:
            vanna_impact = "SELL_IMBALANCE"
            vanna_narrative = (
                f"VIX expansion ({vix_change_pct:+.1f}%) increases dealer put delta exposure "
                f"(est. {vanna_flow_b:+.2f}B delta sell pressure)."
            )
        else:
            vanna_impact = "NEUTRAL"
            vanna_narrative = f"IV move ({vix_change_pct:+.1f}%) is mild; minimal Vanna flow pressure."

        # Charm decay phase relative to Zero-GEX pivot
        charm_flow_bias = "ABOVE_ZERO_GEX_PIN" if current_spot >= pivot else "BELOW_ZERO_GEX_ACCELERATION"
        if session_minute >= 330:
            charm_phase = "POWER_HOUR_ACCELERATION"
        elif session_minute >= 180:
            charm_phase = "AFTERNOON_DECAY_WINDOW"
        else:
            charm_phase = "MORNING_GAMMA_STABILITY"

        # Empirical Conformal Prediction interval (using calibrated OOF quantile `conformal_q90_ret`)
        q_ret = max(0.0005, float(conformal_q90_ret))
        conformal_half_width = current_spot * q_ret
        conformal_lower = round(current_spot - conformal_half_width, 2)
        conformal_upper = round(current_spot + conformal_half_width, 2)

        return {
            "agent": self.name,
            "vanna_flow_est_b": vanna_flow_b,
            "vanna_impact": vanna_impact,
            "vanna_narrative": vanna_narrative,
            "charm_phase": charm_phase,
            "charm_flow_bias": charm_flow_bias,
            "zero_gex_pivot": round(pivot, 2),
            "conformal_interval_90": {
                "lower_bound": conformal_lower,
                "upper_bound": conformal_upper,
                "half_width_pts": round(conformal_half_width, 2),
                "width_pts": round(conformal_upper - conformal_lower, 2),
                "calibrated_q90_pct": round(q_ret * 100.0, 3),
                "coverage": "Empirical OOF Conformal 90% Interval",
            },
        }

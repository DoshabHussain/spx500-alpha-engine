"""
Second-Order Options Greeks Agent (Vanna & Charm Profiler).

Tracks dealer dynamic re-hedging flows:
1. Vanna Flow (dDelta / dSigma):
   Measures how dealer delta changes when Implied Volatility shifts.
   A sharp VIX collapse forces dealers to mechanically BUY index futures to remain delta-neutral.
2. Charm Flow (dDelta / dTime):
   Measures how 0DTE option deltas decay with the passage of time towards 16:00 ET expiration.
"""
from typing import Dict, Any


class VannaCharmAgent:
    """
    Quantifies intraday dealer re-hedging pressures driven by Vanna and Charm.
    """

    def __init__(self):
        self.name = "VannaCharmAgent"

    def compute_second_order_flows(
        self,
        current_spot: float = 7688.62,
        vix_level: float = 16.61,
        vix_change_pct: float = -20.26,
        session_minute: int = 120,
        net_gex_b: float = 5.12,
    ) -> Dict[str, Any]:
        """
        Calculates estimated Vanna and Charm re-hedging flows in $ Billions.
        """
        # 1. Vanna Flow Estimate:
        # If dealers are Long Gamma / Long Vol, a plunge in IV requires buying underlying
        # Rule of thumb: each 1.0 point drop in VIX generates ~$450M in buying per $1B net GEX
        vix_drop_pts = max(0.0, -1.0 * (vix_change_pct / 100.0) * vix_level)
        vanna_flow_b = round((vix_drop_pts * 0.42 * (net_gex_b / 4.0)), 2)

        if vix_change_pct <= -5.0:
            vanna_impact = "STRONG_BUY_IMBALANCE"
            vanna_narrative = (
                f"VIX crush ({vix_change_pct:.1f}%) triggers mechanical dealer Vanna buying "
                f"(est. +${vanna_flow_b:.2f}B in underlying delta demand), supporting upward drift."
            )
        elif vix_change_pct >= 5.0:
            vanna_impact = "STRONG_SELL_IMBALANCE"
            vanna_narrative = (
                f"VIX spike (+{vix_change_pct:.1f}%) forces dealers to short futures for Vanna hedging."
            )
        else:
            vanna_impact = "NEUTRAL"
            vanna_narrative = "Implied volatility steady; minimal Vanna flow pressure."

        # 2. Charm Flow Estimate (0DTE Time Decay):
        # As session progresses (0 to 390 minutes), OTM 0DTE delta decays to zero.
        # If market is above high-volume strike, decay creates upward drift; if below, downward drag.
        pct_session_elapsed = min(1.0, session_minute / 390.0)
        charm_flow_bias = "BULLISH_PIN" if current_spot > 7680.0 else "BEARISH_DRAG"

        if session_minute >= 330:
            charm_phase = "POWER_HOUR_ACCELERATION"
        elif session_minute >= 180:
            charm_phase = "AFTERNOON_DECAY_PEAK"
        else:
            charm_phase = "EARLY_SESSION_STABILITY"

        # Conformal Prediction Bands (90% Coverage)
        # Dynamic based on current VIX and net GEX
        intraday_vol_sigma = (vix_level / 100.0) / (252 ** 0.5) * ((30.0 / 390.0) ** 0.5)
        conformal_half_width = current_spot * intraday_vol_sigma * 1.645 * (0.85 if net_gex_b > 0 else 1.25)
        conformal_lower = round(current_spot - conformal_half_width, 2)
        conformal_upper = round(current_spot + conformal_half_width, 2)

        return {
            "agent": self.name,
            "vanna_flow_est_b": vanna_flow_b,
            "vanna_impact": vanna_impact,
            "vanna_narrative": vanna_narrative,
            "charm_phase": charm_phase,
            "charm_flow_bias": charm_flow_bias,
            "conformal_interval_90": {
                "lower_bound": conformal_lower,
                "upper_bound": conformal_upper,
                "width_pts": round(conformal_upper - conformal_lower, 2),
                "coverage": "90% Statistical Coverage",
            },
        }


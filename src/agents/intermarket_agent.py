"""
Cross-Asset Intermarket Macro Agent for S&P 500 Intraday Forecasting.

Tracks global transmission mechanisms:
1. U.S. 10-Year & 2-Year Treasury Yields (Discount rates & equity multiples)
2. U.S. Dollar Index (DXY) currency pressure
3. Crude Oil (WTI) energy inflation impact
4. VIX Term Structure contango / backwardation
"""
from typing import Dict, Any, Optional


class CrossAssetIntermarketAgent:
    """
    Monitors global macro and intermarket assets to confirm or invalidate
    S&P 500 intraday directional moves.
    """

    def __init__(self):
        self.name = "CrossAssetIntermarketAgent"

    def evaluate_intermarket_macro(
        self,
        us10y_yield: float = 4.38,
        us10y_change_bps: float = -4.0,
        dxy_index: float = 101.23,
        dxy_change_pct: float = -0.26,
        vix_spot: float = 16.61,
        vix_9d: float = 15.80,
        wti_crude: float = 71.40,
    ) -> Dict[str, Any]:
        """
        Assesses cross-asset macro pressure on SPX.
        
        Returns:
            macro_bias: 'STRONG_RISK_ON', 'MODERATE_RISK_ON', 'NEUTRAL', 'RISK_OFF'
            macro_score: float in [-1.0, 1.0] (+1.0 = optimal tailwind, -1.0 = severe headwind)
            yield_pressure: assessment of bond market drag
            dollar_pressure: assessment of FX drag
        """
        score = 0.0
        catalysts = []

        # 1. Yield Analysis: Lower yields = Higher equity multiples
        if us10y_change_bps <= -3.0:
            score += 0.35
            catalysts.append(f"US 10Y yield eased {us10y_change_bps:.1f} bps to {us10y_yield:.2f}%, lifting equity multiples.")
        elif us10y_change_bps >= +4.0:
            score -= 0.40
            catalysts.append(f"US 10Y yield surged +{us10y_change_bps:.1f} bps to {us10y_yield:.2f}%, creating valuation drag.")
        else:
            score += 0.10
            catalysts.append(f"US 10Y yield stable at {us10y_yield:.2f}%.")

        # 2. Dollar Index (DXY): Weaker dollar = Multinationals outperform
        if dxy_change_pct < -0.15:
            score += 0.30
            catalysts.append(f"DXY softened {dxy_change_pct:.2f}% to {dxy_index:.2f}, removing currency friction.")
        elif dxy_change_pct > +0.25:
            score -= 0.30
            catalysts.append(f"DXY strengthened +{dxy_change_pct:.2f}% to {dxy_index:.2f}, tightening financial conditions.")
        else:
            score += 0.05

        # 3. VIX Term Structure: VIX > VIX9D is contango (normal / bullish), VIX < VIX9D is inversion (fear)
        vix_term_slope = vix_spot - vix_9d
        if vix_term_slope > 0.3:
            score += 0.25
            catalysts.append(f"VIX term structure in healthy contango (+{vix_term_slope:.2f}), confirming absence of short-term hedging demand.")
        elif vix_term_slope < -0.5:
            score -= 0.35
            catalysts.append(f"VIX term structure inverted ({vix_term_slope:.2f}), signaling immediate 0DTE protection buying.")

        score = max(-1.0, min(1.0, score))

        if score >= 0.45:
            macro_bias = "STRONG_RISK_ON"
        elif score >= 0.15:
            macro_bias = "MODERATE_RISK_ON"
        elif score <= -0.30:
            macro_bias = "RISK_OFF"
        else:
            macro_bias = "NEUTRAL_MIXED"

        return {
            "agent": self.name,
            "macro_bias": macro_bias,
            "macro_score": round(score, 3),
            "us10y_yield": us10y_yield,
            "dxy_index": dxy_index,
            "vix_spot": vix_spot,
            "vix_term_slope": round(vix_term_slope, 2),
            "catalysts": catalysts,
            "equity_tailwind": score > 0,
        }

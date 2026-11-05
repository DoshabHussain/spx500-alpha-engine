"""
Adversarial "Devil's Advocate" Critic Agent for S&P 500 Intraday Forecasting.

Functions like an institutional risk partner / skeptic:
1. Challenges directional consensus by stress-testing against market breadth divergences.
2. Identifies intraday liquidity traps (e.g., the 12:00 - 13:30 ET lunch-hour dead zone).
3. Flags extreme NYSE $TICK overbought/oversold exhaustion readings.
4. Detects resistance pin risk against high open-interest dealer gamma strikes.
"""
from typing import Dict, Any, Optional
import pandas as pd


class AdversarialCriticAgent:
    """
    Independent critic agent designed to eliminate confirmation bias
    in multi-agent trading consensus.
    """

    def __init__(self):
        self.name = "AdversarialCriticAgent"

    def critique_proposal(
        self,
        proposed_signal: str,
        current_spot: float,
        session_minute: int = 120,
        tick_proxy: float = 650.0,
        add_breadth: float = 1450.0,
        zero_gex_pivot: float = 7680.0,
        nearest_high_oi_strike: float = 7700.0,
    ) -> Dict[str, Any]:
        """
        Challenges the proposed trade signal with adversarial counter-evidence.
        
        Returns:
            critique_verdict: 'APPROVED', 'CAUTION_OVERBOUGHT', 'CHALLENGED_EXPIRATION_PIN', 'VETO_LUNCH_TRAP'
            critique_score: float in [0.0, 1.0] (0.0 = total skepticism/veto, 1.0 = zero objection)
            objections: List of specific counter-arguments
        """
        objections = []
        skepticism_score = 0.0  # Higher means more skeptic

        # Check 1: Lunch Hour Dead Zone (12:00 - 13:30 ET, minutes 150 to 240)
        is_lunch_zone = 150 <= session_minute <= 240
        if is_lunch_zone:
            skepticism_score += 0.35
            objections.append(
                "Session is in the 12:00-13:30 ET institutional lunch lull; liquidity drops 40% and chop increases."
            )

        # Check 2: NYSE $TICK Exhaustion
        if proposed_signal in ["BUY_CALL", "CALL_BUY", "LONG"]:
            if tick_proxy > 1000:
                skepticism_score += 0.30
                objections.append(
                    f"NYSE $TICK is extreme overbought (+{tick_proxy:.0f}); high risk of mean-reverting pullback."
                )
            elif add_breadth < 0:
                skepticism_score += 0.40
                objections.append(
                    f"Bearish Breadth Divergence: Advance-Decline ($ADD={add_breadth:.0f}) is negative while price is rallying."
                )

        elif proposed_signal in ["BUY_PUT", "PUT_BUY", "SHORT"]:
            if tick_proxy < -1000:
                skepticism_score += 0.30
                objections.append(
                    f"NYSE $TICK is extreme oversold ({tick_proxy:.0f}); short squeeze bounce risk elevated."
                )
            elif add_breadth > 0:
                skepticism_score += 0.40
                objections.append(
                    f"Bullish Breadth Divergence: Advance-Decline ($ADD=+{add_breadth:.0f}) is positive while price is dropping."
                )

        # Check 3: Strike Pinning Barrier
        distance_to_pin = abs(current_spot - nearest_high_oi_strike)
        if distance_to_pin < 8.0 and current_spot < nearest_high_oi_strike and proposed_signal in ["BUY_CALL", "CALL_BUY"]:
            skepticism_score += 0.20
            objections.append(
                f"Proximity Warning: Spot is within {distance_to_pin:.1f} pts of major Call OI wall at {nearest_high_oi_strike:.0f}; dealer hedging may pin and cap further upside."
            )

        # Determine verdict
        if skepticism_score >= 0.65:
            verdict = "VETO_TRADE_BLOCKED"
        elif skepticism_score >= 0.35:
            verdict = "CAUTION_SIZE_REDUCED"
        else:
            verdict = "PROPOSAL_APPROVED"

        confidence_clearance = round(max(0.05, 1.0 - skepticism_score), 2)

        return {
            "agent": self.name,
            "verdict": verdict,
            "clearance_confidence": confidence_clearance,
            "skepticism_level": round(skepticism_score, 2),
            "objections_count": len(objections),
            "objections": objections if objections else ["No major microstructure divergences found. Clear runway."],
            "recommended_action": "REDUCE_SIZE_OR_WAIT" if verdict != "PROPOSAL_APPROVED" else "PROCEED",
        }

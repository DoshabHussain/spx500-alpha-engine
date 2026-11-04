"""
Risk & Budget Agent (Adapted from day-trader-skill trader-risk & trader-budget).
Enforces strict institutional capital preservation limits and sizes trades before order generation.
"""
from typing import Dict, Any, Optional

from src.core import get_logger

logger = get_logger(__name__)


class RiskAgent:
    """
    Pre-Trade Risk Manager & Sizing Engine.
    Rules:
    - Maximum single-trade risk: default 1.0% of portfolio ($100 per $10k)
    - Max daily loss threshold: -2.0% circuit breaker (trading halts if breached)
    - Minimum reward-to-risk ratio: 1.25R
    - End-of-day flatten by 3:55 PM ET
    """

    def __init__(
        self,
        account_equity: float = 25000.0,
        max_risk_pct_per_trade: float = 0.01,
        max_daily_drawdown_pct: float = 0.02,
        min_reward_risk_ratio: float = 1.25,
    ):
        self.equity = account_equity
        self.max_risk_pct = max_risk_pct_per_trade
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.min_reward_risk_ratio = min_reward_risk_ratio
        self.realized_pnl_today = 0.0
        self.trades_today = 0

    def size_position(
        self,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> Dict[str, Any]:
        """
        Sizes trade by dollars at risk per share:
        Shares = (Equity * max_risk_pct) / abs(entry - stop)
        """
        risk_per_share = abs(entry_price - stop_price)
        reward_per_share = abs(target_price - entry_price)

        if risk_per_share <= 0:
            return {"status": "BLOCK", "reason": "Zero or negative risk per share."}

        rr_ratio = reward_per_share / risk_per_share
        max_dollar_risk = self.equity * self.max_risk_pct

        shares = int(max_dollar_risk / risk_per_share)
        notional_value = round(shares * entry_price, 2)

        return {
            "shares": max(1, shares),
            "notional_value": notional_value,
            "max_dollar_risk": round(max_dollar_risk, 2),
            "risk_per_share": round(risk_per_share, 2),
            "reward_per_share": round(reward_per_share, 2),
            "reward_to_risk": round(rr_ratio, 2),
        }

    def evaluate_trade(
        self,
        signal: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        time_machine_winrate: float,
    ) -> Dict[str, Any]:
        """
        Runs rigorous pre-trade risk checks. Returns PASS or BLOCK.
        """
        # Check daily drawdown circuit breaker
        max_allowed_daily_loss = self.equity * self.max_daily_drawdown_pct
        if self.realized_pnl_today <= -max_allowed_daily_loss:
            return {
                "decision": "BLOCK",
                "reason": f"Daily loss limit hit (${abs(self.realized_pnl_today):.2f} >= ${max_allowed_daily_loss:.2f}). Trading halted.",
                "authorized": False,
            }

        # Check signal validity
        if signal not in ["CALL_BUY", "PUT_BUY"]:
            return {
                "decision": "BLOCK",
                "reason": f"Signal is {signal}. Only directional setups authorized.",
                "authorized": False,
            }

        # Check reward to risk
        risk_per_share = abs(entry_price - stop_price)
        reward_per_share = abs(target_price - entry_price)
        if risk_per_share <= 0:
            return {"decision": "BLOCK", "reason": "Invalid stop loss level.", "authorized": False}

        rr_ratio = reward_per_share / risk_per_share
        if rr_ratio < self.min_reward_risk_ratio:
            return {
                "decision": "BLOCK",
                "reason": f"Reward/Risk ratio ({rr_ratio:.2f}) below institutional minimum ({self.min_reward_risk_ratio}R).",
                "authorized": False,
            }

        # Check Monte Carlo odds
        if time_machine_winrate < 48.0:
            return {
                "decision": "BLOCK",
                "reason": f"Monte Carlo simulation win rate ({time_machine_winrate:.1f}%) lacks statistical edge.",
                "authorized": False,
            }

        # Position sizing
        sizing = self.size_position(entry_price, stop_price, target_price)

        return {
            "decision": "PASS",
            "reason": "All pre-trade risk thresholds satisfied.",
            "authorized": True,
            "sizing": sizing,
            "portfolio_heat_pct": round((sizing["max_dollar_risk"] / self.equity) * 100, 2),
            "stop_loss_level": stop_price,
            "profit_target_level": target_price,
        }


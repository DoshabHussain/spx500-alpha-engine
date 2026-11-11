"""
Master Autonomous Quant Committee & Multi-Agent Consensus Orchestrator.
Synthesizes real market observations from all 7 sub-agents + Purged ML Ensemble:
1. ScannerAgent: 15m ORB Breakout, ATR(14), VWAP & Volume Surge Monitor
2. NewsSentimentAgent: Live Yahoo RSS Polarity Scorer
3. CrossAssetIntermarketAgent: Real US10Y (^TNX), DXY (DX-Y.NYB), and ^VIX
4. VannaCharmAgent: Real Cboe GEX / Vanna / Charm + Empirical Conformal 90% Bounds
5. ML Ensemble + TreeSHAP: Purged Walk-Forward Probability Model + SHAP feature drivers
6. AdversarialCriticAgent: Devil's Advocate stress-testing (breadth, pin walls, lunch chop)
7. TimeMachineAgent: Empirical Historical Bootstrap simulation (both LONG and SHORT)
8. RiskAgent: ATR-based Stop/Targets, >=1.25R filter, 1% equity risk sizing & -2% halt
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

from src.agents.news_agent import NewsSentimentAgent
from src.agents.scanner_agent import ScannerAgent
from src.agents.timemachine_agent import TimeMachineAgent
from src.agents.risk_agent import RiskAgent
from src.agents.critic_agent import AdversarialCriticAgent
from src.agents.intermarket_agent import CrossAssetIntermarketAgent
from src.agents.vanna_charm_agent import VannaCharmAgent
from src.modeling.purged_trainer import SPXEnsembleForecaster
from src.core import get_logger

logger = get_logger(__name__)
NY_TZ = pytz.timezone("America/New_York")


class MasterTriEngineAgent:
    """
    Transparent Multi-Agent Committee Orchestrator.
    Zero hard-coded fake market prices or templated narratives: every agent reports
    its exact inputs, rule/formula basis, vote, and numeric output.
    """

    def __init__(
        self,
        forecaster: Optional[SPXEnsembleForecaster] = None,
        account_equity: float = 100000.0,
    ):
        self.news_agent = NewsSentimentAgent()
        self.scanner_agent = ScannerAgent()
        self.timemachine_agent = TimeMachineAgent(default_paths=5000, seed=42)
        self.risk_agent = RiskAgent(account_equity=account_equity)
        self.critic_agent = AdversarialCriticAgent()
        self.intermarket_agent = CrossAssetIntermarketAgent()
        self.vanna_charm_agent = VannaCharmAgent()
        self.forecaster = forecaster or SPXEnsembleForecaster()

    @staticmethod
    def _current_et_session_minute(df_intraday: Optional[pd.DataFrame] = None) -> int:
        if df_intraday is not None and not df_intraday.empty and "timestamp" in df_intraday.columns:
            last_ts = pd.to_datetime(df_intraday["timestamp"].iloc[-1])
            if getattr(last_ts, "tzinfo", None) is not None:
                last_ts = last_ts.tz_convert(NY_TZ)
            mins = (last_ts.hour - 9) * 60 + (last_ts.minute - 30)
            return int(max(0, min(390, mins)))
        now_ny = datetime.now(NY_TZ)
        mins = (now_ny.hour - 9) * 60 + (now_ny.minute - 30)
        return int(max(0, min(390, mins)))

    def generate_intraday_consensus(
        self,
        current_spot: float = 500.0,
        df_intraday: Optional[pd.DataFrame] = None,
        df_history: Optional[pd.DataFrame] = None,
        feature_vector: Optional[np.ndarray] = None,
        net_gex_b: float = 0.0,
        pc_ratio_0dte: float = 1.0,
        zero_gex_pivot: Optional[float] = None,
        call_wall: Optional[float] = None,
        put_wall: Optional[float] = None,
        vix_level: float = 18.0,
        vix_change_pct: float = 0.0,
        us10y_yield: float = 4.40,
        us10y_change_bps: float = 0.0,
        dxy_index: float = 101.0,
        dxy_change_pct: float = 0.0,
        session_minute: Optional[int] = None,
        news_headlines: Optional[List[Dict[str, Any]]] = None,
        provenance: str = "DELAYED",
    ) -> Dict[str, Any]:
        """
        Execute full multi-agent committee deliberation using live/supplied data
        and return structured reasoning & trade blueprint.
        """
        time_str = datetime.now(NY_TZ).strftime("%H:%M:%S ET")
        sess_min = session_minute if session_minute is not None else self._current_et_session_minute(df_intraday)
        pivot = float(zero_gex_pivot if zero_gex_pivot is not None else current_spot)
        c_wall = float(call_wall if call_wall is not None else round(current_spot * 1.005, 2))
        p_wall = float(put_wall if put_wall is not None else round(current_spot * 0.995, 2))

        # 1. News Sentiment Agent
        headlines = news_headlines if news_headlines is not None else self.news_agent.fetch_market_headlines(ticker="SPY")
        sentiment_metrics = self.news_agent.compute_aggregate_sentiment(headlines)
        sent_score = float(sentiment_metrics["sentiment_score"])

        # 2. Pattern Scanner & ATR Volatility Agent
        setup_metrics = self.scanner_agent.scan_opening_range(df_intraday)
        vol_metrics = self.scanner_agent.check_volume_spike(df_intraday)
        atr_14 = float(setup_metrics.get("atr_14") or 0.0)
        if atr_14 <= 0:
            # Conservative volatility proxy from VIX if bars unavailable
            atr_14 = round(current_spot * (vix_level / 100.0) / np.sqrt(252.0 * 14.0), 2)
            atr_14 = max(0.25, atr_14)

        # Derive causal intraday breadth proxies from actual intraday price & VWAP
        if df_intraday is not None and not df_intraday.empty:
            day_open = float(df_intraday["open"].iloc[0])
            day_ret = (current_spot - day_open) / max(day_open, 1e-8)
            short_ret = (
                (current_spot - float(df_intraday["close"].iloc[-min(6, len(df_intraday))]))
                / max(float(df_intraday["close"].iloc[-min(6, len(df_intraday))]), 1e-8)
            )
            tick_proxy = round(float(np.tanh(short_ret * 350.0) * 1150.0), 0)
            add_breadth = round(float(np.tanh(day_ret * 220.0) * 2000.0), 0)
        else:
            tick_proxy = 0.0
            add_breadth = 0.0

        # 3. Cross-Asset Intermarket Macro Agent
        macro_metrics = self.intermarket_agent.evaluate_intermarket_macro(
            us10y_yield=us10y_yield,
            us10y_change_bps=us10y_change_bps,
            dxy_index=dxy_index,
            dxy_change_pct=dxy_change_pct,
            vix_spot=vix_level,
        )

        # 4. Vanna / Charm + Empirical Conformal Prediction Agent
        conformal_q90 = getattr(self.forecaster, "conformal_q90_ret", 0.0025) if self.forecaster else 0.0025
        vanna_charm_metrics = self.vanna_charm_agent.compute_second_order_flows(
            current_spot=current_spot,
            vix_level=vix_level,
            vix_change_pct=vix_change_pct,
            session_minute=sess_min,
            net_gex_b=net_gex_b,
            zero_gex_pivot=pivot,
            conformal_q90_ret=conformal_q90,
        )

        # 5. Purged Walk-Forward ML Ensemble + TreeSHAP
        feature_contributions: List[Dict[str, Any]] = []
        if self.forecaster is not None and self.forecaster.is_trained and feature_vector is not None:
            ml_pred = self.forecaster.predict_intraday(feature_vector)
            p_up = float(ml_pred["p_up"])
            p_flat = float(ml_pred["p_flat"])
            p_down = float(ml_pred["p_down"])
            feature_contributions = ml_pred.get("feature_contributions", [])
            ml_source = "TRAINED_PURGED_ENSEMBLE"
        else:
            # Neutral uninformative baseline prior when model unavailable (never bullish 0.60 bias!)
            p_up, p_flat, p_down = 0.33, 0.34, 0.33
            ml_source = "UNINFORMATIVE_PRIOR_MODEL_UNAVAILABLE"

        # Aggregate Transparent Weighted Consensus Score in [-1, +1]
        ml_edge = p_up - p_down
        orb_vote = 1.0 if setup_metrics["entry_signal"] == "BUY_CALL" else (-1.0 if setup_metrics["entry_signal"] == "BUY_PUT" else 0.0)
        gex_vote = 0.4 if (net_gex_b > 0 and pc_ratio_0dte < 0.95) else (-0.4 if (net_gex_b < 0 and pc_ratio_0dte > 1.05) else 0.0)
        macro_vote = float(macro_metrics["macro_score"])

        raw_consensus_score = (
            ml_edge * 0.40
            + orb_vote * 0.25
            + macro_vote * 0.20
            + sent_score * 0.10
            + gex_vote * 0.05
        )

        # Require clear conviction >= 0.08 or <= -0.08 (since honest ML edge + scanner agreement shifts cleanly)
        if raw_consensus_score >= 0.08:
            initial_signal = "BUY_CALL"
        elif raw_consensus_score <= -0.08:
            initial_signal = "BUY_PUT"
        else:
            initial_signal = "NEUTRAL"

        # 6. Adversarial Critic Agent
        pin_wall = c_wall if initial_signal == "BUY_CALL" else p_wall
        critic_evaluation = self.critic_agent.critique_proposal(
            proposed_signal=initial_signal,
            current_spot=current_spot,
            session_minute=sess_min,
            tick_proxy=tick_proxy,
            add_breadth=add_breadth,
            zero_gex_pivot=pivot,
            nearest_high_oi_strike=pin_wall,
        )

        # 7. ATR-Based Stop Loss & Targets (Never fixed +/-25.4 / -14.1!)
        step = 10.0 if current_spot > 2000 else 1.0
        atm_strike = int(round(current_spot / step) * step)
        spread_width = int(step * 3) if current_spot > 2000 else int(step * 5)

        stop_dist = round(max(atr_14 * 1.0, current_spot * 0.0012), 2)
        tp1_dist = round(stop_dist * 1.5, 2)
        tp2_dist = round(stop_dist * 2.5, 2)

        if critic_evaluation["verdict"] == "VETO_TRADE_BLOCKED":
            consensus_signal = "NEUTRAL"
            trade_direction = "NO_TRADE"
            alpha_strategy = "NO TRADE — VETOED BY ADVERSARIAL CRITIC"
            target_1 = round(current_spot + tp1_dist, 2)
            target_2 = round(current_spot + tp2_dist, 2)
            stop_price = round(current_spot - stop_dist, 2)
        elif initial_signal == "BUY_CALL":
            consensus_signal = "CALL_BUY"
            trade_direction = "LONG"
            alpha_strategy = f"BULL CALL SPREAD (0DTE {atm_strike}/{atm_strike + spread_width} C)"
            target_1 = round(current_spot + tp1_dist, 2)
            target_2 = round(current_spot + tp2_dist, 2)
            stop_price = round(current_spot - stop_dist, 2)
        elif initial_signal == "BUY_PUT":
            consensus_signal = "PUT_BUY"
            trade_direction = "SHORT"
            alpha_strategy = f"BEAR PUT SPREAD (0DTE {atm_strike}/{atm_strike - spread_width} P)"
            target_1 = round(current_spot - tp1_dist, 2)
            target_2 = round(current_spot - tp2_dist, 2)
            stop_price = round(current_spot + stop_dist, 2)
        else:
            consensus_signal = "NEUTRAL"
            trade_direction = "NO_TRADE"
            alpha_strategy = "NO TRADE — INSUFFICIENT CONVICTION / STAY FLAT"
            target_1 = round(current_spot + tp1_dist, 2)
            target_2 = round(current_spot + tp2_dist, 2)
            stop_price = round(current_spot - stop_dist, 2)

        # 8. Time Machine Empirical Historical Bootstrap Simulation (supports BOTH LONG & SHORT)
        sim_dir = "SHORT" if trade_direction == "SHORT" else "LONG"
        target_pct = abs(target_1 - current_spot) / max(current_spot, 1e-8)
        stop_pct = abs(stop_price - current_spot) / max(current_spot, 1e-8)
        sim_results = self.timemachine_agent.simulate_trade(
            entry_price=current_spot,
            target_pct=max(0.0008, target_pct),
            stop_pct=max(0.0005, stop_pct),
            horizon_bars=12,
            paths=5000,
            direction=sim_dir,
            historical_df=df_history if df_history is not None and len(df_history) > 30 else df_intraday,
        )

        # 9. Pre-Trade Risk Guardian & Position Sizing
        risk_evaluation = self.risk_agent.evaluate_trade(
            signal=consensus_signal,
            entry_price=current_spot,
            stop_price=stop_price,
            target_price=target_1,
            time_machine_winrate=sim_results["win_rate"],
        )

        # If RiskAgent blocks due to low Monte Carlo win rate or R:R -> downgrade final actionable signal cleanly
        final_actionable = "NO_TRADE"
        if risk_evaluation.get("authorized", False):
            final_actionable = trade_direction
        elif trade_direction in ("LONG", "SHORT"):
            # Show why Risk Guardian held back execution if Monte Carlo odds were unfavorable
            alpha_strategy = f"{trade_direction} BIAS — BLOCKED BY RISK GUARDIAN"

        rr_ratio = round(abs(target_1 - current_spot) / max(abs(current_spot - stop_price), 1e-8), 2)

        # Plain-English thesis explaining exactly how the trade plan was built
        if final_actionable == "NO_TRADE":
            reason_text = (
                f"Standing aside: Consensus score ({raw_consensus_score:+.3f}), "
                f"Critic verdict ({critic_evaluation['verdict']}), or Risk check ({risk_evaluation['reason']})."
            )
        else:
            reason_text = (
                f"{final_actionable} setup driven by Scanner ({setup_metrics['status']}, ATR={atr_14:.2f}), "
                f"ML P(Up/Down)={p_up*100:.1f}%/{p_down*100:.1f}%, Macro score={macro_vote:+.2f}, "
                f"and Bootstrap odds {sim_results['win_rate']:.1f}% target-before-stop at {rr_ratio:.2f}R."
            )

        # 10. Per-Agent Planning Breakdown ("How every agent works & on what basis")
        sizing_info = risk_evaluation.get("sizing", {})
        agent_reasoning = [
            {
                "step": 1,
                "agent": "ScannerAgent",
                "role": "Intraday Structure & ATR Sizing",
                "status": setup_metrics["status"],
                "vote": "BULLISH" if setup_metrics["entry_signal"] == "BUY_CALL" else ("BEARISH" if setup_metrics["entry_signal"] == "BUY_PUT" else "NEUTRAL"),
                "weight_pct": 25,
                "inputs": {
                    "spot": round(current_spot, 2),
                    "orb_high": setup_metrics["range_high"],
                    "orb_low": setup_metrics["range_low"],
                    "vwap": setup_metrics.get("vwap", 0.0),
                    "atr_14": round(atr_14, 2),
                    "vol_ratio": vol_metrics["volume_ratio"],
                },
                "basis_rule": "Detects price vs 15m Opening Range [ORB_Low, ORB_High], VWAP alignment, and computes Stop = 1.0×ATR(14), TP1 = 1.5×ATR(14), TP2 = 2.5×ATR(14).",
                "output_summary": f"Status={setup_metrics['status']} | ATR(14)={atr_14:.2f} pts | Vol={vol_metrics['volume_ratio']}x avg",
                "timestamp": time_str,
            },
            {
                "step": 2,
                "agent": "ML_PurgedEnsemble",
                "role": "Purged Walk-Forward Forecaster + TreeSHAP",
                "status": ml_source,
                "vote": "BULLISH" if ml_edge > 0.03 else ("BEARISH" if ml_edge < -0.03 else "NEUTRAL"),
                "weight_pct": 40,
                "inputs": {
                    "p_up": round(p_up, 4),
                    "p_flat": round(p_flat, 4),
                    "p_down": round(p_down, 4),
                    "top_shap_driver": feature_contributions[0]["feature"] if feature_contributions else "N/A",
                },
                "basis_rule": "Evaluates calibrated HGB + RF + LightGBM ensemble trained strictly on historical training folds (zero lookahead); decomposes decision via TreeSHAP log-odds.",
                "output_summary": f"P(Up)={p_up*100:.1f}%, P(Flat)={p_flat*100:.1f}%, P(Down)={p_down*100:.1f}%",
                "timestamp": time_str,
            },
            {
                "step": 3,
                "agent": "CrossAssetIntermarketAgent",
                "role": "Rates, Dollar & VIX Macro Filter",
                "status": macro_metrics["macro_bias"],
                "vote": "BULLISH" if macro_vote > 0.1 else ("BEARISH" if macro_vote < -0.1 else "NEUTRAL"),
                "weight_pct": 20,
                "inputs": {
                    "us10y_yield": round(us10y_yield, 3),
                    "us10y_chg_bps": round(us10y_change_bps, 1),
                    "dxy": round(dxy_index, 2),
                    "dxy_chg_pct": round(dxy_change_pct, 2),
                    "vix": round(vix_level, 2),
                },
                "basis_rule": "Scores equity tailwind (+0.35 if 10Y drops >=3 bps, +0.30 if DXY drops >=0.15%, penalizes yield/dollar spikes >=+4 bps / +0.25%).",
                "output_summary": f"Macro Bias={macro_metrics['macro_bias']} (Score={macro_vote:+.2f})",
                "timestamp": time_str,
            },
            {
                "step": 4,
                "agent": "NewsSentimentAgent",
                "role": "Yahoo RSS Financial NLP",
                "status": sentiment_metrics["sentiment_label"],
                "vote": sentiment_metrics["sentiment_label"],
                "weight_pct": 10,
                "inputs": {
                    "headlines_parsed": len(headlines),
                    "bullish_ratio": sentiment_metrics.get("bullish_ratio", 0.5),
                    "sentiment_score": sent_score,
                },
                "basis_rule": "Parses live/cached Yahoo Finance RSS titles against financial lexicon (+0.25 bullish term, -0.25 bearish term, clipped [-1, +1]).",
                "output_summary": f"Score={sent_score:+.3f} ({sentiment_metrics['sentiment_label']}) across {len(headlines)} headlines",
                "timestamp": time_str,
            },
            {
                "step": 5,
                "agent": "VannaCharmAgent",
                "role": "Cboe GEX & Conformal 90% Bounds",
                "status": vanna_charm_metrics["vanna_impact"],
                "vote": "BULLISH" if (net_gex_b > 0 and vix_change_pct < -2) else ("BEARISH" if (net_gex_b < 0 and vix_change_pct > 2) else "NEUTRAL"),
                "weight_pct": 5,
                "inputs": {
                    "net_gex_b": round(net_gex_b, 3),
                    "zero_gex_pivot": round(pivot, 2),
                    "pc_ratio": round(pc_ratio_0dte, 3),
                    "vix_chg_pct": round(vix_change_pct, 2),
                },
                "basis_rule": "Uses real Cboe options strike gamma, Vanna delta unwind (-dVIX * GEX), and Stage 4 empirical Conformal 90% OOF residual quantile.",
                "output_summary": f"Conformal 90%: [{vanna_charm_metrics['conformal_interval_90']['lower_bound']}, {vanna_charm_metrics['conformal_interval_90']['upper_bound']}] | {vanna_charm_metrics['charm_phase']}",
                "timestamp": time_str,
            },
            {
                "step": 6,
                "agent": "AdversarialCriticAgent",
                "role": "Devil's Advocate & Breadth Veto",
                "status": critic_evaluation["verdict"],
                "vote": critic_evaluation["verdict"],
                "weight_pct": 0,
                "inputs": {
                    "session_minute": sess_min,
                    "tick_proxy": tick_proxy,
                    "add_proxy": add_breadth,
                    "nearest_oi_wall": pin_wall,
                },
                "basis_rule": "Veto gate (skepticism >= 0.65): penalizes 12:00-13:30 ET lunch chop (+0.35), extreme |$TICK|>1000 (+0.30), breadth divergence (+0.40), OI wall pin (<8 pts, +0.20).",
                "output_summary": f"{critic_evaluation['verdict']} (Clearance={critic_evaluation['clearance_confidence']}) — {critic_evaluation['objections'][0]}",
                "timestamp": time_str,
            },
            {
                "step": 7,
                "agent": "TimeMachineAgent",
                "role": "Empirical Bootstrap Simulator",
                "status": sim_results["verdict"],
                "vote": sim_results["trade_verdict"],
                "weight_pct": 0,
                "inputs": {
                    "paths": sim_results["paths_simulated"],
                    "direction": sim_dir,
                    "target_pct": sim_results["target_pct"],
                    "stop_pct": sim_results["stop_pct"],
                },
                "basis_rule": f"Resamples {sim_results['paths_simulated']} empirical 12-bar OHLC trajectories ({sim_results['bootstrap_mode']}); evaluates intrabar Stop BEFORE Target conservatively.",
                "output_summary": f"Target-Before-Stop: {sim_results['win_rate']}% (Breakeven: {sim_results['breakeven_winrate']}%) | Exp R: {sim_results['expected_r']:+.3f}R",
                "timestamp": time_str,
            },
            {
                "step": 8,
                "agent": "RiskGuardianAgent",
                "role": "VaR Budget & Final Authorization",
                "status": risk_evaluation["decision"],
                "vote": "PASS" if risk_evaluation.get("authorized") else "BLOCK",
                "weight_pct": 0,
                "inputs": {
                    "account_equity": round(self.risk_agent.equity, 2),
                    "max_risk_pct": round(self.risk_agent.max_risk_pct * 100, 1),
                    "rr_ratio": rr_ratio,
                    "bootstrap_win_rate": sim_results["win_rate"],
                },
                "basis_rule": "Requires directional edge, R:R >= 1.25, daily DD > -2.0%, and Bootstrap winrate >= 48%. Sizes max dollar loss = 1.0% equity.",
                "output_summary": f"{risk_evaluation['decision']} — {risk_evaluation['reason']}",
                "timestamp": time_str,
            },
        ]

        # Chronological Trace built strictly from real agent outputs
        debate_log = [
            {
                "time": time_str,
                "agent": a["agent"],
                "badge": a["vote"],
                "color": "text-cyan-400" if a["step"] <= 5 else ("text-amber-400" if a["step"] == 6 else "text-emerald-400"),
                "message": f"{a['output_summary']} | Basis: {a['basis_rule']}",
            }
            for a in agent_reasoning
        ]

        return {
            "timestamp": datetime.now(NY_TZ).isoformat(),
            "provenance": provenance,
            "spot_price": round(current_spot, 2),
            "consensus_signal": consensus_signal,
            "final_actionable_signal": final_actionable,
            "consensus_score": round(float(raw_consensus_score), 3),
            "alpha_strategy": alpha_strategy,
            "trade_plan": {
                "signal": final_actionable,
                "raw_direction": trade_direction,
                "mandate": alpha_strategy,
                "entry": round(current_spot, 2),
                "stop_loss": stop_price,
                "tp1": target_1,
                "tp2": target_2,
                "atr_14": round(atr_14, 2),
                "reward_to_risk": rr_ratio,
                "prob_target_before_stop": sim_results["win_rate"],
                "expected_r": sim_results["expected_r"],
                "suggested_shares": sizing_info.get("shares", 0),
                "max_dollar_risk": sizing_info.get("max_dollar_risk", round(self.risk_agent.equity * self.risk_agent.max_risk_pct, 2)),
                "mandatory_exit": "15:45 ET (Flatten Intraday)",
                "reason": reason_text,
            },
            "probabilities": {
                "p_up": p_up,
                "p_flat": p_flat,
                "p_down": p_down,
                "source": ml_source,
            },
            "feature_contributions": feature_contributions,
            "conformal_prediction_90": vanna_charm_metrics["conformal_interval_90"],
            "scanner_engine": setup_metrics,
            "volume_engine": vol_metrics,
            "microstructure": {
                "net_gex_b": round(net_gex_b, 4),
                "zero_gex_pivot": round(pivot, 2),
                "call_wall": round(c_wall, 2),
                "put_wall": round(p_wall, 2),
                "gamma_regime": "LONG_GAMMA_PINNED" if net_gex_b > 0 else "SHORT_GAMMA_VOLATILE",
                "pc_ratio_0dte": round(pc_ratio_0dte, 3),
                "vanna_imbalance_b": vanna_charm_metrics["vanna_flow_est_b"],
                "charm_phase": vanna_charm_metrics["charm_phase"],
            },
            "intermarket_macro": macro_metrics,
            "critic_audit": critic_evaluation,
            "sentiment_engine": {
                "score": sent_score,
                "label": sentiment_metrics["sentiment_label"],
                "headlines": headlines,
            },
            "timemachine_simulation": sim_results,
            "risk_manager": risk_evaluation,
            "agent_reasoning": agent_reasoning,
            "committee_debate_log": debate_log,
        }

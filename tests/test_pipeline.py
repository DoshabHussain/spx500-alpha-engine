"""
Comprehensive Unit & Integration Test Suite for SPX 500 Forecasting System.
Validates Stages 1 through 7.
"""
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import get_config, TradingCalendar
from src.processing.features.microstructure_gex import MicrostructureGEXCalculator, compute_microstructure_alpha
from src.processing.labels.targets import TargetGenerator
from src.modeling.purged_trainer import PurgedWalkForwardValidator, SPXEnsembleForecaster
from src.agents import (
    NewsSentimentAgent,
    ScannerAgent,
    TimeMachineAgent,
    RiskAgent,
    MasterTriEngineAgent,
)
from src.modeling.backtest.backtester import IntradayBacktester


def test_stage1_calendar_and_config():
    """Verify Stage 1: Config loading and DST-safe trading calendar."""
    config = get_config("dev")
    assert config is not None
    assert config.environment["timezone"] == "America/New_York"

    cal = TradingCalendar("NYSE")
    # 2026-02-06 is a Friday (trading day)
    assert cal.is_trading_day(datetime(2026, 2, 6)) is True
    # 2026-01-01 is New Year's Day (holiday)
    assert cal.is_trading_day(datetime(2026, 1, 1)) is False


def test_stage3_microstructure_gex():
    """Verify Stage 3: Net Dealer GEX mathematical formulas."""
    calc = MicrostructureGEXCalculator()
    chain = pd.DataFrame([
        {"strike": 500.0, "type": "CALL", "open_interest": 10000, "volume": 5000, "gamma": 0.05, "spot": 500.0},
        {"strike": 500.0, "type": "PUT", "open_interest": 8000, "volume": 4000, "gamma": 0.05, "spot": 500.0},
    ])
    gex_df = calc.compute_strike_gex(chain, spot_price=500.0)
    assert "gex" in gex_df.columns

    metrics = calc.aggregate_gex_metrics(gex_df, spot_price=500.0)
    assert "net_gex_total_b" in metrics
    assert "gamma_regime" in metrics
    assert metrics["net_gex_total_b"] > 0
    assert metrics["gamma_regime"] == "LONG_GAMMA_PINNED"


def test_stage3_target_labels():
    """Verify Stage 3: Zero lookahead bias and forward returns generation."""
    generator = TargetGenerator()
    df = pd.DataFrame({
        "close": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0],
    })
    df_labels = generator.calculate_all_labels(df, horizons={"30m": 2})
    assert "forward_ret_30m" in df_labels.columns
    assert "label_30m" in df_labels.columns
    # The last 2 rows must have NaN label because future price hasn't happened yet
    assert pd.isna(df_labels["label_30m"].iloc[-1])
    assert pd.isna(df_labels["label_30m"].iloc[-2])
    # Prior rows must be valid
    assert not pd.isna(df_labels["label_30m"].iloc[0])


def test_stage4_purged_walk_forward_cv():
    """Verify Stage 4: Purged & Embargoed Time-Series Cross-Validation."""
    validator = PurgedWalkForwardValidator(n_splits=3, embargo_pct=0.05)
    splits = list(validator.split(n_samples=200))
    assert len(splits) == 3

    for train_idx, val_idx in splits:
        # Guarantee temporal order: max train index must be strictly less than min val index
        assert train_idx[-1] < val_idx[0]
        # Guarantee embargo gap: val_start - train_end >= embargo
        assert (val_idx[0] - train_idx[-1]) >= 2


def test_stage4_ensemble_forecaster():
    """Verify Stage 4: Multi-model ensemble training and probability calibration."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    y = np.random.choice([0, 1, 2], size=n)

    forecaster = SPXEnsembleForecaster(random_state=42)
    metrics = forecaster.train_with_purging(X, y, n_splits=3)
    assert forecaster.is_trained is True
    assert "mean_cv_accuracy" in metrics

    pred = forecaster.predict_intraday(X[0])
    assert "p_up" in pred and "p_down" in pred and "p_flat" in pred
    assert "signal" in pred
    assert abs((pred["p_up"] + pred["p_down"] + pred["p_flat"]) - 1.0) < 0.05


def test_stage5_multi_agents():
    """Verify Stage 5: Autonomous agents from day-trader-skill."""
    # 1. News agent
    news = NewsSentimentAgent()
    score, tags = news.analyze_text("SPX surges to all-time highs on strong earnings")
    assert score > 0
    assert "EARNINGS" in tags

    # 2. Scanner agent
    scanner = ScannerAgent(orb_minutes=15)
    df_orb = pd.DataFrame({
        "open": [500, 501, 502, 505],
        "high": [501, 502, 503, 506],
        "low": [499, 500, 501, 504],
        "close": [501, 502, 503, 506],
    })
    scan_res = scanner.scan_opening_range(df_orb)
    assert scan_res["setup"] == "OPENING_RANGE_BREAKOUT"
    assert scan_res["status"] == "ORB_BULLISH_BREAKOUT"

    # 3. Time Machine simulation
    tm = TimeMachineAgent(default_paths=500, seed=42)
    sim = tm.simulate_trade(entry_price=500.0, target_pct=0.003, stop_pct=0.002, paths=500)
    assert "win_rate" in sim
    assert "expected_r" in sim
    assert sim["paths_simulated"] == 500

    # 4. Risk agent
    risk = RiskAgent(account_equity=25000.0)
    risk_res = risk.evaluate_trade(
        signal="CALL_BUY",
        entry_price=500.0,
        stop_price=498.0,
        target_price=504.0,
        time_machine_winrate=65.0,
    )
    assert risk_res["decision"] == "PASS"
    assert risk_res["authorized"] is True

    # 5. Master Orchestrator
    orchestrator = MasterTriEngineAgent()
    consensus = orchestrator.generate_intraday_consensus(current_spot=504.0)
    assert "consensus_signal" in consensus
    assert "probabilities" in consensus


def test_stage6_intraday_backtester():
    """Verify Stage 6: Institutional backtest analytics."""
    backtester = IntradayBacktester(initial_capital=50000.0)
    df = pd.DataFrame({
        "close": [500.0 + i * 0.2 for i in range(50)],
        "signal": [2] * 50,  # All long
    })
    res = backtester.run_backtest(df, signals_col="signal", horizon_periods=5)
    assert res["total_trades"] > 0
    assert "sharpe_ratio" in res
    assert "max_drawdown_pct" in res
    assert res["final_equity"] > 0


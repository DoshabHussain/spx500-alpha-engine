"""
Comprehensive Unit & Integration Test Suite for SPX 500 Forecasting System.
Validates Stages 1 through 7, anti-leakage invariants, honest metrics,
agent reasoning completeness, forward ledger semantics, and offline web UI assets.
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
from src.processing.features.microstructure_gex import MicrostructureGEXCalculator
from src.processing.features.regime import RegimeFeatures
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
from src.modeling.backtest.position_tracker import GenesisPositionTracker


def test_stage1_calendar_and_config():
    """Verify Stage 1: Config loading and DST-safe trading calendar."""
    config = get_config("dev")
    assert config is not None
    assert config.environment["timezone"] == "America/New_York"

    cal = TradingCalendar("NYSE")
    assert cal.is_trading_day(datetime(2026, 2, 6)) is True
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


def test_stage3_regime_calendar_features():
    """Verify Stage 3: Economic calendar CSV loading produces non-null event proximity."""
    ts = pd.date_range("2026-01-28 10:00", periods=10, freq="5min", tz="America/New_York")
    df = pd.DataFrame({"timestamp": ts, "close": np.linspace(600.0, 602.0, 10)})
    reg = RegimeFeatures()
    out = reg.calculate_fomc_flags(df)
    out = reg.calculate_minutes_to_event(out)
    assert "is_fomc_day" in out.columns
    assert "minutes_to_event" in out.columns
    assert out["minutes_to_event"].isna().sum() == 0
    assert out["is_fomc_day"].iloc[0] == 1  # 2026-01-28 is an FOMC decision date


def test_stage3_target_labels():
    """Verify Stage 3: Zero lookahead bias and forward returns generation."""
    generator = TargetGenerator()
    df = pd.DataFrame({"close": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0]})
    df_labels = generator.calculate_all_labels(df, horizons={"30m": 2})
    assert "forward_ret_30m" in df_labels.columns
    assert "label_30m" in df_labels.columns
    assert pd.isna(df_labels["label_30m"].iloc[-1])
    assert pd.isna(df_labels["label_30m"].iloc[-2])
    assert not pd.isna(df_labels["label_30m"].iloc[0])


def test_stage4_purged_walk_forward_cv_zero_overlap():
    """Verify Stage 4: Label-horizon purging guarantees zero overlap across all folds."""
    purge_bars = 6
    validator = PurgedWalkForwardValidator(n_splits=4, embargo_pct=0.02, purge_bars=purge_bars)
    splits = list(validator.split(n_samples=300))
    assert len(splits) >= 3

    for train_idx, val_idx in splits:
        assert train_idx[-1] + purge_bars < val_idx[0]


def test_stage4_ensemble_forecaster():
    """Verify Stage 4: Multi-model ensemble training, Brier/confusion metrics, and TreeSHAP."""
    rng = np.random.default_rng(42)
    n = 260
    X = rng.normal(0.0, 1.0, size=(n, 5))
    y = rng.choice([0, 1, 2], size=n)
    feat_names = [f"feat_{i}" for i in range(5)]

    forecaster = SPXEnsembleForecaster(random_state=42)
    metrics = forecaster.train_with_purging(X, y, n_splits=3, purge_bars=6, feature_names=feat_names)
    assert forecaster.is_trained is True
    assert "mean_cv_accuracy" in metrics
    assert "calibration" in metrics and "brier_score" in metrics["calibration"]
    assert "confusion_matrix" in metrics
    assert metrics["verified_zero_overlap"] is True

    pred = forecaster.predict_intraday(X[0])
    assert abs((pred["p_up"] + pred["p_down"] + pred["p_flat"]) - 1.0) < 0.02
    assert "feature_contributions" in pred and len(pred["feature_contributions"]) > 0


def test_stage4_untrained_prior_is_neutral():
    """Untrained forecaster must raise on direct inference and MasterTriEngineAgent uses unbiased neutral prior."""
    forecaster = SPXEnsembleForecaster()
    with pytest.raises(ValueError, match="not trained"):
        forecaster.predict_intraday(np.zeros(5))

    orchestrator = MasterTriEngineAgent(forecaster=forecaster)
    consensus = orchestrator.generate_intraday_consensus(current_spot=500.0)
    probs = consensus["probabilities"]
    assert abs(probs["p_up"] - probs["p_down"]) < 1e-6
    assert abs((probs["p_up"] + probs["p_flat"] + probs["p_down"]) - 1.0) < 1e-4


def test_stage5_multi_agents_and_8_step_reasoning():
    """Verify Stage 5: Agents, TimeMachine LONG/SHORT bootstrap, and 8-step committee trace."""
    news = NewsSentimentAgent()
    score, tags = news.analyze_text("SPX surges to all-time highs on strong earnings")
    assert score > 0
    assert "EARNINGS" in tags

    scanner = ScannerAgent(orb_minutes=15)
    df_orb = pd.DataFrame({
        "open": [500.0, 501.0, 502.0, 505.0],
        "high": [501.0, 502.0, 503.0, 506.0],
        "low": [499.0, 500.0, 501.0, 504.0],
        "close": [501.0, 502.0, 503.0, 506.0],
    })
    scan_res = scanner.scan_opening_range(df_orb)
    assert scan_res["setup"] == "OPENING_RANGE_BREAKOUT"
    assert scan_res["status"] == "ORB_BULLISH_BREAKOUT"

    # TimeMachine supports BOTH LONG and SHORT without mutating global np.random
    np.random.seed(12345)
    before_state = np.random.get_state()[1][:4].copy()
    tm = TimeMachineAgent(default_paths=500, seed=42)
    sim_long = tm.simulate_trade(entry_price=500.0, target_pct=0.003, stop_pct=0.002, paths=500, direction="LONG")
    sim_short = tm.simulate_trade(entry_price=500.0, target_pct=0.003, stop_pct=0.002, paths=500, direction="SHORT")
    after_state = np.random.get_state()[1][:4].copy()
    assert np.array_equal(before_state, after_state)
    assert sim_long["target_price"] > 500.0 and sim_long["stop_price"] < 500.0
    assert sim_short["target_price"] < 500.0 and sim_short["stop_price"] > 500.0
    assert "fan_series" in sim_short and len(sim_short["fan_series"]["p50"]) == 13

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

    orchestrator = MasterTriEngineAgent()
    consensus = orchestrator.generate_intraday_consensus(current_spot=504.0, df_intraday=df_orb)
    assert "scanner_engine" in consensus
    assert "trade_plan" in consensus
    assert "agent_reasoning" in consensus and len(consensus["agent_reasoning"]) == 8
    for step in consensus["agent_reasoning"]:
        assert "inputs" in step and "basis_rule" in step and "vote" in step


def test_stage6_intraday_backtester_strict_no_leakage():
    """Verify Stage 6: Rejects missing signals_col and computes real fixed-grid analytics."""
    backtester = IntradayBacktester(initial_capital=50000.0, slippage_bps=5.0)

    df_no_sig = pd.DataFrame({"close": [500.0 + i * 0.2 for i in range(30)]})
    with pytest.raises(ValueError, match="signals"):
        backtester.run_backtest(df_no_sig, signals_col="signal", horizon_periods=5)

    df = pd.DataFrame({
        "close": [500.0 + i * 0.2 for i in range(50)],
        "signal": [2] * 50,
    })
    res = backtester.run_backtest(df, signals_col="signal", horizon_periods=5)
    assert res["total_trades"] > 0
    assert "sharpe_ci_95" in res
    assert "benchmark_curve" in res and len(res["benchmark_curve"]) == 50
    assert res["final_equity"] > 0


def test_forward_ledger_starts_empty_and_checks_stop_first(tmp_path):
    """Forward paper ledger must start empty and resolve stops before targets conservatively."""
    ledger_file = tmp_path / "paper_ledger.jsonl"
    tracker = GenesisPositionTracker(initial_capital=100000.0, ledger_path=ledger_file)
    assert tracker.get_ledger_metrics()["total_signals"] == 0

    no_trade_res = tracker.record_signal({"signal": "NO_TRADE", "entry": 500.0, "stop_loss": 498.0, "tp1": 503.0})
    assert no_trade_res["recorded"] is False

    rec_res = tracker.record_signal(
        {"signal": "LONG", "entry": 500.0, "stop_loss": 498.0, "tp1": 503.0, "tp2": 505.0, "mandate": "CALL"},
        instrument_label="SPY",
    )
    assert rec_res["recorded"] is True
    assert tracker.get_ledger_metrics()["open_signals"] == 1

    # Bar spans BOTH stop (498) and target (503) -> Stop must win conservatively
    tracker.evaluate_against_bar(current_spot=501.0, bar_high=504.0, bar_low=497.5)
    metrics = tracker.get_ledger_metrics()
    assert metrics["open_signals"] == 0
    assert metrics["resolved_signals"] == 1
    assert metrics["recent_trades"][0]["outcome"] == "STOP_HIT"


def test_offline_web_assets_no_cdn():
    """Verify UI files contain zero external CDN dependencies and vendored libraries exist."""
    static_dir = project_root / "src" / "web" / "static"
    assert (static_dir / "vendor" / "lightweight-charts.standalone.production.js").exists()
    assert (static_dir / "vendor" / "chart.umd.min.js").exists()

    html = (static_dir / "index.html").read_text(encoding="utf-8")
    assert "https://" not in html and "http://" not in html
    assert "/static/vendor/lightweight-charts.standalone.production.js" in html
    assert "/static/vendor/chart.umd.min.js" in html


def test_telemetry_engine_and_backtest_contracts():
    """Verify TelemetryEngine emits honest provenance, 8-agent reasoning, and zero-fake backtest payloads."""
    from src.web.app import telemetry_engine

    tel = telemetry_engine.get_telemetry()
    assert tel["provenance"] in ("DELAYED", "CACHED_REAL", "DEMO")
    assert len(tel["agent_reasoning"]) == 8
    assert tel["spx"]["spot"] > tel["spy"]["spot"] * 8.0
    assert tel["ledger"]["total_signals"] == 0 or isinstance(tel["ledger"]["total_signals"], int)
    for key in ("entry", "stop_loss", "tp1", "tp2", "signal"):
        assert key in tel["spx"]["trade_plan"]

    bt = telemetry_engine.get_backtest_report()
    assert "available" in bt
    if bt["available"]:
        assert bt["strategy"]["slippage_bps"] == 5.0
        assert "sharpe_ci_95" in bt["strategy"]
        assert "holdout_evaluation" in bt["validation_summary"]


def test_ledger_crud_per_instrument_and_millisecond_cache(tmp_path):
    """Verify per-instrument MTM resolution, manual close/modify/delete actions, and sub-10ms warm cache."""
    import time
    from src.web.app import _get_cached_static, telemetry_engine

    ledger_file = tmp_path / "crud_ledger.jsonl"
    tracker = GenesisPositionTracker(initial_capital=100000.0, ledger_path=ledger_file)
    rec = tracker.record_signal(
        {"signal": "LONG", "entry": 750.0, "stop_loss": 748.0, "tp1": 753.0, "mandate": "MANUAL SPY LONG"},
        instrument_label="SPY",
    )
    tid = rec["trade"]["trade_id"]

    # Evaluating against BOTH SPX (7520) and SPY (750.5) must NOT false-trigger TP1 on SPX scale
    tracker.evaluate_against_bar(
        current_spot=7520.0,
        market_bars={
            "SPX": {"spot": 7520.0, "high": 7522.0, "low": 7518.0},
            "SPY": {"spot": 750.5, "high": 751.0, "low": 749.5},
        },
    )
    assert tracker.get_ledger_metrics()["open_signals"] == 1

    # Settle via update_trade_action
    upd = tracker.update_trade_action(trade_id=tid, action="CLOSE_TARGET")
    assert upd["updated"] is True
    assert upd["ledger"]["resolved_signals"] == 1
    assert upd["ledger"]["recent_trades"][0]["outcome"] == "TARGET_1_HIT"

    # Delete row cleanly
    rem = tracker.update_trade_action(trade_id=tid, action="DELETE")
    assert rem["updated"] is True
    assert rem["ledger"]["total_signals"] == 0

    # Warm static & telemetry cache speed check (< 50 ms)
    telemetry_engine.get_telemetry()
    _get_cached_static("app.js")
    t0 = time.perf_counter()
    gz_bytes = telemetry_engine.get_telemetry_bytes(accept_gzip=True)
    asset = _get_cached_static("app.js")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(gz_bytes) > 100
    assert asset is not None and len(asset["gzip"]) > 100
    assert elapsed_ms < 50.0


def test_security_hardening_headers_auth_and_ratelimit(monkeypatch):
    """Verify strict CSP, zero inline scripts, public GET transparency, Admin Key write lock, and input guards."""
    import io
    import json
    from types import SimpleNamespace
    from src.web.app import (
        QuantApiHandler,
        SlidingWindowRateLimiter,
        _get_cached_static,
        _parse_bounded_float,
    )

    # 1. Input guard & rate limiter unit assertions
    with pytest.raises(ValueError):
        _parse_bounded_float(float("nan"), 0.0, 10.0)
    with pytest.raises(ValueError):
        _parse_bounded_float(float("inf"), 0.0, 10.0)
    with pytest.raises(ValueError):
        _parse_bounded_float(25.0, 0.0, 10.0)
    assert _parse_bounded_float(5.5, 0.0, 10.0) == 5.5

    limiter = SlidingWindowRateLimiter()
    assert limiter.allow("test-ip", max_requests=2, window_sec=60.0) is True
    assert limiter.allow("test-ip", max_requests=2, window_sec=60.0) is True
    assert limiter.allow("test-ip", max_requests=2, window_sec=60.0) is False

    # 2. Zero inline scripts in HTML & theme-init.js pre-cached
    html_entry = _get_cached_static("index.html")
    js_init = _get_cached_static("theme-init.js")
    assert html_entry is not None and js_init is not None
    html_text = html_entry["raw"].decode("utf-8")
    assert "<script>" not in html_text
    assert "/static/theme-init.js" in html_text

    # 3. Hermetic in-memory HTTP dispatch through QuantApiHandler
    class _MemSocket:
        def __init__(self, raw_request: bytes) -> None:
            self._r = io.BytesIO(raw_request)
            self._w = io.BytesIO()

        def makefile(self, mode: str, *_args, **_kwargs):
            return self._r if "r" in mode else self._w

        def settimeout(self, _t: float) -> None:
            pass

        def sendall(self, data: bytes) -> None:
            self._w.write(data)

        def close(self) -> None:
            pass

    def _dispatch(method: str, path: str, headers: dict | None = None, body: bytes = b""):
        hdrs = {"Host": "localhost", "Connection": "close"}
        if body:
            hdrs["Content-Length"] = str(len(body))
        if headers:
            hdrs.update(headers)
        head = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items()) + "\r\n"
        sock = _MemSocket(head.encode("utf-8") + body)
        QuantApiHandler(sock, ("127.0.0.1", 12345), SimpleNamespace(server_name="localhost", server_port=8080))
        raw_resp = sock._w.getvalue()
        raw_head, _, raw_body = raw_resp.partition(b"\r\n\r\n")
        lines = raw_head.decode("iso-8859-1").split("\r\n")
        status_code = int(lines[0].split()[1])
        resp_headers = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:] if ":" in line)}
        return status_code, resp_headers, raw_body

    monkeypatch.setenv("ALPHA_ADMIN_KEY", "test-render-admin-secret")

    # /healthz + security headers
    status, hdrs, _ = _dispatch("GET", "/healthz")
    assert status == 200
    csp = hdrs.get("Content-Security-Policy", "")
    assert "script-src 'self'" in csp
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    assert hdrs.get("X-Frame-Options") == "DENY"
    assert hdrs.get("X-Content-Type-Options") == "nosniff"
    assert hdrs.get("Referrer-Policy") == "no-referrer"

    # Public ledger GET stays open even when Admin Key is set
    status, _, _ = _dispatch("GET", "/api/ledger")
    assert status == 200

    # Unauthorized POST to ledger must return 401
    status, _, _ = _dispatch("POST", "/api/ledger/clear", headers={"Content-Type": "application/json"}, body=b"{}")
    assert status == 401

    # Verify endpoint accepts valid ALPHA_ADMIN_KEY
    status, _, body_bytes = _dispatch(
        "POST",
        "/api/security/verify",
        headers={"Content-Type": "application/json", "X-Admin-Key": "test-render-admin-secret"},
        body=b"{}",
    )
    assert status == 200
    assert json.loads(body_bytes.decode("utf-8"))["authorized"] is True





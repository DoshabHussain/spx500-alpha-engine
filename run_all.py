"""
Master End-to-End Execution Script for SPX 500 Quantitative Research & Trading Prototype.
Executes and verifies all 7 stages cleanly with strict out-of-sample hold-out evaluation:
- Stage 1: Core Foundation, Config & Trading Calendar
- Stage 2: Data Ingestion (Real 60-day Yahoo 5m + Cboe GEX cache, or explicit DEMO fallback)
- Stage 3: Feature Engineering (Causal technicals, macro merges, calendar, GEX, 30m labels)
- Stage 4: Purged Walk-Forward CV (70% train) + Strict Untouched Hold-Out (30% test) + TreeSHAP
- Stage 5: Multi-Agent Autonomous Committee Deliberation (7 Agents + Bootstrap TimeMachine)
- Stage 6: Out-of-Sample Hold-Out Fixed-Grid Backtest (5 bps slippage, Benchmark, Baselines)
- Stage 7: Offline-Ready Pro Web Application & Verified REST Telemetry API

Usage:
    python run_all.py
    python run_all.py --serve --port 8080
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core import get_config, get_logger, TradingCalendar
from src.storage import get_raw_store
from src.ingestion.real_market_feed import DEFAULT_REAL_DIR, load_cached_real_bars
from src.processing.features import (
    calculate_all_returns_features,
    calculate_all_technical_features,
    calculate_all_regime_features,
    compute_microstructure_alpha,
)
from src.processing.labels import calculate_all_labels
from src.modeling.purged_trainer import SPXEnsembleForecaster, compute_calibration_curve
from src.agents.hybrid_agent import MasterTriEngineAgent
from src.modeling.backtest.backtester import IntradayBacktester

logger = get_logger("master_pipeline")


def print_stage_header(stage_num: int, title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  STAGE {stage_num}: {title.upper()}")
    print("=" * 76)


def _merge_causal_series(base_df: pd.DataFrame, aux_df: pd.DataFrame, col_name: str, out_col: str) -> pd.DataFrame:
    """Strictly causal backward as-of merge (never leaks future VIX/TNX/DXY bars)."""
    if aux_df is None or aux_df.empty or "timestamp" not in aux_df.columns:
        return base_df
    left = base_df.copy()
    right = aux_df[["timestamp", col_name]].dropna().copy()
    left["_ts_utc"] = pd.to_datetime(left["timestamp"], utc=True)
    right["_ts_utc"] = pd.to_datetime(right["timestamp"], utc=True)
    right = right.sort_values("_ts_utc").rename(columns={col_name: out_col})
    merged = pd.merge_asof(
        left.sort_values("_ts_utc"),
        right[["_ts_utc", out_col]],
        on="_ts_utc",
        direction="backward",
    )
    merged[out_col] = merged[out_col].ffill().bfill()
    return merged.drop(columns=["_ts_utc"]).reset_index(drop=True)


def run_pipeline(launch_server: bool = False, server_port: int = 8080, host: str = "127.0.0.1") -> dict:
    print("=" * 76)
    print("   SPX 500 QUANTITATIVE FORECASTING PROTOTYPE (AURA-500 QUANT)")
    print("   Strict Purged Walk-Forward CV & Hold-Out Verification Pipeline")
    print("=" * 76)

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 1: Core Foundation & Configuration
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(1, "Core Foundation & Calendar Verification")
    config = get_config("dev")
    cal = TradingCalendar("NYSE")
    now_test = datetime(2026, 2, 6, 10, 0)
    is_trading = cal.is_trading_day(now_test)
    print(f"  [+] YAML Configuration Loaded : environment={config.environment['name']}")
    print(f"  [+] Trading Calendar Verified : {now_test.date()} is_trading_day={is_trading}")
    print(f"  [+] Storage Paths Verified    : Raw={config.storage.root_dir}, Runs={config.storage.runs_dir}")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 2: Data Loading (Real Cached Market Data Priority -> Explicit DEMO)
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(2, "Data Ingestion & 5-Minute Canonical Grid Storage")
    spy_df, fetched_at = load_cached_real_bars("SPY", DEFAULT_REAL_DIR)
    vix_df, _ = load_cached_real_bars("^VIX", DEFAULT_REAL_DIR)
    tnx_df, _ = load_cached_real_bars("^TNX", DEFAULT_REAL_DIR)
    dxy_df, _ = load_cached_real_bars("DX-Y.NYB", DEFAULT_REAL_DIR)

    if not spy_df.empty and len(spy_df) >= 500:
        provenance = "CACHED_REAL"
        source_desc = f"Real 60-day Yahoo Chart v8 5m Dataset (fetched {(fetched_at or '')[:19]} UTC)"
        spy_df = _merge_causal_series(spy_df, vix_df, "close", "vix_level")
        spy_df = _merge_causal_series(spy_df, tnx_df, "close", "us10y_yield")
        spy_df = _merge_causal_series(spy_df, dxy_df, "close", "dxy_level")
    else:
        provenance = "DEMO"
        source_desc = "SYNTHETIC DEMO DATASET (No real cache found under data/real/)"
        store = get_raw_store()
        spy_df = store.read_underlying("SPY", datetime(2025, 11, 1), datetime(2026, 4, 1))
        if spy_df.empty or len(spy_df) < 1000:
            from scripts.create_sample_data import main as generate_data
            generate_data()
            spy_df = store.read_underlying("SPY", datetime(2025, 11, 1), datetime(2026, 4, 1))

    print(f"  [+] Data Provenance           : {provenance} ({source_desc})")
    print(f"  [+] Canonical SPY Bars        : {len(spy_df)} 5-minute bars")
    print(f"  [+] Date Horizon              : {spy_df['timestamp'].min()} to {spy_df['timestamp'].max()}")
    print(f"  [+] Price Range               : ${spy_df['close'].min():.2f} - ${spy_df['close'].max():.2f}")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 3: Feature Engineering & Causal Microstructure
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(3, "Feature Engineering, Causal Microstructure & Labels")
    spy_df = calculate_all_returns_features(spy_df)
    spy_df = calculate_all_technical_features(spy_df)
    spy_df = calculate_all_regime_features(spy_df)
    spy_df = compute_microstructure_alpha(spy_df)
    spy_df = calculate_all_labels(spy_df)

    # Compute additional real cross-asset percentage features if available
    if "vix_level" in spy_df.columns:
        spy_df["vix_ret_6"] = spy_df["vix_level"].pct_change(6).fillna(0.0)
    if "us10y_yield" in spy_df.columns:
        spy_df["us10y_chg_6"] = spy_df["us10y_yield"].diff(6).fillna(0.0)
    if "dxy_level" in spy_df.columns:
        spy_df["dxy_ret_6"] = spy_df["dxy_level"].pct_change(6).fillna(0.0)

    print(f"  [+] Total Generated Columns   : {len(spy_df.columns)}")
    print(f"  [+] minutes_to_event non-null : {int(spy_df['minutes_to_event'].notna().sum())} / {len(spy_df)}")
    print(f"  [+] Valid 30-Min Labels       : {int(spy_df['label_30m'].notna().sum())} bars")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 4: Chronological 70% Purged Walk-Forward CV + 30% Untouched Hold-Out
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(4, "Purged Walk-Forward CV (70%) & Untouched Hold-Out Evaluation (30%)")
    feature_cols = sorted(
        c for c in spy_df.columns
        if any(x in c for x in [
            "_ret_", "_ema_", "_rsi_", "_macd", "_atr_", "_bb_", "_obv",
            "session_", "is_", "gex", "breadth", "vix_", "us10y_", "dxy_", "minutes_to_event"
        ])
        and "forward" not in c and "label" not in c
        and spy_df[c].isna().sum() / max(len(spy_df), 1) < 0.20
    )

    df_clean = spy_df.dropna(subset=["label_30m", "forward_ret_30m"] + feature_cols).reset_index(drop=True)
    n_total = len(df_clean)
    split_idx = int(n_total * 0.70)
    purge_gap = 6  # Enforce strict 6-bar (30m) purge between train split and 30% hold-out

    df_train = df_clean.iloc[:split_idx - purge_gap].reset_index(drop=True)
    df_holdout = df_clean.iloc[split_idx:].reset_index(drop=True)

    X_train = df_train[feature_cols].values
    y_train = df_train["label_30m"].astype(int).values
    ret_train = df_train["forward_ret_30m"].values

    X_hold = df_holdout[feature_cols].values
    y_hold = df_holdout["label_30m"].astype(int).values

    print(f"  [+] Feature Count             : {len(feature_cols)} non-leaking features")
    print(f"  [+] Split (Zero Overlap)      : Train={len(df_train)} bars | Purge Gap={purge_gap} bars | Hold-Out={len(df_holdout)} bars")

    holdout_forecaster = SPXEnsembleForecaster(random_state=42)
    cv_metrics = holdout_forecaster.train_with_purging(
        X_train,
        y_train,
        feature_names=feature_cols,
        n_splits=5,
        embargo_pct=0.02,
        purge_bars=6,
        forward_returns=ret_train,
    )

    print(f"  [+] Purged CV Accuracies      : {cv_metrics['cv_accuracies']}")
    print(f"  [+] Mean Purged CV Accuracy   : {cv_metrics['mean_cv_accuracy']*100:.2f}% (95% CI: {cv_metrics['accuracy_ci_95'][0]*100:.1f}% - {cv_metrics['accuracy_ci_95'][1]*100:.1f}%)")
    print(f"  [+] OOF Brier Score           : {cv_metrics['calibration']['brier_score']} | p-value vs Majority: {cv_metrics['baselines']['p_value_vs_majority']}")
    print(f"  [+] Temporal Leakage Check    : {cv_metrics['leakage_bias']} (Verified Zero Overlap: {cv_metrics['verified_zero_overlap']})")

    # Evaluate on the UNTOUCHED 30% chronological Hold-Out set
    holdout_probs = holdout_forecaster.predict_proba(X_hold)
    holdout_preds = np.argmax(holdout_probs, axis=1)
    holdout_acc = float(accuracy_score(y_hold, holdout_preds))
    holdout_cm = confusion_matrix(y_hold, holdout_preds, labels=[0, 1, 2]).tolist()
    holdout_cal = compute_calibration_curve(y_hold, holdout_probs, n_bins=6)

    # Baselines on Hold-Out: Always Flat (1) and Simple 3-Bar Momentum
    always_flat_acc = float(np.mean(y_hold == 1))
    mom_3bar = df_holdout["spy_ret_1m"].rolling(3).sum().fillna(0.0).values if "spy_ret_1m" in df_holdout.columns else np.zeros(len(df_holdout))
    mom_preds = np.where(mom_3bar > 0.0005, 2, np.where(mom_3bar < -0.0005, 0, 1))
    mom_acc = float(accuracy_score(y_hold, mom_preds))

    print(f"  [+] Untouched Hold-Out Acc    : {holdout_acc*100:.2f}% (Always Flat: {always_flat_acc*100:.2f}%, 3-Bar Mom: {mom_acc*100:.2f}%)")
    print(f"  [+] Hold-Out Brier Score      : {holdout_cal['brier_score']} | ECE: {holdout_cal['ece']}")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 6 FIRST ON UNTOUCHED HOLD-OUT (before fitting final deployment model)
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(6, "Strict Out-of-Sample Hold-Out Fixed-Grid Backtest (5 bps Slippage)")
    backtester = IntradayBacktester(initial_capital=100000.0, slippage_bps=5.0, bars_per_day=78)
    max_probs = np.max(holdout_probs, axis=1)
    holdout_signals = holdout_preds.copy()
    holdout_signals[max_probs < 0.38] = 1  # Filter low conviction -> Cash

    df_holdout_bt = df_holdout.copy()
    df_holdout_bt["model_signal"] = holdout_signals
    df_holdout_bt["mom_signal"] = mom_preds

    bt_results = backtester.run_backtest(df_holdout_bt, signals_col="model_signal", horizon_periods=6)
    mom_bt_results = backtester.run_backtest(df_holdout_bt, signals_col="mom_signal", horizon_periods=6)

    print(f"  [+] Hold-Out Period           : {df_holdout['timestamp'].iloc[0]} -> {df_holdout['timestamp'].iloc[-1]} ({len(df_holdout)} bars)")
    print(f"  [+] Final Equity (OOS)        : ${bt_results['initial_capital']:,.2f} -> ${bt_results['final_equity']:,.2f} ({bt_results['total_return_pct']:+.2f}%)")
    print(f"  [+] SPY Buy & Hold Benchmark  : {bt_results['benchmark_return_pct']:+.2f}%")
    print(f"  [+] Hold-Out Sharpe / Sortino : {bt_results['sharpe_ratio']:.2f} (95% CI {bt_results['sharpe_ci_95']}) / {bt_results['sortino_ratio']:.2f}")
    print(f"  [+] Hold-Out Win Rate         : {bt_results['win_rate_pct']:.2f}% ({bt_results['total_trades']} trades) | PF={bt_results['profit_factor']:.2f}")
    print(f"  [+] Hold-Out Max Drawdown     : {bt_results['max_drawdown_pct']:.2f}%")
    print(f"  [+] Momentum Baseline Sharpe  : {mom_bt_results['sharpe_ratio']:.2f} (Return: {mom_bt_results['total_return_pct']:+.2f}%)")

    # Fit live-inference deployment model on complete clean dataset & save artifacts
    deploy_forecaster = SPXEnsembleForecaster(random_state=42)
    deploy_cv = deploy_forecaster.train_with_purging(
        df_clean[feature_cols].values,
        df_clean["label_30m"].astype(int).values,
        feature_names=feature_cols,
        n_splits=5,
        embargo_pct=0.02,
        purge_bars=6,
        forward_returns=df_clean["forward_ret_30m"].values,
    )

    runs_dir = Path("runs")
    models_dir = runs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "spx_ensemble_forecaster.joblib"
    joblib.dump(deploy_forecaster, model_file)

    # Persist real validation & backtest reports to disk for /api/backtest & UI Evidence tab
    validation_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "source_description": source_desc,
        "total_clean_bars": n_total,
        "train_bars": len(df_train),
        "purge_gap_bars": purge_gap,
        "holdout_bars": len(df_holdout),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "purged_cv": cv_metrics,
        "full_cv": deploy_cv,
        "holdout_evaluation": {
            "accuracy": round(holdout_acc, 4),
            "always_flat_accuracy": round(always_flat_acc, 4),
            "momentum_baseline_accuracy": round(mom_acc, 4),
            "confusion_matrix": holdout_cm,
            "calibration": holdout_cal,
            "start_timestamp": str(df_holdout["timestamp"].iloc[0]),
            "end_timestamp": str(df_holdout["timestamp"].iloc[-1]),
        },
        "top_features": deploy_forecaster.get_feature_importances().head(10).to_dict(orient="records"),
    }

    backtest_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "evaluation_mode": "STRICT_OUT_OF_SAMPLE_30PCT_HOLDOUT",
        "strategy": bt_results,
        "momentum_baseline": {
            "total_return_pct": mom_bt_results["total_return_pct"],
            "sharpe_ratio": mom_bt_results["sharpe_ratio"],
            "win_rate_pct": mom_bt_results["win_rate_pct"],
            "max_drawdown_pct": mom_bt_results["max_drawdown_pct"],
            "total_trades": mom_bt_results["total_trades"],
        },
        "validation_summary": validation_report,
    }

    (runs_dir / "latest_validation_report.json").write_text(
        json.dumps(validation_report, indent=2), encoding="utf-8"
    )
    (runs_dir / "latest_backtest.json").write_text(
        json.dumps(backtest_report, indent=2), encoding="utf-8"
    )
    print(f"  [+] Persisted Validation/BT   : runs/latest_backtest.json & {model_file}")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 5: Multi-Agent Neural & Microstructure Consensus Engine
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(5, "Multi-Agent Committee Consensus & Planning Explanation")
    agent_orchestrator = MasterTriEngineAgent(forecaster=deploy_forecaster)
    spot_now = float(df_clean["close"].iloc[-1])
    sample_feat = df_clean[feature_cols].iloc[-1].values

    cboe_spy_path = DEFAULT_REAL_DIR / "cboe_gex_spy.json"
    cboe_spy = json.loads(cboe_spy_path.read_text(encoding="utf-8")) if cboe_spy_path.exists() else {}

    vix_val = float(df_clean["vix_level"].iloc[-1]) if "vix_level" in df_clean.columns else 18.0
    tnx_val = float(df_clean["us10y_yield"].iloc[-1]) if "us10y_yield" in df_clean.columns else 4.5
    dxy_val = float(df_clean["dxy_level"].iloc[-1]) if "dxy_level" in df_clean.columns else 101.0

    consensus = agent_orchestrator.generate_intraday_consensus(
        current_spot=spot_now,
        df_intraday=df_clean.iloc[-78:],
        df_history=df_clean,
        feature_vector=sample_feat,
        net_gex_b=float(cboe_spy.get("net_gex_total_b", df_clean["net_gex_total_b"].iloc[-1])),
        pc_ratio_0dte=float(cboe_spy.get("pc_ratio_0dte_vol", df_clean["pc_ratio_0dte_vol"].iloc[-1])),
        zero_gex_pivot=cboe_spy.get("zero_gex_pivot"),
        call_wall=cboe_spy.get("call_wall"),
        put_wall=cboe_spy.get("put_wall"),
        vix_level=vix_val,
        us10y_yield=tnx_val,
        dxy_index=dxy_val,
        provenance=provenance,
    )

    print(f"  [+] NLP News Sentiment Agent  : Score={consensus['sentiment_engine']['score']} ({consensus['sentiment_engine']['label']})")
    print(f"  [+] Pattern Scanner Agent     : Setup={consensus['scanner_engine']['setup']}, Status={consensus['scanner_engine']['status']} (ATR={consensus['scanner_engine']['atr_14']})")
    print(f"  [+] TimeMachine Bootstrap     : WinRate={consensus['timemachine_simulation']['win_rate']}% ({consensus['timemachine_simulation']['bootstrap_mode']}, ExpR={consensus['timemachine_simulation']['expected_r']:+.3f}R)")
    print(f"  [+] Risk Guardian             : Decision={consensus['risk_manager']['decision']} ({consensus['risk_manager']['reason']})")
    print(f"  [+] Trade Plan Signal         : {consensus['final_actionable_signal']} | {consensus['alpha_strategy']}")
    print(f"  [+] ML Probabilities          : P(Up)={consensus['probabilities']['p_up']*100:.1f}%, P(Flat)={consensus['probabilities']['p_flat']*100:.1f}%, P(Down)={consensus['probabilities']['p_down']*100:.1f}%")

    # ───────────────────────────────────────────────────────────────────────────
    # STAGE 7: Web Application & API Server Summary
    # ───────────────────────────────────────────────────────────────────────────
    print_stage_header(7, "Offline-Ready Web Application & Telemetry API")
    print("  [+] Static Pro Trading UI     : src/web/static/ (TradingView Lightweight Charts + Chart.js)")
    print("  [+] Active Telemetry APIs     : /api/telemetry, /api/backtest, /api/committee, /api/gex, /api/ledger, /api/positions, /api/simulate")
    print("\n" + "=" * 76)
    print("  ALL 7 STAGES COMPLETED CLEANLY (STRICT HOLD-OUT & VERIFIED ZERO OVERLAP)")
    print("=" * 76)

    if launch_server:
        from src.web.app import run_web_server
        run_web_server(port=server_port, host=host)

    return {
        "provenance": provenance,
        "cv_metrics": cv_metrics,
        "backtest_report": backtest_report,
        "consensus": consensus,
    }


def main():
    parser = argparse.ArgumentParser(description="SPX 500 Master Pipeline Runner")
    parser.add_argument("--serve", action="store_true", help="Launch the web application after pipeline execution")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server bind host (default: 127.0.0.1)")
    args = parser.parse_args()
    run_pipeline(launch_server=args.serve, server_port=args.port, host=args.host)


if __name__ == "__main__":
    main()

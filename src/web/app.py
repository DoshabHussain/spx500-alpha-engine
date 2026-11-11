"""
Stage 7: AURA-500 Quantitative Research & Pro Trading Terminal Web Server.
Features:
- 100% offline-ready static UI served from src/web/static/ (Zero CDN dependencies)
- Honest telemetry provenance tags (DELAYED / CACHED_REAL / DEMO)
- Live ML ensemble inference + TreeSHAP feature contributions loaded from runs/models/
- Verified Stage 4/6 out-of-sample backtest & validation served from runs/latest_backtest.json
- Thread-safe TTL caching and empty-start persistent Forward Paper Ledger
"""
from __future__ import annotations

import json
import mimetypes
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.hybrid_agent import MasterTriEngineAgent
from src.agents.news_agent import NewsSentimentAgent
from src.ingestion.real_market_feed import LiveMarketSnapshotService
from src.modeling.backtest.position_tracker import GenesisPositionTracker
from src.processing.features import (
    calculate_all_regime_features,
    calculate_all_returns_features,
    calculate_all_technical_features,
    compute_microstructure_alpha,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = project_root / "runs"
MODEL_PATH = RUNS_DIR / "models" / "spx_ensemble_forecaster.joblib"
BACKTEST_REPORT_PATH = RUNS_DIR / "latest_backtest.json"
VALIDATION_REPORT_PATH = RUNS_DIR / "latest_validation_report.json"


class TelemetryEngine:
    """
    Thread-safe orchestrator binding real market snapshots, trained ML forecaster,
    multi-agent committee, news RSS, and forward paper ledger.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.snapshot_service = LiveMarketSnapshotService(ttl_seconds=20)
        self.news_agent = NewsSentimentAgent()
        self.position_tracker = GenesisPositionTracker(initial_capital=100000.0, risk_per_trade_pct=0.01)
        self.forecaster = self._load_forecaster()
        self.committee_agent = MasterTriEngineAgent(forecaster=self.forecaster, account_equity=100000.0)

        self._cached_payload: Optional[Dict[str, Any]] = None
        self._cached_at: float = 0.0
        self._news_cache: Optional[List[Dict[str, Any]]] = None
        self._news_at: float = 0.0

    @staticmethod
    def _load_forecaster():
        if MODEL_PATH.exists():
            try:
                return joblib.load(MODEL_PATH)
            except Exception:
                return None
        return None

    def _compute_latest_features(self, spy_df: pd.DataFrame, macro: Dict[str, Any]) -> Optional[np.ndarray]:
        if self.forecaster is None or not getattr(self.forecaster, "is_trained", False):
            return None
        if spy_df is None or len(spy_df) < 35:
            return None

        try:
            df = spy_df.copy().reset_index(drop=True)
            df["vix_level"] = float(macro.get("vix", 18.0))
            df["us10y_yield"] = float(macro.get("us10y", 4.5))
            df["dxy_level"] = float(macro.get("dxy", 101.0))

            df = calculate_all_returns_features(df)
            df = calculate_all_technical_features(df)
            df = calculate_all_regime_features(df)
            df = compute_microstructure_alpha(df)

            df["vix_ret_6"] = 0.0
            df["us10y_chg_6"] = 0.0
            df["dxy_ret_6"] = 0.0

            req_cols = getattr(self.forecaster, "feature_names", [])
            if not req_cols:
                return None
            row = df.iloc[-1]
            vec = [float(row[c]) if c in df.columns and pd.notna(row[c]) else 0.0 for c in req_cols]
            return np.asarray(vec, dtype=float)
        except Exception:
            return None

    def _get_news(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and self._news_cache is not None and (now - self._news_at < 60):
            return {"items": self._news_cache, "provenance": self._news_provenance}
        items = self.news_agent.fetch_market_headlines(ticker="SPY", count=8)
        is_real_rss = any(h.get("source") == "Yahoo RSS" for h in items)
        self._news_cache = items
        self._news_provenance = "DELAYED_RSS" if is_real_rss else "DEMO"
        self._news_at = now
        return {"items": items, "provenance": self._news_provenance}

    def get_telemetry(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            if not force_refresh and self._cached_payload is not None and (now - self._cached_at < 10):
                payload = dict(self._cached_payload)
                payload["server_cache_age_sec"] = int(now - self._cached_at)
                # Refresh live ledger view cleanly
                payload["ledger"] = self.position_tracker.get_ledger_metrics()
                return payload

        if self.forecaster is None:
            self.forecaster = self._load_forecaster()
            if self.forecaster is not None:
                self.committee_agent.forecaster = self.forecaster

        snap = self.snapshot_service.get_snapshot(force_refresh=force_refresh)
        news_res = self._get_news(force=force_refresh)

        spx = snap["spx"]
        spy = snap["spy"]
        macro = snap["macro"]
        provenance = snap["provenance"]

        spy_df_hist = spy.get("df_history", pd.DataFrame())
        spy_df_day = spy.get("df_day", pd.DataFrame())
        spx_gex = spx.get("gex", {})
        spy_gex = spy.get("gex", {})

        feat_vec = self._compute_latest_features(spy_df_hist, macro)

        # Compute committee consensus on SPY canonical scale (where ML model was trained)
        consensus = self.committee_agent.generate_intraday_consensus(
            current_spot=float(spy["spot"]),
            df_intraday=spy_df_day,
            df_history=spy_df_hist,
            feature_vector=feat_vec,
            net_gex_b=float(spx_gex.get("net_gex_total_b", 0.0)),
            pc_ratio_0dte=float(spx_gex.get("pc_ratio_0dte_vol", 1.0)),
            zero_gex_pivot=float(spy_gex.get("zero_gex_pivot") or spy["spot"]),
            call_wall=float(spy_gex.get("call_wall") or spy["spot"] * 1.005),
            put_wall=float(spy_gex.get("put_wall") or spy["spot"] * 0.995),
            vix_level=float(macro["vix"]),
            vix_change_pct=float(macro["vix_change_pct"]),
            us10y_yield=float(macro["us10y"]),
            us10y_change_bps=float(macro["us10y_change_bps"]),
            dxy_index=float(macro["dxy"]),
            dxy_change_pct=float(macro["dxy_change_pct"]),
            news_headlines=news_res["items"],
            provenance=provenance,
        )

        # Build aligned SPY & SPX trade plan views using exact spot ratio
        ratio = float(spx["spot"]) / max(float(spy["spot"]), 1e-6)
        spy_plan = consensus["trade_plan"]
        spx_step = 5.0
        atm_spx = int(round(spx["spot"] / spx_step) * spx_step)
        spx_tp1 = round(spy_plan["tp1"] * ratio, 2)
        spx_tp2 = round(spy_plan["tp2"] * ratio, 2)
        spx_sl = round(spy_plan["stop_loss"] * ratio, 2)
        spx_entry = round(float(spx["spot"]), 2)
        spx_atr = round(spy_plan["atr_14"] * ratio, 2)

        if spy_plan["raw_direction"] == "LONG":
            spx_mandate = (
                f"BULL CALL SPREAD (0DTE SPX {atm_spx}/{atm_spx + 25} C)"
                if spy_plan["signal"] == "LONG"
                else spy_plan["mandate"]
            )
        elif spy_plan["raw_direction"] == "SHORT":
            spx_mandate = (
                f"BEAR PUT SPREAD (0DTE SPX {atm_spx}/{atm_spx - 25} P)"
                if spy_plan["signal"] == "SHORT"
                else spy_plan["mandate"]
            )
        else:
            spx_mandate = spy_plan["mandate"]

        spx_plan = {
            **spy_plan,
            "entry": spx_entry,
            "stop_loss": spx_sl,
            "tp1": spx_tp1,
            "tp2": spx_tp2,
            "atr_14": spx_atr,
            "mandate": spx_mandate,
        }

        # Scale Conformal 90% and TimeMachine fan cleanly for both SPY and SPX views
        spy_conformal = consensus["conformal_prediction_90"]
        spx_conformal = {
            **spy_conformal,
            "lower_bound": round(spy_conformal["lower_bound"] * ratio, 2),
            "upper_bound": round(spy_conformal["upper_bound"] * ratio, 2),
            "half_width_pts": round(spy_conformal["half_width_pts"] * ratio, 2),
            "width_pts": round(spy_conformal["width_pts"] * ratio, 2),
        }

        tm_spy = consensus["timemachine_simulation"]
        fan_spy = tm_spy.get("fan_series", {})
        tm_spx_fan = {
            "steps": fan_spy.get("steps", []),
            "p5": [round(x * ratio, 2) for x in fan_spy.get("p5", [])],
            "p25": [round(x * ratio, 2) for x in fan_spy.get("p25", [])],
            "p50": [round(x * ratio, 2) for x in fan_spy.get("p50", [])],
            "p75": [round(x * ratio, 2) for x in fan_spy.get("p75", [])],
            "p95": [round(x * ratio, 2) for x in fan_spy.get("p95", [])],
        }

        # Evaluate open forward ledger positions against real current bar OHLC
        spx_candles = spx.get("candles", [])
        latest_hi = spx_candles[-1]["high"] if spx_candles else spx["spot"]
        latest_lo = spx_candles[-1]["low"] if spx_candles else spx["spot"]
        mtm = self.position_tracker.update_mark_to_market(
            current_spot=float(spx["spot"]), bar_high=latest_hi, bar_low=latest_lo
        )
        ledger = self.position_tracker.get_ledger_metrics()

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "provenance_detail": snap["provenance_detail"],
            "fetched_at_utc": snap["fetched_at_utc"],
            "server_cache_age_sec": 0,
            "spx": {
                "spot": spx["spot"],
                "prev_close": spx["prev_close"],
                "change_pts": spx["change_pts"],
                "change_pct": spx["change_pct"],
                "high": spx["high"],
                "low": spx["low"],
                "open": spx["open"],
                "candles": spx_candles,
                "gex": spx_gex,
                "trade_plan": spx_plan,
                "conformal_90": spx_conformal,
                "fan_series": tm_spx_fan,
            },
            "spy": {
                "spot": spy["spot"],
                "prev_close": spy["prev_close"],
                "change_pts": spy["change_pts"],
                "change_pct": spy["change_pct"],
                "high": spy["high"],
                "low": spy["low"],
                "open": spy["open"],
                "candles": spy.get("candles", []),
                "gex": spy_gex,
                "trade_plan": spy_plan,
                "conformal_90": spy_conformal,
                "fan_series": fan_spy,
            },
            "macro": macro,
            "probabilities": consensus["probabilities"],
            "feature_contributions": consensus["feature_contributions"],
            "consensus_signal": consensus["consensus_signal"],
            "consensus_score": consensus["consensus_score"],
            "final_actionable_signal": consensus["final_actionable_signal"],
            "timemachine_simulation": tm_spy,
            "critic_audit": consensus["critic_audit"],
            "intermarket_macro": consensus["intermarket_macro"],
            "microstructure": consensus["microstructure"],
            "agent_reasoning": consensus["agent_reasoning"],
            "committee_debate_log": consensus["committee_debate_log"],
            "news": {
                "provenance": news_res["provenance"],
                "sentiment_score": consensus["sentiment_engine"]["score"],
                "sentiment_label": consensus["sentiment_engine"]["label"],
                "items": news_res["items"],
            },
            "positions": mtm,
            "ledger": ledger,
        }

        with self._lock:
            self._cached_payload = payload
            self._cached_at = time.time()
        return dict(payload)

    def get_backtest_report(self) -> Dict[str, Any]:
        if BACKTEST_REPORT_PATH.exists():
            try:
                data = json.loads(BACKTEST_REPORT_PATH.read_text(encoding="utf-8"))
                return {"available": True, **data}
            except Exception as exc:
                return {"available": False, "error": str(exc)}
        return {
            "available": False,
            "message": "No stored hold-out backtest report found. Run 'python run_all.py' first.",
        }

    def get_validation_report(self) -> Dict[str, Any]:
        if VALIDATION_REPORT_PATH.exists():
            try:
                data = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
                return {"available": True, **data}
            except Exception as exc:
                return {"available": False, "error": str(exc)}
        return {"available": False, "message": "Run 'python run_all.py' to generate validation report."}


telemetry_engine = TelemetryEngine()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class QuantApiHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel_path: str) -> None:
        safe_rel = rel_path.lstrip("/").replace("\\", "/")
        if ".." in safe_rel:
            self.send_error(403, "Forbidden")
            return
        file_path = (STATIC_DIR / safe_rel).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
            self.send_error(404, "File not found")
            return
        ctype, _ = mimetypes.guess_type(str(file_path))
        if file_path.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/telemetry":
            force = qs.get("refresh", ["0"])[0] == "1"
            self._send_json(telemetry_engine.get_telemetry(force_refresh=force))
        elif path == "/api/backtest":
            self._send_json(telemetry_engine.get_backtest_report())
        elif path == "/api/validation":
            self._send_json(telemetry_engine.get_validation_report())
        elif path == "/api/gex":
            t = telemetry_engine.get_telemetry()
            self._send_json(
                {
                    "provenance": t["provenance"],
                    "spx": t["spx"]["gex"],
                    "spy": t["spy"]["gex"],
                }
            )
        elif path == "/api/committee":
            t = telemetry_engine.get_telemetry()
            self._send_json(
                {
                    "provenance": t["provenance"],
                    "consensus_signal": t["consensus_signal"],
                    "final_actionable_signal": t["final_actionable_signal"],
                    "consensus_score": t["consensus_score"],
                    "probabilities": t["probabilities"],
                    "feature_contributions": t["feature_contributions"],
                    "agent_reasoning": t["agent_reasoning"],
                    "debate_log": t["committee_debate_log"],
                    "trade_plan_spx": t["spx"]["trade_plan"],
                    "trade_plan_spy": t["spy"]["trade_plan"],
                }
            )
        elif path == "/api/ledger":
            self._send_json(telemetry_engine.position_tracker.get_ledger_metrics())
        elif path == "/api/positions":
            t = telemetry_engine.get_telemetry()
            self._send_json(t["positions"])
        elif path == "/api/agents":
            t = telemetry_engine.get_telemetry()
            self._send_json({"agents": t["agent_reasoning"]})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body_data: Dict[str, Any] = {}
        if length > 0:
            try:
                body_data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                body_data = {}

        if path == "/api/simulate":
            t = telemetry_engine.get_telemetry()
            inst = str(body_data.get("instrument", "SPX")).upper()
            section = t["spy"] if inst == "SPY" else t["spx"]
            plan = section["trade_plan"]
            direction = body_data.get("direction") or (plan["raw_direction"] if plan["raw_direction"] in ("LONG", "SHORT") else "LONG")
            target_pct = float(body_data.get("target_pct", abs(plan["tp1"] - plan["entry"]) / max(plan["entry"], 1.0)))
            stop_pct = float(body_data.get("stop_pct", abs(plan["entry"] - plan["stop_loss"]) / max(plan["entry"], 1.0)))
            paths = min(10000, max(500, int(body_data.get("paths", 5000))))

            snap = telemetry_engine.snapshot_service.get_snapshot()
            hist_df = snap["spy"].get("df_history") if inst == "SPY" else snap["spx"].get("df_history")

            sim = telemetry_engine.committee_agent.timemachine_agent.simulate_trade(
                entry_price=float(section["spot"]),
                target_pct=max(0.0005, target_pct),
                stop_pct=max(0.0005, stop_pct),
                horizon_bars=12,
                paths=paths,
                direction=direction,
                historical_df=hist_df,
            )
            self._send_json(sim)

        elif path == "/api/ledger/record":
            t = telemetry_engine.get_telemetry()
            inst = str(body_data.get("instrument", "SPX")).upper()
            section = t["spy"] if inst == "SPY" else t["spx"]
            plan = dict(section["trade_plan"])
            # Allow recording directional setup only when trade_plan is actionable or explicitly overridden as manual direction
            if body_data.get("allow_bias") and plan["signal"] == "NO_TRADE" and plan["raw_direction"] in ("LONG", "SHORT"):
                plan["signal"] = plan["raw_direction"]
            res = telemetry_engine.position_tracker.record_signal(
                plan, provenance=t["provenance"], instrument_label=inst
            )
            res["ledger"] = telemetry_engine.position_tracker.get_ledger_metrics()
            self._send_json(res)

        elif path == "/api/ledger/clear":
            res = telemetry_engine.position_tracker.clear_ledger()
            self._send_json({"cleared": True, "ledger": res})
        else:
            self._send_json({"error": "Not found"}, status=404)


def run_web_server(port: int = 8080, host: str = "127.0.0.1") -> None:
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, QuantApiHandler)
    print("=" * 72)
    print("  AURA-500 Quantitative Trading Prototype (Offline-Ready UI)")
    print(f"  Dashboard URL : http://{host}:{port}")
    print("  Endpoints     : /api/telemetry, /api/backtest, /api/validation,")
    print("                  /api/committee, /api/gex, /api/ledger, /api/simulate")
    print("=" * 72)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Server shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
    host_arg = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    run_web_server(port=port_arg, host=host_arg)

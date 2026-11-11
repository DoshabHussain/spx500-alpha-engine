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

import gzip
import hashlib
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
        self._cached_json_bytes: Optional[bytes] = None
        self._cached_gzip_bytes: Optional[bytes] = None
        self._bt_cache: Optional[Dict[str, Any]] = None
        self._bt_mtime: float = 0.0
        self._bt_json_bytes: Optional[bytes] = None
        self._bt_gzip_bytes: Optional[bytes] = None
        self._bg_refresh_lock = threading.Lock()
        self._bg_refresh_running = False
        self._news_cache: Optional[List[Dict[str, Any]]] = None
        self._news_at: float = 0.0
        self._news_provenance: str = "DEMO"

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

    def _serialize_cache_locked(self, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        gz = gzip.compress(raw, compresslevel=5)
        self._cached_payload = payload
        self._cached_json_bytes = raw
        self._cached_gzip_bytes = gz

    def sync_ledger_into_cache(self) -> Dict[str, Any]:
        """Fast <0.5ms ledger/positions update in memory without recomputing ML/bootstrap."""
        with self._lock:
            ledger = self.position_tracker.get_ledger_metrics()
            if self._cached_payload is not None:
                spx = self._cached_payload.get("spx", {})
                spy = self._cached_payload.get("spy", {})
                spx_candles = spx.get("candles", [])
                spy_candles = spy.get("candles", [])
                market_bars = {
                    "SPX": {
                        "spot": float(spx.get("spot", 0.0)),
                        "high": float(spx_candles[-1]["high"]) if spx_candles else float(spx.get("spot", 0.0)),
                        "low": float(spx_candles[-1]["low"]) if spx_candles else float(spx.get("spot", 0.0)),
                    },
                    "SPY": {
                        "spot": float(spy.get("spot", 0.0)),
                        "high": float(spy_candles[-1]["high"]) if spy_candles else float(spy.get("spot", 0.0)),
                        "low": float(spy_candles[-1]["low"]) if spy_candles else float(spy.get("spot", 0.0)),
                    },
                }
                mtm = self.position_tracker.update_mark_to_market(
                    current_spot=float(spx.get("spot", 0.0)),
                    market_bars=market_bars,
                )
                ledger = self.position_tracker.get_ledger_metrics()
                self._cached_payload["positions"] = mtm
                self._cached_payload["ledger"] = ledger
                self._serialize_cache_locked(self._cached_payload)
            return ledger

    def trigger_background_refresh(self) -> None:
        """Refresh delayed market snapshot and ML telemetry off the HTTP request thread."""
        with self._bg_refresh_lock:
            if self._bg_refresh_running:
                return
            self._bg_refresh_running = True

        def _worker() -> None:
            try:
                self._build_and_cache_telemetry(force_network=True)
            except Exception:
                pass
            finally:
                with self._bg_refresh_lock:
                    self._bg_refresh_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def get_telemetry(self, force_refresh: bool = False, wait_sync: bool = False) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            if self._cached_payload is not None and not wait_sync:
                age = now - self._cached_at
                self._cached_payload["server_cache_age_sec"] = int(age)
                self._cached_payload["refresh_in_flight"] = self._bg_refresh_running
                self._cached_payload["ledger"] = self.position_tracker.get_ledger_metrics()
                if force_refresh or age >= 20:
                    self.trigger_background_refresh()
                return dict(self._cached_payload)
        return self._build_and_cache_telemetry(force_network=force_refresh)

    def get_telemetry_bytes(self, force_refresh: bool = False, accept_gzip: bool = False) -> bytes:
        """Return pre-serialized telemetry JSON (or gzip) bytes in <1ms when warm."""
        self.get_telemetry(force_refresh=force_refresh, wait_sync=False)
        with self._lock:
            if self._cached_payload is not None:
                new_age = int(time.time() - self._cached_at)
                if self._cached_json_bytes is None or self._cached_payload.get("server_cache_age_sec") != new_age:
                    self._cached_payload["server_cache_age_sec"] = new_age
                    self._cached_payload["ledger"] = self.position_tracker.get_ledger_metrics()
                    self._serialize_cache_locked(self._cached_payload)
                if accept_gzip and self._cached_gzip_bytes is not None:
                    return self._cached_gzip_bytes
                if self._cached_json_bytes is not None:
                    return self._cached_json_bytes
        return json.dumps(self._build_and_cache_telemetry(force_network=force_refresh)).encode("utf-8")

    def _build_and_cache_telemetry(self, force_network: bool = False) -> Dict[str, Any]:

        if self.forecaster is None:
            self.forecaster = self._load_forecaster()
            if self.forecaster is not None:
                self.committee_agent.forecaster = self.forecaster

        snap = self.snapshot_service.get_snapshot(force_refresh=force_network)
        news_res = self._get_news(force=force_network)

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

        # Evaluate open forward ledger positions against real per-instrument bar OHLC
        spx_candles = spx.get("candles", [])
        spy_candles = spy.get("candles", [])
        latest_hi = spx_candles[-1]["high"] if spx_candles else spx["spot"]
        latest_lo = spx_candles[-1]["low"] if spx_candles else spx["spot"]
        spy_hi = spy_candles[-1]["high"] if spy_candles else spy["spot"]
        spy_lo = spy_candles[-1]["low"] if spy_candles else spy["spot"]
        market_bars = {
            "SPX": {"spot": float(spx["spot"]), "high": float(latest_hi), "low": float(latest_lo)},
            "SPY": {"spot": float(spy["spot"]), "high": float(spy_hi), "low": float(spy_lo)},
        }
        mtm = self.position_tracker.update_mark_to_market(
            current_spot=float(spx["spot"]),
            bar_high=latest_hi,
            bar_low=latest_lo,
            market_bars=market_bars,
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
            self._cached_at = time.time()
            self._serialize_cache_locked(payload)
        return dict(payload)

    def get_backtest_report(self) -> Dict[str, Any]:
        if BACKTEST_REPORT_PATH.exists():
            try:
                mtime = BACKTEST_REPORT_PATH.stat().st_mtime
                with self._lock:
                    if self._bt_cache is not None and mtime == self._bt_mtime:
                        return self._bt_cache
                    data = json.loads(BACKTEST_REPORT_PATH.read_text(encoding="utf-8"))
                    out = {"available": True, **data}
                    raw = json.dumps(out, separators=(",", ":")).encode("utf-8")
                    self._bt_cache = out
                    self._bt_mtime = mtime
                    self._bt_json_bytes = raw
                    self._bt_gzip_bytes = gzip.compress(raw, compresslevel=5)
                    return out
            except Exception as exc:
                return {"available": False, "error": str(exc)}
        return {
            "available": False,
            "message": "No stored hold-out backtest report found. Run 'python run_all.py' first.",
        }

    def get_backtest_bytes(self, accept_gzip: bool = False) -> Optional[bytes]:
        report = self.get_backtest_report()
        with self._lock:
            if report.get("available") and self._bt_json_bytes is not None:
                return self._bt_gzip_bytes if (accept_gzip and self._bt_gzip_bytes is not None) else self._bt_json_bytes
        return None

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
    allow_reuse_address = True


_STATIC_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_cached_static(safe_rel: str) -> Optional[Dict[str, Any]]:
    cached = _STATIC_MEMORY_CACHE.get(safe_rel)
    file_path = (STATIC_DIR / safe_rel).resolve()
    if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
        return None
    mtime = file_path.stat().st_mtime
    if cached is not None and cached["mtime"] == mtime:
        return cached
    ctype, _ = mimetypes.guess_type(str(file_path))
    if file_path.suffix == ".js":
        ctype = "application/javascript; charset=utf-8"
    elif file_path.suffix == ".css":
        ctype = "text/css; charset=utf-8"
    elif file_path.suffix == ".html":
        ctype = "text/html; charset=utf-8"
    raw = file_path.read_bytes()
    gz = gzip.compress(raw, compresslevel=5)
    etag = f'"{hashlib.md5(raw).hexdigest()}"'
    entry = {"mtime": mtime, "ctype": ctype or "application/octet-stream", "raw": raw, "gzip": gz, "etag": etag}
    _STATIC_MEMORY_CACHE[safe_rel] = entry
    return entry


class QuantApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _accepts_gzip(self) -> bool:
        return "gzip" in str(self.headers.get("Accept-Encoding", "")).lower()

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = 200,
        is_gzip: bool = False,
        cache_control: str = "no-store",
        etag: Optional[str] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        if etag:
            self.send_header("ETag", etag)
        if is_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        use_gzip = self._accepts_gzip() and len(raw) > 512
        body = gzip.compress(raw, compresslevel=5) if use_gzip else raw
        self._send_bytes(body, "application/json; charset=utf-8", status=status, is_gzip=use_gzip)

    def _serve_static(self, rel_path: str) -> None:
        safe_rel = rel_path.lstrip("/").replace("\\", "/")
        if ".." in safe_rel:
            self._send_bytes(b"Forbidden", "text/plain", status=403)
            return
        entry = _get_cached_static(safe_rel)
        if entry is None:
            self._send_bytes(b"File not found", "text/plain", status=404)
            return
        inm = self.headers.get("If-None-Match")
        if inm and inm.strip() == entry["etag"]:
            self.send_response(304)
            self.send_header("ETag", entry["etag"])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        is_vendor = safe_rel.startswith("vendor/")
        cache_hdr = "public, max-age=86400" if is_vendor else "public, max-age=5, must-revalidate"
        use_gzip = self._accepts_gzip()
        body = entry["gzip"] if use_gzip else entry["raw"]
        self._send_bytes(body, entry["ctype"], status=200, is_gzip=use_gzip, cache_control=cache_hdr, etag=entry["etag"])

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
            gz = self._accepts_gzip()
            body = telemetry_engine.get_telemetry_bytes(force_refresh=force, accept_gzip=gz)
            self._send_bytes(body, "application/json; charset=utf-8", status=200, is_gzip=gz)
        elif path == "/api/backtest":
            gz = self._accepts_gzip()
            cached_bt = telemetry_engine.get_backtest_bytes(accept_gzip=gz)
            if cached_bt is not None:
                self._send_bytes(cached_bt, "application/json; charset=utf-8", status=200, is_gzip=gz)
            else:
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
            seed_val = int(body_data.get("seed", int(time.time() * 1000) % 100000))
            sim_engine = telemetry_engine.committee_agent.timemachine_agent.__class__(default_paths=paths, seed=seed_val)
            sim = sim_engine.simulate_trade(
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
            if plan["signal"] == "NO_TRADE" and body_data.get("allow_bias", False):
                if plan.get("raw_direction") in ("LONG", "SHORT"):
                    plan["signal"] = plan["raw_direction"]
                    plan["mandate"] = f"{plan.get('mandate', 'BIAS')} [Manual Override]"
                else:
                    # Fallback to probability sign when strictly neutral
                    probs = t.get("probabilities", {})
                    plan["signal"] = "LONG" if probs.get("p_up", 0.5) >= probs.get("p_down", 0.5) else "SHORT"
                    plan["mandate"] = f"{inst} PAPER {plan['signal']} [Manual Override]"
            res = telemetry_engine.position_tracker.record_signal(
                plan, provenance=t["provenance"], instrument_label=inst
            )
            res["ledger"] = telemetry_engine.sync_ledger_into_cache()
            self._send_json(res)

        elif path == "/api/ledger/manual":
            t = telemetry_engine.get_telemetry()
            inst = str(body_data.get("instrument", "SPX")).upper()
            section = t["spy"] if inst == "SPY" else t["spx"]
            spot = float(section["spot"])
            direction = str(body_data.get("direction", "LONG")).upper()
            if direction not in ("LONG", "SHORT"):
                direction = "LONG"
            atr = max(float(section["trade_plan"].get("atr_14", spot * 0.002)), 0.25)
            entry = float(body_data.get("entry_price") or spot)
            default_stop = round(entry - atr if direction == "LONG" else entry + atr, 2)
            default_tp = round(entry + 1.5 * atr if direction == "LONG" else entry - 1.5 * atr, 2)
            stop_px = float(body_data.get("stop") or default_stop)
            tp_px = float(body_data.get("target") or default_tp)
            manual_plan = {
                "signal": direction,
                "entry": round(entry, 2),
                "stop_loss": round(stop_px, 2),
                "tp1": round(tp_px, 2),
                "tp2": round(tp_px, 2),
                "mandate": str(body_data.get("setup") or f"MANUAL {inst} {direction}"),
                "reason": "Logged manually from Forward Ledger drawer",
            }
            res = telemetry_engine.position_tracker.record_signal(
                manual_plan, provenance=t["provenance"], instrument_label=inst
            )
            res["ledger"] = telemetry_engine.sync_ledger_into_cache()
            self._send_json(res)

        elif path == "/api/ledger/update":
            trade_id = str(body_data.get("trade_id", ""))
            action = str(body_data.get("action", "CLOSE_MARKET"))
            t = telemetry_engine.get_telemetry()
            inst = str(body_data.get("instrument", "SPX")).upper()
            market_px = float(t["spy"]["spot"] if inst == "SPY" else t["spx"]["spot"])
            exit_px = body_data.get("exit_price")
            res = telemetry_engine.position_tracker.update_trade_action(
                trade_id=trade_id,
                action=action,
                exit_price=float(exit_px) if exit_px is not None else market_px,
                new_stop=body_data.get("new_stop"),
                new_target=body_data.get("new_target"),
            )
            res["ledger"] = telemetry_engine.sync_ledger_into_cache()
            self._send_json(res)

        elif path == "/api/ledger/evaluate":
            ledger = telemetry_engine.sync_ledger_into_cache()
            t = telemetry_engine.get_telemetry()
            self._send_json({"evaluated": True, "ledger": ledger, "positions": t.get("positions", {})})

        elif path == "/api/ledger/clear":
            res = telemetry_engine.position_tracker.clear_ledger()
            res = telemetry_engine.sync_ledger_into_cache()
            self._send_json({"cleared": True, "ledger": res})
        else:
            self._send_json({"error": "Not found"}, status=404)


def run_web_server(port: int = 8080, host: str = "127.0.0.1") -> None:
    # Pre-warm static assets + telemetry + backtest cache so browser loads complete in milliseconds
    for name in ("index.html", "styles.css", "app.js", "vendor/lightweight-charts.standalone.production.js", "vendor/chart.umd.min.js"):
        _get_cached_static(name)
    telemetry_engine.get_backtest_report()
    telemetry_engine.get_telemetry(force_refresh=False, wait_sync=True)

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

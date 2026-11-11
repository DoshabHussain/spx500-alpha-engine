"""
Real Market Data & Options Microstructure Ingestion Engine.

Fetches genuine delayed market data (zero API key / zero cookie flow required):
1. Yahoo Finance Chart v8 API: 5-minute OHLCV for SPY, ^GSPC (SPX), ^VIX, ^TNX (US 10Y), DX-Y.NYB (DXY).
2. Cboe Delayed Options Quotes API: Real strike-level SPX & SPY option chains (OI, Volume, Gamma, IV, Delta)
   used to compute real Net Dealer Gamma Exposure (GEX), Zero-GEX pivot, Call/Put walls, and P/C ratios.
3. Persistent Local Real-Data Cache (data/real/): Allows the pipeline and web UI to operate on real
   historical data even when offline, tagged honestly as DELAYED / CACHED_REAL / DEMO.
"""
from __future__ import annotations

import json
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from src.core import get_logger
from src.processing.features.microstructure_gex import MicrostructureGEXCalculator

logger = get_logger(__name__)
NY_TZ = pytz.timezone("America/New_York")

DEFAULT_REAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "real"

# Regex to decode OCC option ticker: e.g. SPX261120C07500000 or SPXW261111P07450000
_OCC_PATTERN = re.compile(r"^([A-Z]+?)(\d{6})([CP])(\d{8})$")


def _http_get_json(url: str, timeout: int = 8) -> Dict[str, Any]:
    """Fetch JSON over HTTPS with browser User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_yahoo_chart_bars(
    symbol: str,
    range_str: str = "60d",
    interval: str = "5m",
    rth_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch historical 5m OHLCV bars and quote metadata from Yahoo Finance v8 chart API.
    """
    encoded = urllib.parse.quote(symbol, safe="")
    last_err: Optional[Exception] = None
    payload: Optional[Dict[str, Any]] = None

    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{encoded}?range={range_str}&interval={interval}"
        try:
            payload = _http_get_json(url, timeout=10)
            break
        except Exception as exc:
            last_err = exc

    if payload is None:
        raise RuntimeError(f"Failed to fetch Yahoo chart for {symbol}: {last_err}")

    results = payload.get("chart", {}).get("result", [])
    if not results:
        raise RuntimeError(f"Empty Yahoo chart result for {symbol}")

    res = results[0]
    meta = res.get("meta", {})
    timestamps = res.get("timestamp", []) or []
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]

    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    rows = []
    for ts, op, hi, lo, cl, vol in zip(timestamps, opens, highs, lows, closes, volumes):
        if op is None or cl is None:
            continue
        dt_ny = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(NY_TZ)
        if rth_only and symbol in ("SPY", "^GSPC", "QQQ"):
            hm = dt_ny.hour * 60 + dt_ny.minute
            if hm < 9 * 60 + 30 or hm > 16 * 60:
                continue
        hi_val = float(hi if hi is not None else max(op, cl))
        lo_val = float(lo if lo is not None else min(op, cl))
        vol_val = int(vol) if vol is not None else 0
        rows.append(
            {
                "timestamp": dt_ny,
                "symbol": symbol.replace("^", ""),
                "open": float(op),
                "high": hi_val,
                "low": lo_val,
                "close": float(cl),
                "volume": vol_val,
                "interval": interval,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        # Compute daily cumulative intraday VWAP (strictly causal)
        dates = df["timestamp"].dt.date
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol_safe = df["volume"].where(df["volume"] > 0, 1)
        cum_pv = (typical * vol_safe).groupby(dates).cumsum()
        cum_v = vol_safe.groupby(dates).cumsum()
        df["vwap"] = (cum_pv / cum_v).round(4)

    spot = float(meta.get("regularMarketPrice") or (df["close"].iloc[-1] if not df.empty else 0.0))
    prev_close = float(
        meta.get("chartPreviousClose")
        or meta.get("previousClose")
        or (df["close"].iloc[0] if not df.empty else spot)
    )
    market_time = meta.get("regularMarketTime")
    market_ts_iso = (
        datetime.fromtimestamp(market_time, tz=timezone.utc).isoformat()
        if market_time
        else datetime.now(timezone.utc).isoformat()
    )

    info = {
        "symbol": symbol,
        "spot": round(spot, 4),
        "prev_close": round(prev_close, 4),
        "regular_market_time_utc": market_ts_iso,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "bars": len(df),
        "source": "Yahoo Finance Chart v8 (Delayed)",
    }
    return df, info


def fetch_cboe_gex_snapshot(symbol: str = "_SPX") -> Dict[str, Any]:
    """
    Fetch real Cboe delayed option chain quotes and compute genuine strike-level
    Net Dealer Gamma Exposure (GEX), Zero-GEX pivot, Call/Put walls, and P/C ratios.
    """
    clean_sym = symbol.lstrip("_")
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    raw = _http_get_json(url, timeout=12)
    data = raw.get("data", {})
    options = data.get("options", [])
    spot = float(data.get("current_price") or data.get("close") or 0.0)
    if spot <= 0 or not options:
        raise RuntimeError(f"Invalid Cboe options response for {symbol}")

    min_strike = spot * 0.94
    max_strike = spot * 1.06
    parsed_rows: List[Dict[str, Any]] = []

    for opt in options:
        ticker = opt.get("option", "")
        m = _OCC_PATTERN.match(ticker)
        if not m:
            continue
        root_sym, exp_str, cp, strike_raw = m.groups()
        strike = int(strike_raw) / 1000.0
        if strike < min_strike or strike > max_strike:
            continue
        oi = float(opt.get("open_interest") or 0.0)
        gamma = float(opt.get("gamma") or 0.0)
        vol = float(opt.get("volume") or 0.0)
        iv = float(opt.get("iv") or 0.0)
        if oi <= 0 and vol <= 0:
            continue
        parsed_rows.append(
            {
                "root": root_sym,
                "expiry": f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}",
                "strike": strike,
                "type": "CALL" if cp == "C" else "PUT",
                "open_interest": oi,
                "volume": vol,
                "gamma": gamma,
                "iv": iv,
            }
        )

    if not parsed_rows:
        raise RuntimeError(f"No near-spot contracts parsed for {symbol}")

    chain_df = pd.DataFrame(parsed_rows)
    calc = MicrostructureGEXCalculator(contract_multiplier=100.0)
    # Note: Dealer GEX per 1% spot move uses S^2 * 0.01 * gamma * OI * 100;
    # MicrostructureGEXCalculator computes OI * gamma * S^2 * 100 / 1e9.
    # For SPX index (~7500), dividing by 1e9 yields raw $ per 1pt / scaling; multiply by 0.01 for per-1% move ($B).
    gex_df = calc.compute_strike_gex(chain_df, spot_price=spot)
    gex_df["gex_pct_move_m"] = gex_df["gex"] * 0.01 / 1e6  # $ Millions per 1% spot move
    gex_df["gex"] = gex_df["gex"] * 0.01  # Normalize to $ per 1% move before aggregate divides by 1e9

    metrics = calc.aggregate_gex_metrics(gex_df, spot_price=spot, atm_bandwidth=spot * 0.005)

    # Compute grouped strike chart around ATM (+/- 1.5%)
    step = 25.0 if spot > 2000 else 2.5
    gex_df["strike_bucket"] = (gex_df["strike"] / step).round() * step
    by_strike = (
        gex_df.groupby(["strike_bucket", "type"])["gex_pct_move_m"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    by_strike["call_gex_m"] = by_strike.get("CALL", 0.0).round(2)
    by_strike["put_gex_m"] = by_strike.get("PUT", 0.0).round(2)  # negative values
    by_strike["net_gex_m"] = (by_strike["call_gex_m"] + by_strike["put_gex_m"]).round(2)

    near_mask = (by_strike["strike_bucket"] >= spot * 0.98) & (by_strike["strike_bucket"] <= spot * 1.02)
    near_strikes = by_strike.loc[near_mask].sort_values("strike_bucket")
    if near_strikes.empty:
        near_strikes = by_strike.sort_values("strike_bucket").head(12)

    strikes_payload = [
        {
            "strike": float(row["strike_bucket"]),
            "call_gex": float(row["call_gex_m"]),
            "put_gex": float(abs(row["put_gex_m"])),
            "net_gex": float(row["net_gex_m"]),
        }
        for _, row in near_strikes.iterrows()
    ]

    # Call & Put OI Walls
    call_oi = chain_df[chain_df["type"] == "CALL"].groupby("strike")["open_interest"].sum()
    put_oi = chain_df[chain_df["type"] == "PUT"].groupby("strike")["open_interest"].sum()
    call_wall = float(call_oi.idxmax()) if not call_oi.empty else round(spot + step, 2)
    put_wall = float(put_oi.idxmax()) if not put_oi.empty else round(spot - step, 2)

    # Nearest expiry P/C ratio (0DTE / shortest active expiry)
    nearest_exp = sorted(chain_df["expiry"].unique())[0]
    near_exp_df = chain_df[chain_df["expiry"] == nearest_exp]
    c_vol_0 = float(near_exp_df.loc[near_exp_df["type"] == "CALL", "volume"].sum())
    p_vol_0 = float(near_exp_df.loc[near_exp_df["type"] == "PUT", "volume"].sum())
    pc_vol_nearest = round(p_vol_0 / max(c_vol_0, 1.0), 3)

    return {
        "symbol": clean_sym,
        "spot": round(spot, 2),
        "iv30": round(float(data.get("iv30") or 0.0), 2),
        "net_gex_total_b": metrics["net_gex_total_b"],
        "net_gex_atm_b": metrics["net_gex_atm_b"],
        "zero_gex_pivot": metrics["gamma_flip_pivot"],
        "call_wall": call_wall,
        "put_wall": put_wall,
        "nearest_expiry": nearest_exp,
        "pc_ratio_0dte_vol": pc_vol_nearest if c_vol_0 > 0 else metrics["pc_ratio_0dte_vol"],
        "pc_ratio_oi": metrics["pc_ratio_oi"],
        "gamma_regime": metrics["gamma_regime"],
        "contracts_parsed": len(chain_df),
        "strikes": strikes_payload,
        "last_trade_time": data.get("last_trade_time", ""),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Cboe Delayed Options Chain API",
    }


def ingest_and_cache_real_dataset(real_dir: Path = DEFAULT_REAL_DIR) -> Dict[str, Any]:
    """
    Download 60 days of real 5-minute SPY, ^GSPC, ^VIX, ^TNX, DX-Y.NYB bars and Cboe options GEX,
    and save locally under data/real/ for offline reproducibility and training.
    """
    real_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": {},
        "cboe_gex": {},
    }

    symbol_files = {
        "SPY": ("spy_5m.parquet", "60d", True),
        "^GSPC": ("spx_5m.parquet", "60d", True),
        "^VIX": ("vix_5m.parquet", "60d", False),
        "^TNX": ("tnx_5m.parquet", "60d", False),
        "DX-Y.NYB": ("dxy_5m.parquet", "60d", False),
    }

    for sym, (fname, rng, rth) in symbol_files.items():
        df, info = fetch_yahoo_chart_bars(sym, range_str=rng, interval="5m", rth_only=rth)
        out_path = real_dir / fname
        if not df.empty:
            df.to_parquet(out_path, index=False)
        manifest["symbols"][sym] = info
        logger.info("Cached real Yahoo bars", symbol=sym, rows=len(df), path=str(out_path))

    for opt_sym, fname in (("_SPX", "cboe_gex_spx.json"), ("SPY", "cboe_gex_spy.json")):
        try:
            gex_snap = fetch_cboe_gex_snapshot(opt_sym)
            (real_dir / fname).write_text(json.dumps(gex_snap, indent=2), encoding="utf-8")
            manifest["cboe_gex"][opt_sym] = {
                "spot": gex_snap["spot"],
                "net_gex_total_b": gex_snap["net_gex_total_b"],
                "zero_gex_pivot": gex_snap["zero_gex_pivot"],
                "contracts_parsed": gex_snap["contracts_parsed"],
            }
        except Exception as exc:
            logger.warning("Cboe GEX snapshot fetch skipped", symbol=opt_sym, error=str(exc))

    (real_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_cached_real_bars(symbol: str = "SPY", real_dir: Path = DEFAULT_REAL_DIR) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load cached real 5m bars from data/real/. Returns (df, fetched_at_utc).
    """
    fname_map = {
        "SPY": "spy_5m.parquet",
        "^GSPC": "spx_5m.parquet",
        "SPX": "spx_5m.parquet",
        "^VIX": "vix_5m.parquet",
        "VIX": "vix_5m.parquet",
        "^TNX": "tnx_5m.parquet",
        "TNX": "tnx_5m.parquet",
        "DX-Y.NYB": "dxy_5m.parquet",
        "DXY": "dxy_5m.parquet",
    }
    fname = fname_map.get(symbol)
    if not fname:
        return pd.DataFrame(), None

    fpath = real_dir / fname
    if not fpath.exists():
        return pd.DataFrame(), None

    df = pd.read_parquet(fpath)
    fetched_at = None
    manifest_path = real_dir / "manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            fetched_at = m.get("fetched_at_utc")
        except Exception:
            pass
    return df, fetched_at


class LiveMarketSnapshotService:
    """
    Thread-safe snapshot provider combining live delayed Yahoo/Cboe APIs with
    persistent real-data cache (CACHED_REAL) and explicit synthetic DEMO fallback.
    """

    def __init__(self, real_dir: Path = DEFAULT_REAL_DIR, ttl_seconds: int = 20):
        self.real_dir = real_dir
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cached_at: float = 0.0

    def get_snapshot(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            if not force_refresh and self._cached_snapshot and (now - self._cached_at < self.ttl_seconds):
                age_s = int(now - self._cached_at)
                snap = dict(self._cached_snapshot)
                snap["cache_age_sec"] = age_s
                return snap

        # 1. Try fresh Yahoo + Cboe delayed feed
        try:
            snap = self._fetch_fresh_delayed()
            with self._lock:
                self._cached_snapshot = snap
                self._cached_at = time.time()
            return dict(snap)
        except Exception as exc:
            logger.debug("Online delayed feed unavailable, falling back to local cache", error=str(exc))

        # 2. Try cached REAL dataset on disk
        try:
            snap = self._load_from_real_cache()
            if snap is not None:
                with self._lock:
                    self._cached_snapshot = snap
                    self._cached_at = time.time()
                return dict(snap)
        except Exception as exc:
            logger.warning("Failed reading local real cache", error=str(exc))

        # 3. Last resort: Synthetic local bars clearly marked DEMO
        snap = self._load_synthetic_demo()
        with self._lock:
            self._cached_snapshot = snap
            self._cached_at = time.time()
        return dict(snap)

    def _fetch_fresh_delayed(self) -> Dict[str, Any]:
        spy_df, spy_info = fetch_yahoo_chart_bars("SPY", range_str="5d", interval="5m", rth_only=True)
        spx_df, spx_info = fetch_yahoo_chart_bars("^GSPC", range_str="5d", interval="5m", rth_only=True)
        vix_df, vix_info = fetch_yahoo_chart_bars("^VIX", range_str="5d", interval="5m", rth_only=False)
        tnx_df, tnx_info = fetch_yahoo_chart_bars("^TNX", range_str="5d", interval="5m", rth_only=False)
        dxy_df, dxy_info = fetch_yahoo_chart_bars("DX-Y.NYB", range_str="5d", interval="5m", rth_only=False)

        gex_spx = fetch_cboe_gex_snapshot("_SPX")
        gex_spy = fetch_cboe_gex_snapshot("SPY")

        # Determine true daily prev_close from prior session's last bar if multi-day
        spx_prev = self._prior_session_close(spx_df, spx_info["prev_close"])
        spy_prev = self._prior_session_close(spy_df, spy_info["prev_close"])
        vix_prev = self._prior_session_close(vix_df, vix_info["prev_close"])
        tnx_prev = self._prior_session_close(tnx_df, tnx_info["prev_close"])
        dxy_prev = self._prior_session_close(dxy_df, dxy_info["prev_close"])

        return self._assemble_snapshot(
            provenance="DELAYED",
            provenance_detail="Yahoo Chart v8 + Cboe Delayed Options (15m delayed)",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            spy_df=spy_df,
            spx_df=spx_df,
            spx_spot=spx_info["spot"],
            spx_prev=spx_prev,
            spy_spot=spy_info["spot"],
            spy_prev=spy_prev,
            vix_spot=vix_info["spot"],
            vix_prev=vix_prev,
            tnx_spot=tnx_info["spot"],
            tnx_prev=tnx_prev,
            dxy_spot=dxy_info["spot"],
            dxy_prev=dxy_prev,
            gex_spx=gex_spx,
            gex_spy=gex_spy,
        )

    def _load_from_real_cache(self) -> Optional[Dict[str, Any]]:
        spy_df, fetched_at = load_cached_real_bars("SPY", self.real_dir)
        spx_df, _ = load_cached_real_bars("^GSPC", self.real_dir)
        vix_df, _ = load_cached_real_bars("^VIX", self.real_dir)
        tnx_df, _ = load_cached_real_bars("^TNX", self.real_dir)
        dxy_df, _ = load_cached_real_bars("DX-Y.NYB", self.real_dir)

        if spy_df.empty or spx_df.empty:
            return None

        gex_spx_path = self.real_dir / "cboe_gex_spx.json"
        gex_spy_path = self.real_dir / "cboe_gex_spy.json"
        gex_spx = json.loads(gex_spx_path.read_text(encoding="utf-8")) if gex_spx_path.exists() else None
        gex_spy = json.loads(gex_spy_path.read_text(encoding="utf-8")) if gex_spy_path.exists() else None

        spx_spot = float(spx_df["close"].iloc[-1])
        spy_spot = float(spy_df["close"].iloc[-1])
        vix_spot = float(vix_df["close"].iloc[-1]) if not vix_df.empty else 18.0
        tnx_spot = float(tnx_df["close"].iloc[-1]) if not tnx_df.empty else 4.5
        dxy_spot = float(dxy_df["close"].iloc[-1]) if not dxy_df.empty else 101.0

        return self._assemble_snapshot(
            provenance="CACHED_REAL",
            provenance_detail=f"Cached Real Yahoo/Cboe Snapshot ({(fetched_at or '')[:16]} UTC)",
            fetched_at_utc=fetched_at or datetime.now(timezone.utc).isoformat(),
            spy_df=spy_df.tail(400).copy(),
            spx_df=spx_df.tail(400).copy(),
            spx_spot=spx_spot,
            spx_prev=self._prior_session_close(spx_df, spx_spot),
            spy_spot=spy_spot,
            spy_prev=self._prior_session_close(spy_df, spy_spot),
            vix_spot=vix_spot,
            vix_prev=self._prior_session_close(vix_df, vix_spot) if not vix_df.empty else vix_spot,
            tnx_spot=tnx_spot,
            tnx_prev=self._prior_session_close(tnx_df, tnx_spot) if not tnx_df.empty else tnx_spot,
            dxy_spot=dxy_spot,
            dxy_prev=self._prior_session_close(dxy_df, dxy_spot) if not dxy_df.empty else dxy_spot,
            gex_spx=gex_spx,
            gex_spy=gex_spy,
        )

    def _load_synthetic_demo(self) -> Dict[str, Any]:
        from src.storage import get_raw_store
        store = get_raw_store()
        spy_df = store.read_underlying("SPY", datetime(2025, 11, 1), datetime(2026, 4, 1))
        if spy_df.empty:
            from scripts.create_sample_data import generate_intraday_5m_data
            spy_df = generate_intraday_5m_data(n_days=5)

        spy_df = spy_df.tail(390).copy().reset_index(drop=True)
        spx_df = spy_df.copy()
        for col in ("open", "high", "low", "close", "vwap"):
            if col in spx_df.columns:
                spx_df[col] = (spx_df[col] * 10.0).round(2)
        spx_df["symbol"] = "GSPC"

        spy_spot = float(spy_df["close"].iloc[-1])
        spx_spot = float(spx_df["close"].iloc[-1])
        return self._assemble_snapshot(
            provenance="DEMO",
            provenance_detail="SYNTHETIC DEMO DATA (Offline Fallback — Not Real Market Prices)",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            spy_df=spy_df,
            spx_df=spx_df,
            spx_spot=spx_spot,
            spx_prev=self._prior_session_close(spx_df, spx_spot),
            spy_spot=spy_spot,
            spy_prev=self._prior_session_close(spy_df, spy_spot),
            vix_spot=16.50,
            vix_prev=16.50,
            tnx_spot=4.30,
            tnx_prev=4.30,
            dxy_spot=101.00,
            dxy_prev=101.00,
            gex_spx=None,
            gex_spy=None,
        )

    @staticmethod
    def _prior_session_close(df: pd.DataFrame, default_val: float) -> float:
        if df.empty or "timestamp" not in df.columns:
            return default_val
        ts = pd.to_datetime(df["timestamp"])
        dates = ts.dt.date
        unique_dates = sorted(dates.unique())
        if len(unique_dates) >= 2:
            prev_date = unique_dates[-2]
            prev_rows = df[dates == prev_date]
            if not prev_rows.empty:
                return float(prev_rows["close"].iloc[-1])
        return default_val

    def _assemble_snapshot(
        self,
        provenance: str,
        provenance_detail: str,
        fetched_at_utc: str,
        spy_df: pd.DataFrame,
        spx_df: pd.DataFrame,
        spx_spot: float,
        spx_prev: float,
        spy_spot: float,
        spy_prev: float,
        vix_spot: float,
        vix_prev: float,
        tnx_spot: float,
        tnx_prev: float,
        dxy_spot: float,
        dxy_prev: float,
        gex_spx: Optional[Dict[str, Any]],
        gex_spy: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _fmt_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
            if df.empty:
                return []
            # Latest trading day bars (or last 78 bars)
            ts = pd.to_datetime(df["timestamp"])
            last_date = ts.dt.date.max()
            day_df = df[ts.dt.date == last_date]
            if len(day_df) < 12:
                day_df = df.tail(78)
            out = []
            for _, r in day_df.iterrows():
                t_val = pd.Timestamp(r["timestamp"])
                out.append(
                    {
                        "time": int(t_val.timestamp()),
                        "label": t_val.strftime("%H:%M"),
                        "open": round(float(r["open"]), 2),
                        "high": round(float(r["high"]), 2),
                        "low": round(float(r["low"]), 2),
                        "close": round(float(r["close"]), 2),
                        "volume": int(r.get("volume", 0)),
                        "vwap": round(float(r.get("vwap", r["close"])), 2),
                    }
                )
            return out

        def _day_df(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            ts = pd.to_datetime(df["timestamp"])
            last_date = ts.dt.date.max()
            sub = df[ts.dt.date == last_date].copy()
            return sub if len(sub) >= 6 else df.tail(78).copy()

        spx_day = _day_df(spx_df)
        spy_day = _day_df(spy_df)

        spx_chg = spx_spot - spx_prev
        spx_chg_pct = (spx_chg / spx_prev * 100.0) if spx_prev else 0.0
        spy_chg = spy_spot - spy_prev
        spy_chg_pct = (spy_chg / spy_prev * 100.0) if spy_prev else 0.0
        vix_chg_pct = ((vix_spot - vix_prev) / vix_prev * 100.0) if vix_prev else 0.0
        tnx_chg_bps = (tnx_spot - tnx_prev) * 100.0
        dxy_chg_pct = ((dxy_spot - dxy_prev) / dxy_prev * 100.0) if dxy_prev else 0.0

        # Default fallback GEX structure if Cboe file wasn't available
        default_spx_gex = {
            "symbol": "SPX",
            "spot": round(spx_spot, 2),
            "net_gex_total_b": 0.0,
            "zero_gex_pivot": round(spx_spot, 2),
            "call_wall": round(spx_spot + 25.0, 2),
            "put_wall": round(spx_spot - 25.0, 2),
            "pc_ratio_0dte_vol": 1.0,
            "pc_ratio_oi": 1.0,
            "gamma_regime": "UNAVAILABLE_APPROX",
            "strikes": [],
            "source": "Unavailable (Approx)",
        }

        return {
            "provenance": provenance,
            "provenance_detail": provenance_detail,
            "fetched_at_utc": fetched_at_utc,
            "cache_age_sec": 0,
            "spx": {
                "spot": round(spx_spot, 2),
                "prev_close": round(spx_prev, 2),
                "change_pts": round(spx_chg, 2),
                "change_pct": round(spx_chg_pct, 2),
                "high": round(float(spx_day["high"].max()), 2) if not spx_day.empty else round(spx_spot, 2),
                "low": round(float(spx_day["low"].min()), 2) if not spx_day.empty else round(spx_spot, 2),
                "open": round(float(spx_day["open"].iloc[0]), 2) if not spx_day.empty else round(spx_spot, 2),
                "candles": _fmt_candles(spx_df),
                "df_history": spx_df,
                "df_day": spx_day,
                "gex": gex_spx or default_spx_gex,
            },
            "spy": {
                "spot": round(spy_spot, 2),
                "prev_close": round(spy_prev, 2),
                "change_pts": round(spy_chg, 2),
                "change_pct": round(spy_chg_pct, 2),
                "high": round(float(spy_day["high"].max()), 2) if not spy_day.empty else round(spy_spot, 2),
                "low": round(float(spy_day["low"].min()), 2) if not spy_day.empty else round(spy_spot, 2),
                "open": round(float(spy_day["open"].iloc[0]), 2) if not spy_day.empty else round(spy_spot, 2),
                "candles": _fmt_candles(spy_df),
                "df_history": spy_df,
                "df_day": spy_day,
                "gex": gex_spy or {**default_spx_gex, "symbol": "SPY", "spot": round(spy_spot, 2)},
            },
            "macro": {
                "vix": round(vix_spot, 2),
                "vix_prev": round(vix_prev, 2),
                "vix_change_pct": round(vix_chg_pct, 2),
                "us10y": round(tnx_spot, 3),
                "us10y_change_bps": round(tnx_chg_bps, 1),
                "dxy": round(dxy_spot, 2),
                "dxy_change_pct": round(dxy_chg_pct, 2),
            },
        }

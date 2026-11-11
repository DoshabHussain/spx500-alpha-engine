"""
Genuine Forward-Test Paper Ledger & Defined-Risk Position Tracker.
Replaces previous fabricated ledger samples with an honest, empty-start
persistent paper-trading ledger:
- Starts empty (zero invented trades or fake equity gains)
- Thread-safe with threading.Lock()
- Appends OPEN records only when genuine signals are logged
- Resolves OPEN trades conservatively against real bar highs/lows (Stop checked before Target)
- Computes win rate, expectancy (R), profit factor, and equity strictly from resolved records
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz

NY_TZ = pytz.timezone("America/New_York")
DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "runs" / "paper_ledger.jsonl"


class GenesisPositionTracker:
    """
    Thread-safe Forward Paper-Trading Ledger & Defined-Risk OMS.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade_pct: float = 0.01,
        ledger_path: Optional[Path] = DEFAULT_LEDGER_PATH,
    ):
        self.initial_capital = float(initial_capital)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.risk_budget = round(self.initial_capital * self.risk_per_trade_pct, 2)
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._lock = threading.Lock()
        self.trade_ledger: List[Dict[str, Any]] = []
        self._load_persisted_ledger()

    def _load_persisted_ledger(self) -> None:
        if self.ledger_path is None or not self.ledger_path.exists():
            return
        rows: List[Dict[str, Any]] = []
        try:
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            self.trade_ledger = rows
        except Exception:
            self.trade_ledger = []

    def _persist_ledger_locked(self) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(json.dumps(r) for r in self.trade_ledger)
        if content:
            content += "\n"
        self.ledger_path.write_text(content, encoding="utf-8")

    def clear_ledger(self) -> Dict[str, Any]:
        with self._lock:
            self.trade_ledger = []
            self._persist_ledger_locked()
            return self._compute_metrics_locked()

    def record_signal(
        self,
        trade_plan: Dict[str, Any],
        provenance: str = "DELAYED",
        instrument_label: str = "SPX",
    ) -> Dict[str, Any]:
        """
        Record a real trade plan emitted by the Quant Committee as an OPEN forward-test entry.
        Refuses to record NO_TRADE plans.
        """
        signal_dir = str(trade_plan.get("signal", "NO_TRADE")).upper()
        if signal_dir not in ("LONG", "SHORT"):
            return {"recorded": False, "reason": f"Signal '{signal_dir}' is not directional (NO_TRADE)."}

        entry = float(trade_plan["entry"])
        stop_loss = float(trade_plan["stop_loss"])
        tp1 = float(trade_plan["tp1"])
        tp2 = float(trade_plan.get("tp2", tp1))
        risk_pts = abs(entry - stop_loss)
        if risk_pts <= 0:
            return {"recorded": False, "reason": "Zero stop distance."}

        now_ny = datetime.now(NY_TZ)
        with self._lock:
            trade_id = f"{instrument_label}-{now_ny.strftime('%Y%m%d-%H%M%S')}-{len(self.trade_ledger) + 1:02d}"
            record = {
                "trade_id": trade_id,
                "timestamp": now_ny.isoformat(),
                "time": now_ny.strftime("%H:%M:%S ET"),
                "date": now_ny.strftime("%Y-%m-%d"),
                "instrument": instrument_label,
                "setup": trade_plan.get("mandate", signal_dir),
                "direction": signal_dir,
                "entry_price": round(entry, 2),
                "exit_price": None,
                "target": round(tp1, 2),
                "target_2": round(tp2, 2),
                "stop": round(stop_loss, 2),
                "current_stop": round(stop_loss, 2),
                "trailing_active": False,
                "risk_budget_usd": self.risk_budget,
                "return_r": 0.0,
                "pnl_usd": 0.0,
                "outcome": "OPEN",
                "provenance": provenance,
                "reason": trade_plan.get("reason", ""),
            }
            self.trade_ledger.insert(0, record)
            self._persist_ledger_locked()
            return {"recorded": True, "trade": dict(record)}

    def evaluate_against_bar(
        self,
        current_spot: float,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Conservatively evaluate all OPEN forward-test signals against real bar OHLC.
        Stop loss is always checked BEFORE target to avoid optimistic tie bias.
        """
        hi = float(bar_high if bar_high is not None else current_spot)
        lo = float(bar_low if bar_low is not None else current_spot)
        spot = float(current_spot)
        now_ny = datetime.now(NY_TZ)
        changed = False

        with self._lock:
            for rec in self.trade_ledger:
                if rec.get("outcome") != "OPEN":
                    continue
                entry = float(rec["entry_price"])
                stop = float(rec["current_stop"])
                target = float(rec["target"])
                risk_pts = max(abs(entry - float(rec["stop"])), 1e-6)
                risk_usd = float(rec.get("risk_budget_usd", self.risk_budget))

                if rec["direction"] == "LONG":
                    # Stop first (conservative)
                    if lo <= stop:
                        rec["exit_price"] = round(stop, 2)
                        r_mult = round((stop - entry) / risk_pts, 2)
                        rec["return_r"] = max(-1.0, r_mult)
                        rec["pnl_usd"] = round(rec["return_r"] * risk_usd, 2)
                        rec["outcome"] = "BREAKEVEN_STOP" if abs(r_mult) < 0.05 else "STOP_HIT"
                        rec["resolved_at"] = now_ny.isoformat()
                        changed = True
                    elif hi >= target:
                        rec["exit_price"] = round(target, 2)
                        r_mult = round((target - entry) / risk_pts, 2)
                        rec["return_r"] = r_mult
                        rec["pnl_usd"] = round(r_mult * risk_usd, 2)
                        rec["outcome"] = "TARGET_1_HIT"
                        rec["resolved_at"] = now_ny.isoformat()
                        changed = True
                    else:
                        # Unrealized mark-to-market; arm breakeven once >= 50% toward TP1
                        fav = spot - entry
                        if fav >= 0.5 * abs(target - entry) and not rec.get("trailing_active"):
                            rec["current_stop"] = entry
                            rec["trailing_active"] = True
                            changed = True
                        r_live = max(-1.0, round(fav / risk_pts, 2))
                        rec["return_r"] = r_live
                        rec["pnl_usd"] = round(r_live * risk_usd, 2)
                else:  # SHORT
                    if hi >= stop:
                        rec["exit_price"] = round(stop, 2)
                        r_mult = round((entry - stop) / risk_pts, 2)
                        rec["return_r"] = max(-1.0, r_mult)
                        rec["pnl_usd"] = round(rec["return_r"] * risk_usd, 2)
                        rec["outcome"] = "BREAKEVEN_STOP" if abs(r_mult) < 0.05 else "STOP_HIT"
                        rec["resolved_at"] = now_ny.isoformat()
                        changed = True
                    elif lo <= target:
                        rec["exit_price"] = round(target, 2)
                        r_mult = round((entry - target) / risk_pts, 2)
                        rec["return_r"] = r_mult
                        rec["pnl_usd"] = round(r_mult * risk_usd, 2)
                        rec["outcome"] = "TARGET_1_HIT"
                        rec["resolved_at"] = now_ny.isoformat()
                        changed = True
                    else:
                        fav = entry - spot
                        if fav >= 0.5 * abs(entry - target) and not rec.get("trailing_active"):
                            rec["current_stop"] = entry
                            rec["trailing_active"] = True
                            changed = True
                        r_live = max(-1.0, round(fav / risk_pts, 2))
                        rec["return_r"] = r_live
                        rec["pnl_usd"] = round(r_live * risk_usd, 2)

            if changed:
                self._persist_ledger_locked()

            open_trades = [dict(r) for r in self.trade_ledger if r.get("outcome") == "OPEN"]
            metrics = self._compute_metrics_locked()

        return {
            "open_count": len(open_trades),
            "active_trade": open_trades[0] if open_trades else None,
            "current_spot": round(spot, 2),
            "ledger_metrics": metrics,
        }

    def update_mark_to_market(self, current_spot: float, bar_high: Optional[float] = None, bar_low: Optional[float] = None) -> Dict[str, Any]:
        """Wrapper updating open forward trades and returning honest account equity."""
        eval_res = self.evaluate_against_bar(current_spot, bar_high=bar_high, bar_low=bar_low)
        metrics = eval_res["ledger_metrics"]
        active = eval_res["active_trade"]
        unrealized = float(active["pnl_usd"]) if active else 0.0
        r_live = float(active["return_r"]) if active else 0.0

        return {
            "has_active_position": active is not None,
            "active_trade": active,
            "current_spot": round(current_spot, 2),
            "unrealized_pnl_usd": round(unrealized, 2),
            "r_multiple": r_live,
            "account_equity": round(self.initial_capital + metrics["total_realized_pnl"] + unrealized, 2),
            "realized_pnl_usd": metrics["total_realized_pnl"],
            "risk_budget_usd": self.risk_budget,
        }

    def _compute_metrics_locked(self) -> Dict[str, Any]:
        resolved = [t for t in self.trade_ledger if t.get("outcome") != "OPEN"]
        open_list = [t for t in self.trade_ledger if t.get("outcome") == "OPEN"]

        n_resolved = len(resolved)
        wins = [t for t in resolved if float(t.get("pnl_usd", 0.0)) > 0]
        losses = [t for t in resolved if float(t.get("pnl_usd", 0.0)) < 0]

        win_rate = round((len(wins) / n_resolved) * 100.0, 1) if n_resolved > 0 else 0.0
        total_pnl = round(sum(float(t.get("pnl_usd", 0.0)) for t in resolved), 2)
        avg_r = round(sum(float(t.get("return_r", 0.0)) for t in resolved) / n_resolved, 2) if n_resolved > 0 else 0.0
        gross_win = sum(float(t["pnl_usd"]) for t in wins)
        gross_loss = abs(sum(float(t["pnl_usd"]) for t in losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (round(gross_win, 2) if gross_win > 0 else 0.0)

        return {
            "total_signals": len(self.trade_ledger),
            "open_signals": len(open_list),
            "resolved_signals": n_resolved,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": avg_r,
            "total_realized_pnl": total_pnl,
            "account_equity": round(self.initial_capital + total_pnl, 2),
            "recent_trades": [dict(t) for t in self.trade_ledger[:25]],
        }

    def get_ledger_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return self._compute_metrics_locked()

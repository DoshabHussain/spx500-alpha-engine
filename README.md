# SPX500 Alpha Engine (AURA-500 Quant)

**Intraday 30-Minute S&P 500 Forecasting, Options Microstructure Analytics & Transparent Multi-Agent Trading Workbench**

---

## Disclaimer

This software is a quantitative research and paper-trading workbench for educational and empirical evaluation purposes only. It is **not** investment advice and is **not** connected to any live brokerage order router. All intraday quotes fetched via Yahoo Finance Chart v8 and options chains fetched via Cboe Delayed APIs carry standard delayed-feed latency (~15 minutes); offline runs fall back explicitly to cached historical market data (`CACHED_REAL`) or clearly badged synthetic fixtures (`DEMO`).

---

## Architecture & Engineering Principles

1. **Leakage-Free 70% / 30% Chronological Evaluation**:
   - Stage 3 features (`returns.py`, `technical.py`, `regime.py`, `microstructure_gex.py`) use strictly causal trailing windows (e.g., 78-bar rolling reference means and `economic_calendar.csv` event proximity for 2025–2027 FOMC/CPI/NFP schedules).
   - Stage 4 splits canonical 5-minute SPY bars (`4,681` bars from `2026-08-18` to `2026-11-10`) chronologically into **70% Purged Walk-Forward CV** (`3,253` bars, `purge_bars=6`, `2%` embargo) and an **untouched 30% Hold-Out window** (`1,397` bars from `2026-10-16` to `2026-11-10`) separated by a 6-bar label-horizon purge gap (`VERIFIED_ZERO_OVERLAP`).
2. **Real Options Microstructure & Delayed Market Feeds (`src/ingestion/real_market_feed.py`)**:
   - Ingests real 60-day 5-minute bars (`SPY`, `^GSPC`, `^VIX`, `^TNX`, `DX-Y.NYB`) and Cboe delayed option chains (`_SPX.json`, `SPY.json`) to compute Dealer Net GEX (`_SPX` / `SPY`), zero-gamma flip pivot, call/put walls, and 0DTE put/call volume & OI ratios.
   - Maintains mathematical SPX/SPY consistency (~10:1 spot ratio).
3. **Empirically Measured Ensemble, Calibration & Conformal Bounds (`src/modeling/purged_trainer.py`)**:
   - Combines LightGBM, HistGradientBoosting, and RandomForest classifiers over 34 non-leaking features.
   - Reports multiclass Brier score, Expected Calibration Error (ECE), 3x3 confusion matrix, Always-Flat majority baseline, Rolling Momentum baseline, exact binomial p-values, empirical OOF Conformal 90% return quantile bounds, and per-bar LightGBM **TreeSHAP** (`pred_contrib=True`) feature attributions.
4. **Fixed-Grid Out-of-Sample Backtester (`src/modeling/backtest/backtester.py`)**:
   - Enforces pre-computed out-of-sample signals (raises `ValueError` if `signals_col` is absent; zero future-return fallbacks).
   - Evaluates bar-by-bar on the canonical 5-minute grid (`78` bars/day) with `5.0 bps` slippage + `$0.005/share` commission, reporting bootstrap 95% Sharpe Confidence Intervals alongside an equal-window Buy & Hold benchmark curve.
5. **8-Agent Committee Reasoning & Empirical Bootstrap Replay (`src/agents/`)**:
   - **Step 1 — ScannerAgent**: Computes 15m Opening Range (ORB), `ATR(14)`, VWAP distance, and volume surge ratio.
   - **Step 2 — NewsSentimentAgent**: Scores Yahoo Finance RSS headlines (or explicitly marks offline `DEMO` headlines).
   - **Step 3 — CrossAssetIntermarketAgent**: Evaluates live `^VIX`, US 10Y (`^TNX`), and US Dollar (`DX-Y.NYB`).
   - **Step 4 — VannaCharmAgent**: Computes Cboe Net GEX regime, Zero-GEX pivot, Call/Put walls, and empirical Conformal 90% price intervals (`entry ± entry * q90_ret`).
   - **Step 5 — Purged Ensemble + TreeSHAP**: Predicts `P(Up)`, `P(Flat)`, `P(Down)` with top additive TreeSHAP feature drivers (falls back to neutral `0.333 / 0.334 / 0.333` prior if untrained).
   - **Step 6 — AdversarialCriticAgent**: Audits session timing, pin-wall proximity, and breadth divergences.
   - **Step 7 — TimeMachineAgent**: Runs 5,000-path empirical historical 5m OHLC bootstrap simulations supporting both `LONG` and `SHORT` setups, evaluating intrabar stops before targets conservatively with a local NumPy RNG.
   - **Step 8 — RiskAgent**: Enforces `ATR(14)` stop (`1.0x`), `TP1` (`1.5x`), `TP2` (`2.5x`), minimum `1.25R` risk/reward, bootstrap statistical edge, and `1%` equity risk budget (`-2%` daily DD cutoff).
6. **Offline-Ready 4-Tab Pro Trading UI (`src/web/static/`)**:
   - Built with zero external CDNs using vendored **TradingView Lightweight Charts** and **Chart.js** in both Dark and Light themes across 4 views: `Trade Terminal`, `Model Evidence`, `Market & GEX`, and `Forward Ledger`.
   - Every panel displays explicit data provenance (`DELAYED`, `CACHED_REAL`, `DEMO`).
   - **Forward Paper Ledger (`runs/paper_ledger.jsonl`)** starts strictly empty with zero fabricated historical trades.

---

## Verified Benchmark Results (Real 60-Day Dataset: `2026-08-18` to `2026-11-10`)

Measured directly from running `python run_all.py` on cached real 5-minute SPY bars (`4,681` bars, `34` non-leaking features):

| Evaluation Stage | Metric | Observed Value | Baseline / Benchmark |
| --- | --- | --- | --- |
| **Purged Walk-Forward CV (70%, 3,253 bars)** | Mean CV Accuracy (5 folds, `purge_bars=6`) | **54.54%** (95% CI: 52.6%–56.4%) | Always-Flat Majority: 65.46% (p=1.0) |
| **Purged Walk-Forward CV (70%)** | OOF Brier Score / Overlap Check | **0.5784** / `VERIFIED_ZERO_OVERLAP` | Purge Gap = 6 bars |
| **Untouched Chronological Hold-Out (30%, 1,397 bars)** | 30m Classification Accuracy | **47.82%** | Always-Flat: 46.46% · 3-Bar Mom: 37.44% |
| **Untouched Hold-Out (30%)** | Hold-Out Brier Score / ECE | **0.6788** / **0.154** | Untrained Uniform Prior: ~0.6667 |
| **Hold-Out Fixed-Grid Backtest (5.0 bps slippage)** | Total Return (`2026-10-16` -> `2026-11-10`) | **-2.50%** ($100,000 -> $97,499.70) | SPY Buy & Hold: **-2.96%** · Mom: -5.15% |
| **Hold-Out Fixed-Grid Backtest** | Trades / Win Rate / Max Drawdown / PF | 104 trades / 33.65% / -2.54% / 0.45 | Momentum Sharpe: -19.33 |

> Honest reporting note: During the `2026-10-16` to `2026-11-10` hold-out window, S&P 500 realized choppy negative drift (`-2.96%`), and short-horizon 30m equity predictability remains noisy around the majority baseline (`p=1.0`). These exact out-of-sample figures are surfaced openly in the `Model Evidence` tab rather than hidden behind optimistic in-sample numbers.

---

## Installation & Quickstart

### Prerequisites
- Python **>= 3.10** (verified on Python 3.13.15, Windows)

### Setup

```bash
git clone https://github.com/DoshabHussain/spx500-alpha-engine.git
cd spx500-alpha-engine
pip install -e .
```

### 1. Refresh Real Delayed Market & Cboe GEX Cache (Optional when offline)

```bash
python scripts/ingest_real_data.py
```

Populates `data/real/` with 60-day 5m parquets (`SPY`, `^GSPC`, `^VIX`, `^TNX`, `DX-Y.NYB`) and Cboe delayed option chain snapshots (`cboe_gex_spx.json`, `cboe_gex_spy.json`).

### 2. Run Full 7-Stage Training, Purged CV & Hold-Out Backtest Pipeline

```bash
python run_all.py
```

Outputs trained ensemble artifacts to `runs/models/spx_ensemble_forecaster.joblib` and real validation/backtest JSON reports to `runs/latest_validation_report.json` and `runs/latest_backtest.json`.

### 3. Launch Web Trading Workbench

```bash
python src/web/app.py
```

Open `http://127.0.0.1:8080` in your browser. All frontend charts load offline from `src/web/static/vendor/`.

### REST Telemetry & Action Endpoints

- `GET /api/telemetry` — Live/cached `SPX` & `SPY` quotes, candles, GEX strikes, trade plans, Conformal 90% interval, `agent_reasoning` (all 8 steps), TreeSHAP feature contributions, news, and ledger metrics.
- `GET /api/backtest` — Hold-out equity vs benchmark curve, drawdown curve, calibration bins, 3x3 confusion matrix, baseline comparisons, and Sharpe 95% CI.
- `GET /api/validation` — Detailed Purged Walk-Forward CV and untouched hold-out evaluation JSON.
- `GET /api/gex` — Delayed Cboe strike-level Net/Call/Put GEX profile and zero-gamma pivot.
- `GET /api/committee` & `GET /api/agents` — 8-agent committee consensus trace, inputs, and basis rules.
- `GET /api/ledger` & `GET /api/positions` — Append-only forward paper ledger metrics (`runs/paper_ledger.jsonl`).
- `POST /api/simulate` — Re-run empirical OHLC bootstrap fan for `SPX` or `SPY`.
- `POST /api/ledger/record` & `POST /api/ledger/clear` — Record actionable signal or clear forward paper ledger.

---

## Testing

Run the automated test suite (`12` tests covering config/calendar, GEX formulas, FOMC calendar proximity, target label lookahead, `purge_bars=6` zero-overlap CV, TreeSHAP ensemble & neutral prior, 8-step agent reasoning, LONG/SHORT bootstrap without RNG mutation, strict no-leakage backtester, empty-start stop-first ledger, offline CDN check, and telemetry contracts):

```bash
pytest -v
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Chapter 7: Purged & Embargoed Cross-Validation).
- Baltussen, G., Da, Z., Lammers, S., & Martens, M. (2021). *Hedging Demand and Market Intraday Momentum*. Journal of Financial Economics. (Intraday dealer gamma hedging & closing momentum).
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. (Split/OOF Non-parametric Conformal Prediction bounds).

---

## License

MIT License — Author: Doshab Hussain.

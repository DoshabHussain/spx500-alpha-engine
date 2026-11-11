(() => {
  "use strict";

  const state = {
    instrument: "spx",
    theme: "dark",
    telemetry: null,
    backtest: null,
    charts: {
      tvChart: null,
      candleSeries: null,
      vwapSeries: null,
      volumeSeries: null,
      priceLines: [],
      fan: null,
      shap: null,
      equity: null,
      drawdown: null,
      calibration: null,
      gex: null,
    },
  };

  function el(id) {
    return document.getElementById(id);
  }

  function escHtml(val) {
    return String(val ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function fmtNum(v, decimals = 2) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "--";
    return n.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function fmtSigned(v, decimals = 2, suffix = "") {
    const n = Number(v);
    if (!Number.isFinite(n)) return "--";
    const sign = n > 0 ? "+" : "";
    return `${sign}${fmtNum(n, decimals)}${suffix}`;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function setErrorBanner(msg) {
    const banner = el("globalErrorBanner");
    if (!msg) {
      banner.textContent = "";
      banner.classList.remove("visible");
      return;
    }
    banner.textContent = msg;
    banner.classList.add("visible");
  }

  function updateProvenanceBadges(prov, ageSec, detail) {
    const label = prov || "CACHED_REAL";
    const text = ageSec !== undefined && ageSec !== null ? `${label} · ${ageSec}s` : label;

    const globalBadge = el("globalProvenanceBadge");
    if (globalBadge) {
      globalBadge.textContent = text;
      globalBadge.className = `badge-prov prov-${label}`;
    }
    document.querySelectorAll("[data-prov-slot]").forEach((node) => {
      node.textContent = label;
      node.className = `badge-prov prov-${label}`;
    });
    if (detail) {
      el("stripProvenanceDetail").textContent = detail;
    }
  }

  function votePillClass(vote) {
    const upper = String(vote || "").toUpperCase();
    if (upper.includes("BULL") || upper.includes("LONG") || upper.includes("APPROV") || upper.includes("CLEAR")) {
      return "pill pill-bull";
    }
    if (upper.includes("BEAR") || upper.includes("SHORT") || upper.includes("VETO") || upper.includes("HALT") || upper.includes("REJECT")) {
      return "pill pill-bear";
    }
    if (upper.includes("CAUTION") || upper.includes("WARN") || upper.includes("MIXED")) {
      return "pill pill-warn";
    }
    return "pill pill-neutral";
  }

  // ---- Tab & Theme Controls ----
  function initControls() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab");
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        const panel = el(`tab-${target}`);
        if (panel) panel.classList.add("active");
        if (target === "terminal" && state.charts.tvChart) {
          const box = el("tvChartContainer");
          if (box) state.charts.tvChart.resize(box.clientWidth, box.clientHeight);
        }
      });
    });

    el("btnInstrumentSpx").addEventListener("click", () => switchInstrument("spx"));
    el("btnInstrumentSpy").addEventListener("click", () => switchInstrument("spy"));

    el("btnThemeToggle").addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", state.theme);
      el("btnThemeToggle").textContent = state.theme === "dark" ? "Light Theme" : "Dark Theme";
      renderAll();
    });

    el("btnRefreshAll").addEventListener("click", loadAllData);
    el("btnResimulate").addEventListener("click", runBootstrapSimulation);
    el("btnRecordSignal").addEventListener("click", recordCurrentSignal);
    el("btnLedgerRecord").addEventListener("click", recordCurrentSignal);
    el("btnLedgerClear").addEventListener("click", clearForwardLedger);
  }

  function switchInstrument(inst) {
    state.instrument = inst;
    el("btnInstrumentSpx").classList.toggle("active", inst === "spx");
    el("btnInstrumentSpy").classList.toggle("active", inst === "spy");
    renderTelemetry();
  }

  // ---- Data Fetching ----
  async function fetchJson(url, options = {}) {
    const resp = await fetch(url, options);
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status} from ${url}: ${txt.slice(0, 180)}`);
    }
    return resp.json();
  }

  async function loadAllData() {
    setErrorBanner("");
    try {
      const [telemetry, backtest] = await Promise.all([
        fetchJson("/api/telemetry"),
        fetchJson("/api/backtest"),
      ]);
      state.telemetry = telemetry;
      state.backtest = backtest;
      renderAll();
    } catch (err) {
      setErrorBanner(`Telemetry load failed: ${err.message}`);
    }
  }

  async function runBootstrapSimulation() {
    if (!state.telemetry) return;
    const msgBox = el("recordStatusMsg");
    msgBox.textContent = "Running empirical OHLC return bootstrap...";
    try {
      const inst = state.instrument;
      const instData = state.telemetry[inst];
      const plan = instData.trade_plan || {};
      const payload = {
        instrument: inst.toUpperCase(),
        direction: plan.signal === "SHORT" ? "SHORT" : "LONG",
        paths: 1500,
      };
      const sim = await fetchJson("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.telemetry.timemachine_simulation = sim;
      if (sim.fan_series) {
        state.telemetry[inst].fan_series = sim.fan_series;
      }
      renderFanChart();
      msgBox.textContent = `Re-simulated ${sim.paths_simulated} bootstrap paths (${sim.bootstrap_mode}): Win=${fmtNum(sim.win_rate, 1)}%, Exp=${fmtSigned(sim.expected_r, 3, "R")}`;
    } catch (err) {
      msgBox.textContent = `Simulation error: ${err.message}`;
    }
  }

  async function recordCurrentSignal() {
    const msgBox = el("recordStatusMsg");
    try {
      const res = await fetchJson("/api/ledger/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instrument: state.instrument.toUpperCase() }),
      });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      if (res.recorded) {
        msgBox.textContent = `Recorded signal ${res.trade.trade_id} (${res.trade.direction} ${res.trade.instrument} @ ${fmtNum(res.trade.entry_price)})`;
      } else {
        msgBox.textContent = res.reason || "Signal was not recorded (NO_TRADE mandate).";
      }
    } catch (err) {
      msgBox.textContent = `Ledger error: ${err.message}`;
    }
  }

  async function clearForwardLedger() {
    const msgBox = el("recordStatusMsg");
    try {
      const res = await fetchJson("/api/ledger/clear", { method: "POST" });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      msgBox.textContent = "Forward paper ledger cleared.";
    } catch (err) {
      msgBox.textContent = `Clear error: ${err.message}`;
    }
  }

  // ---- Top-Level Render ----
  function renderAll() {
    renderTelemetry();
    renderBacktestEvidence();
  }

  function renderTelemetry() {
    const t = state.telemetry;
    if (!t) return;
    const instKey = state.instrument;
    const instData = t[instKey];
    if (!instData) return;

    updateProvenanceBadges(
      t.provenance,
      t.server_cache_age_sec,
      `${t.provenance_detail || ""} | Fetched UTC: ${t.fetched_at_utc || "--"}`
    );

    // Macro strip
    const chgColor = instData.change_pts >= 0 ? "var(--bull)" : "var(--bear)";
    el("stripSpot").textContent = `${instKey.toUpperCase()} ${fmtNum(instData.spot)}`;
    el("stripChange").textContent = `${fmtSigned(instData.change_pts)} (${fmtSigned(instData.change_pct, 2, "%")})`;
    el("stripChange").style.color = chgColor;

    const macro = t.macro || {};
    el("stripVix").textContent = fmtNum(macro.vix);
    el("stripUs10y").textContent = `${fmtNum(macro.us10y, 3)}%`;
    el("stripDxy").textContent = fmtNum(macro.dxy, 3);
    const gex = instData.gex || {};
    el("stripGex").textContent = `${fmtSigned(gex.net_gex_total_b, 3, "B")}`;

    renderHeroTradeCard(instKey, instData, t);
    renderTradingViewChart(instKey, instData);
    renderFanChart();
    renderShapChart();
    renderAgentPipeline(t.agent_reasoning || []);
    renderDebateLog(t.committee_debate_log || []);
    renderMarketAndGex(instKey, instData, t);
    renderLedger();
  }

  function renderHeroTradeCard(instKey, instData, t) {
    const plan = instData.trade_plan || {};
    const conf = instData.conformal_90 || {};
    const sig = plan.signal || "NO_TRADE";

    const banner = el("tradeCardBanner");
    banner.className = `trade-card-banner ${sig}`;
    el("tradeSignalText").textContent = `${instKey.toUpperCase()} · ${sig}`;
    el("tradeScoreText").textContent = `${fmtNum(t.consensus_score, 2)} / 10`;

    el("lvlEntry").textContent = fmtNum(plan.entry);
    el("lvlStop").textContent = fmtNum(plan.stop_loss);
    el("lvlTp1").textContent = fmtNum(plan.tp1);
    el("lvlTp2").textContent = fmtNum(plan.tp2);
    el("lvlRr").textContent = `${fmtNum(plan.reward_to_risk, 2)}R · Exp ${fmtSigned(plan.expected_r, 3, "R")}`;
    el("lvlConformal").textContent = `[${fmtNum(conf.lower_bound)} – ${fmtNum(conf.upper_bound)}]`;

    const probs = t.probabilities || {};
    const pUp = Number(probs.p_up || 0) * 100;
    const pFlat = Number(probs.p_flat || 0) * 100;
    const pDown = Number(probs.p_down || 0) * 100;
    el("barProbUp").style.width = `${pUp.toFixed(1)}%`;
    el("barProbFlat").style.width = `${pFlat.toFixed(1)}%`;
    el("barProbDown").style.width = `${pDown.toFixed(1)}%`;
    el("txtProbUp").textContent = `${fmtNum(pUp, 1)}%`;
    el("txtProbFlat").textContent = `${fmtNum(pFlat, 1)}%`;
    el("txtProbDown").textContent = `${fmtNum(pDown, 1)}%`;

    el("tradePlanReason").textContent =
      `${plan.reason || ""} | ATR(14)=${fmtNum(plan.atr_14, 2)} pts | Conformal q90=${fmtNum(conf.calibrated_q90_pct, 3)}% | Cutoff=${plan.mandatory_exit || "15:45 ET"}`;

    el("chartTitle").textContent = `${instKey.toUpperCase()} 5-Minute Session & Execution Levels`;
  }

  // ---- TradingView Lightweight Charts ----
  function renderTradingViewChart(instKey, instData) {
    const container = el("tvChartContainer");
    if (!container || typeof window.LightweightCharts === "undefined") return;

    const bg = cssVar("--bg-surface");
    const text = cssVar("--text-secondary");
    const grid = cssVar("--border-color");
    const bull = cssVar("--bull");
    const bear = cssVar("--bear");
    const accent = cssVar("--accent-primary");

    if (state.charts.tvChart) {
      state.charts.tvChart.remove();
      state.charts.tvChart = null;
      state.charts.priceLines = [];
    }

    const chart = window.LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { color: bg },
        textColor: text,
      },
      grid: {
        vertLines: { color: grid },
        horzLines: { color: grid },
      },
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: grid,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: { borderColor: grid },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: bull,
      downColor: bear,
      borderVisible: false,
      wickUpColor: bull,
      wickDownColor: bear,
    });

    const vwapSeries = chart.addLineSeries({
      color: accent,
      lineWidth: 2,
      title: "VWAP",
    });

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      color: "rgba(100, 116, 139, 0.35)",
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    const rawCandles = Array.isArray(instData.candles) ? instData.candles : [];
    const seenTimes = new Set();
    const cleanCandles = [];
    for (const c of rawCandles) {
      const ts = Number(c.time);
      if (!Number.isFinite(ts) || seenTimes.has(ts)) continue;
      seenTimes.add(ts);
      cleanCandles.push(c);
    }
    cleanCandles.sort((a, b) => a.time - b.time);

    candleSeries.setData(
      cleanCandles.map((c) => ({
        time: c.time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );

    vwapSeries.setData(
      cleanCandles
        .filter((c) => Number.isFinite(Number(c.vwap)))
        .map((c) => ({ time: c.time, value: Number(c.vwap) }))
    );

    volumeSeries.setData(
      cleanCandles.map((c) => ({
        time: c.time,
        value: Number(c.volume || 0),
        color: c.close >= c.open ? "rgba(16, 185, 129, 0.28)" : "rgba(239, 68, 68, 0.28)",
      }))
    );

    const plan = instData.trade_plan || {};
    const levels = [
      { price: plan.entry, color: accent, title: `ENTRY ${instKey.toUpperCase()}`, style: 0 },
      { price: plan.stop_loss, color: bear, title: "STOP LOSS", style: 2 },
      { price: plan.tp1, color: bull, title: "TP1", style: 2 },
      { price: plan.tp2, color: bull, title: "TP2", style: 1 },
    ];
    for (const lv of levels) {
      if (Number.isFinite(Number(lv.price))) {
        candleSeries.createPriceLine({
          price: Number(lv.price),
          color: lv.color,
          lineWidth: 1,
          lineStyle: lv.style,
          axisLabelVisible: true,
          title: lv.title,
        });
      }
    }

    chart.timeScale().fitContent();
    state.charts.tvChart = chart;
    state.charts.candleSeries = candleSeries;
    state.charts.vwapSeries = vwapSeries;
    state.charts.volumeSeries = volumeSeries;
  }

  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      state.charts[key] = null;
    }
  }

  function chartColors() {
    return {
      text: cssVar("--text-secondary"),
      grid: cssVar("--border-color"),
      bull: cssVar("--bull"),
      bear: cssVar("--bear"),
      accent: cssVar("--accent-primary"),
      warn: cssVar("--warn"),
      neutral: cssVar("--neutral"),
    };
  }

  // ---- TimeMachine Bootstrap Fan ----
  function renderFanChart() {
    const canvas = el("fanChartCanvas");
    if (!canvas || typeof window.Chart === "undefined" || !state.telemetry) return;

    const instKey = state.instrument;
    const instData = state.telemetry[instKey] || {};
    const fan = instData.fan_series || {};
    const tm = state.telemetry.timemachine_simulation || {};

    el("tmSummarySubtitle").textContent =
      `${tm.bootstrap_mode || "EMPIRICAL_BAR_BOOTSTRAP"} · Paths=${tm.paths_simulated || "--"} · Win=${fmtNum(tm.win_rate, 1)}% · Exp=${fmtSigned(tm.expected_r, 3, "R")} (${tm.verdict || "--"})`;

    destroyChart("fan");
    const c = chartColors();
    const steps = (fan.steps || []).map((s) => `+${s * 5}m`);

    state.charts.fan = new window.Chart(canvas, {
      type: "line",
      data: {
        labels: steps,
        datasets: [
          {
            label: "P95",
            data: fan.p95 || [],
            borderColor: "rgba(16, 185, 129, 0.5)",
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          },
          {
            label: "P75",
            data: fan.p75 || [],
            borderColor: c.bull,
            backgroundColor: "rgba(16, 185, 129, 0.14)",
            borderWidth: 1.5,
            pointRadius: 0,
            fill: "+1",
          },
          {
            label: "P50 (Median)",
            data: fan.p50 || [],
            borderColor: c.accent,
            borderWidth: 2.5,
            pointRadius: 2,
            fill: false,
          },
          {
            label: "P25",
            data: fan.p25 || [],
            borderColor: c.warn,
            backgroundColor: "rgba(239, 68, 68, 0.10)",
            borderWidth: 1.5,
            pointRadius: 0,
            fill: "-1",
          },
          {
            label: "P5",
            data: fan.p5 || [],
            borderColor: "rgba(239, 68, 68, 0.65)",
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: c.text, boxWidth: 12, font: { size: 11 } } },
        },
        scales: {
          x: { ticks: { color: c.text }, grid: { color: c.grid } },
          y: { ticks: { color: c.text }, grid: { color: c.grid } },
        },
      },
    });
  }

  // ---- TreeSHAP Contributions ----
  function renderShapChart() {
    const canvas = el("shapChartCanvas");
    if (!canvas || typeof window.Chart === "undefined" || !state.telemetry) return;

    const contribs = (state.telemetry.feature_contributions || []).slice(0, 8);
    destroyChart("shap");
    const c = chartColors();

    state.charts.shap = new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: contribs.map((x) => x.feature),
        datasets: [
          {
            label: "TreeSHAP Log-Odds Contribution (UP vs DOWN)",
            data: contribs.map((x) => Number(x.shap_contribution || 0)),
            backgroundColor: contribs.map((x) => (Number(x.shap_contribution) >= 0 ? c.bull : c.bear)),
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: c.text, font: { size: 11 } } },
        },
        scales: {
          x: { ticks: { color: c.text }, grid: { color: c.grid } },
          y: { ticks: { color: c.text, font: { size: 11 } }, grid: { color: c.grid } },
        },
      },
    });
  }

  // ---- 8-Agent Planning Pipeline ----
  function renderAgentPipeline(agents) {
    const grid = el("agentFlowGrid");
    if (!grid) return;

    if (!agents.length) {
      grid.innerHTML = `<div class="empty-state">No agent reasoning trace returned from telemetry.</div>`;
      return;
    }

    grid.innerHTML = agents
      .map((ag) => {
        const inputsObj = ag.inputs || {};
        const inputRows = Object.entries(inputsObj)
          .map(
            ([k, v]) => `
            <div class="agent-input-row">
              <span class="agent-input-key">${escHtml(k)}</span>
              <span class="agent-input-val">${escHtml(v)}</span>
            </div>`
          )
          .join("");

        return `
        <article class="agent-card">
          <div class="agent-card-top">
            <div>
              <div class="agent-step">STEP ${escHtml(ag.step)} · ${escHtml(ag.weight_pct ?? 0)}% WEIGHT</div>
              <h3 class="agent-name">${escHtml(ag.agent)}</h3>
              <div class="agent-role">${escHtml(ag.role)}</div>
            </div>
            <span class="${votePillClass(ag.vote)}">${escHtml(ag.vote)}</span>
          </div>
          <div class="agent-inputs">${inputRows}</div>
          <div class="agent-basis"><strong>Basis Rule:</strong> ${escHtml(ag.basis_rule)}</div>
          <div class="agent-output">
            <div><strong>Output:</strong> ${escHtml(ag.output_summary)}</div>
          </div>
        </article>`;
      })
      .join("");
  }

  function renderDebateLog(entries) {
    const list = el("debateLogList");
    if (!list) return;
    if (!entries.length) {
      list.innerHTML = `<div class="empty-state">No committee log entries available.</div>`;
      return;
    }
    list.innerHTML = entries
      .map(
        (e) => `
      <div class="debate-entry">
        <span class="mono" style="color: var(--text-muted); margin-right: 8px;">[${escHtml(e.time)}]</span>
        <span class="pill pill-neutral" style="margin-right: 8px;">${escHtml(e.badge || "LOG")}</span>
        <strong style="margin-right: 6px;">${escHtml(e.agent)}:</strong>
        <span>${escHtml(e.message)}</span>
      </div>`
      )
      .join("");
  }

  // ---- TAB 2: MODEL EVIDENCE ----
  function renderBacktestEvidence() {
    const bt = state.backtest;
    if (!bt || !bt.available) return;

    const strat = bt.strategy || {};
    const val = bt.validation_summary || {};
    const cv = val.purged_cv || {};
    const ho = val.holdout_evaluation || {};
    const cal = ho.calibration || cv.calibration || {};

    document.querySelectorAll("[data-ev-prov]").forEach((node) => {
      const p = bt.provenance || "CACHED_REAL";
      node.textContent = p;
      node.className = `badge-prov prov-${p}`;
    });

    el("evReturn").textContent = fmtSigned(strat.total_return_pct, 2, "%");
    el("evReturn").style.color = strat.total_return_pct >= 0 ? "var(--bull)" : "var(--bear)";
    el("evBenchmarkReturn").textContent = `Buy & Hold: ${fmtSigned(strat.benchmark_return_pct, 2, "%")} (${strat.bars_evaluated || 0} bars)`;

    const ci = strat.sharpe_ci_95 || [0, 0];
    el("evSharpe").textContent = `${fmtNum(strat.sharpe_ratio, 2)}`;
    el("evSharpeCi").textContent = `95% CI [${fmtNum(ci[0], 2)}, ${fmtNum(ci[1], 2)}] · Sortino ${fmtNum(strat.sortino_ratio, 2)}`;

    el("evDrawdown").textContent = `${fmtNum(strat.max_drawdown_pct, 2)}%`;
    el("evPf").textContent = `PF ${fmtNum(strat.profit_factor, 2)} · Win ${fmtNum(strat.win_rate_pct, 1)}% (${strat.total_trades || 0} trades)`;

    const cvAccPct = Number(cv.mean_cv_accuracy || 0) * 100;
    const ov = cv.verified_zero_overlap ? "ZERO OVERLAP" : "OVERLAP DETECTED";
    el("evCvAcc").textContent = `${fmtNum(cvAccPct, 1)}%`;
    el("evLeakAudit").textContent = `${ov} · Purge=${cv.purge_bars ?? "--"} bars`;

    el("evHoldoutWindow").textContent =
      `Chronological 30% Hold-Out: ${ho.start_timestamp || "--"} to ${ho.end_timestamp || "--"} (${strat.slippage_bps || 5.0} bps slippage)`;

    renderEquityAndDrawdown(strat);
    renderCalibrationAndConfusion(ho, cal);
    renderBaselinesTable(cv, ho, strat);
  }

  function renderEquityAndDrawdown(strat) {
    if (typeof window.Chart === "undefined") return;
    const c = chartColors();
    const labels = (strat.labels || []).map((s) => String(s).slice(5, 16));

    destroyChart("equity");
    const eqCanvas = el("equityChartCanvas");
    if (eqCanvas) {
      state.charts.equity = new window.Chart(eqCanvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Strategy Equity ($)",
              data: strat.equity_curve || [],
              borderColor: c.bull,
              borderWidth: 2,
              pointRadius: 0,
            },
            {
              label: "SPY Buy & Hold ($)",
              data: strat.benchmark_curve || [],
              borderColor: c.text,
              borderDash: [5, 4],
              borderWidth: 1.5,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: c.text } } },
          scales: {
            x: { ticks: { color: c.text, maxTicksLimit: 8 }, grid: { color: c.grid } },
            y: { ticks: { color: c.text }, grid: { color: c.grid } },
          },
        },
      });
    }

    destroyChart("drawdown");
    const ddCanvas = el("drawdownChartCanvas");
    if (ddCanvas) {
      state.charts.drawdown = new window.Chart(ddCanvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Drawdown (%)",
              data: strat.drawdown_curve || [],
              borderColor: c.bear,
              backgroundColor: "rgba(239, 68, 68, 0.16)",
              fill: true,
              borderWidth: 1.5,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: c.text } } },
          scales: {
            x: { ticks: { color: c.text, maxTicksLimit: 8 }, grid: { color: c.grid } },
            y: { ticks: { color: c.text }, grid: { color: c.grid } },
          },
        },
      });
    }
  }

  function renderCalibrationAndConfusion(ho, cal) {
    el("evBrierSubtitle").textContent =
      `Multiclass Brier Score: ${fmtNum(cal.brier_score, 4)} · ECE: ${fmtNum(cal.ece, 4)} · Accuracy: ${fmtNum((ho.accuracy || 0) * 100, 1)}%`;

    const calCanvas = el("calibrationChartCanvas");
    if (calCanvas && typeof window.Chart !== "undefined") {
      destroyChart("calibration");
      const c = chartColors();
      const bins = cal.bins || [];
      state.charts.calibration = new window.Chart(calCanvas, {
        type: "line",
        data: {
          labels: bins.map((b) => fmtNum(b.mean_confidence, 2)),
          datasets: [
            {
              label: "Empirical Accuracy",
              data: bins.map((b) => b.empirical_accuracy),
              borderColor: c.accent,
              backgroundColor: c.accent,
              borderWidth: 2,
              pointRadius: 4,
            },
            {
              label: "Perfect Calibration (y=x)",
              data: bins.map((b) => b.mean_confidence),
              borderColor: c.text,
              borderDash: [4, 4],
              borderWidth: 1.5,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: c.text } } },
          scales: {
            x: { title: { display: true, text: "Predicted Confidence", color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
            y: { min: 0, max: 1, title: { display: true, text: "Observed Accuracy", color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
          },
        },
      });
    }

    const cmWrap = el("confusionMatrixWrap");
    const cm = ho.confusion_matrix || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const names = ["Down (0)", "Flat (1)", "Up (2)"];
    let html = `<div></div>` + names.map((n) => `<div class="cm-head">Pred ${n}</div>`).join("");
    for (let r = 0; r < 3; r++) {
      html += `<div class="cm-head" style="text-align:right;padding-right:6px;">True ${names[r]}</div>`;
      for (let col = 0; col < 3; col++) {
        const cls = r === col ? "cm-cell diag" : "cm-cell";
        html += `<div class="${cls}">${Number((cm[r] && cm[r][col]) || 0)}</div>`;
      }
    }
    cmWrap.innerHTML = html;
  }

  function renderBaselinesTable(cv, ho, strat) {
    const tbody = el("baselineTableBody");
    const base = cv.baselines || {};
    const cvCi = cv.accuracy_ci_95 || [0, 0];
    const rows = [
      [
        "Purged Walk-Forward Ensemble",
        `${fmtNum((cv.mean_cv_accuracy || 0) * 100, 2)}% (95% CI [${fmtNum(cvCi[0] * 100, 1)}%, ${fmtNum(cvCi[1] * 100, 1)}%])`,
        `${fmtNum((ho.accuracy || 0) * 100, 2)}%`,
        `Folds=${(cv.cv_accuracies || []).length}, Purge=${cv.purge_bars ?? 6} bars, Leakage=${cv.leakage_bias || "VERIFIED_ZERO_OVERLAP"}`,
      ],
      [
        "Majority Class Baseline (Always Flat)",
        `${fmtNum((base.always_flat_accuracy || 0) * 100, 2)}%`,
        `${fmtNum((ho.always_flat_accuracy || 0) * 100, 2)}%`,
        `Binomial p-value vs Always-Flat (CV): ${fmtNum(base.binomial_p_vs_majority, 4)}`,
      ],
      [
        "Rolling Momentum Baseline (5-bar)",
        `${fmtNum((base.momentum_baseline_accuracy || 0) * 100, 2)}%`,
        `${fmtNum((ho.momentum_baseline_accuracy || 0) * 100, 2)}%`,
        `Same-direction sign rule on trailing 5-bar return`,
      ],
      [
        "Transaction Costs & Execution Assumptions",
        "--",
        `${fmtSigned(strat.total_return_pct, 2, "%")} vs B&H ${fmtSigned(strat.benchmark_return_pct, 2, "%")}`,
        `Slippage=${strat.slippage_bps || 5.0} bps/side, Commission=$0.005/sh, Signal Threshold>=0.38`,
      ],
    ];
    tbody.innerHTML = rows
      .map(
        (r) =>
          `<tr><td><strong>${escHtml(r[0])}</strong></td><td class="mono">${escHtml(r[1])}</td><td class="mono">${escHtml(r[2])}</td><td>${escHtml(r[3])}</td></tr>`
      )
      .join("");
  }

  // ---- TAB 3: MARKET & GEX ----
  function renderMarketAndGex(instKey, instData, t) {
    const gex = instData.gex || {};
    const micro = t.microstructure || {};
    const macro = t.intermarket_macro || {};
    const critic = t.critic_audit || {};

    el("mkNetGex").textContent = `${fmtSigned(gex.net_gex_total_b, 3, "B")}`;
    el("mkRegime").textContent = `Regime: ${gex.gamma_regime || "--"}`;
    el("mkPivot").textContent = fmtNum(gex.zero_gex_pivot, 2);
    el("mkSource").textContent = `Source: ${gex.source || "--"} (${gex.contracts_parsed || 0} contracts)`;
    el("mkWalls").textContent = `C ${fmtNum(gex.call_wall, 0)} / P ${fmtNum(gex.put_wall, 0)}`;
    el("mkExpiry").textContent = `Nearest Expiry: ${gex.nearest_expiry || "--"}`;
    el("mkPcRatio").textContent = `${fmtNum(gex.pc_ratio_0dte_vol, 2)} Vol · ${fmtNum(gex.pc_ratio_oi, 2)} OI`;
    el("mkVanna").textContent = `Vanna: ${fmtSigned(micro.vanna_imbalance_b, 2, "B")} · ${micro.charm_phase || "--"}`;

    el("gexChartTitle").textContent = `${instKey.toUpperCase()} Dealer Gamma Exposure by Strike ($M)`;

    const gexCanvas = el("gexChartCanvas");
    if (gexCanvas && typeof window.Chart !== "undefined") {
      destroyChart("gex");
      const c = chartColors();
      const strikes = gex.strikes || [];
      state.charts.gex = new window.Chart(gexCanvas, {
        type: "bar",
        data: {
          labels: strikes.map((s) => String(s.strike)),
          datasets: [
            {
              label: "Call GEX ($M)",
              data: strikes.map((s) => s.call_gex),
              backgroundColor: c.bull,
            },
            {
              label: "Put GEX ($M)",
              data: strikes.map((s) => s.put_gex),
              backgroundColor: c.bear,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: c.text } } },
          scales: {
            x: { ticks: { color: c.text }, grid: { color: c.grid } },
            y: { ticks: { color: c.text }, grid: { color: c.grid } },
          },
        },
      });
    }

    el("intermarketDetails").innerHTML = `
      <div class="agent-inputs" style="gap: 8px; padding: 12px; font-size: 13px;">
        <div class="agent-input-row"><span class="agent-input-key">Intermarket Macro Bias</span><span class="agent-input-val">${escHtml(macro.macro_bias || "--")} (${fmtSigned(macro.macro_score, 2)})</span></div>
        <div class="agent-input-row"><span class="agent-input-key">CBOE VIX Spot</span><span class="agent-input-val">${fmtNum(macro.vix_spot, 2)}</span></div>
        <div class="agent-input-row"><span class="agent-input-key">US 10Y Treasury Yield</span><span class="agent-input-val">${fmtNum(macro.us10y_yield, 3)}%</span></div>
        <div class="agent-input-row"><span class="agent-input-key">US Dollar Index (DXY)</span><span class="agent-input-val">${fmtNum(macro.dxy_index, 3)}</span></div>
        <div class="agent-input-row"><span class="agent-input-key">VIX Term Slope</span><span class="agent-input-val">${escHtml(macro.vix_term_slope || "--")}</span></div>
        <div class="agent-input-row"><span class="agent-input-key">Adversarial Critic Verdict</span><span class="agent-input-val">${escHtml(critic.verdict || "--")} (${escHtml(critic.skepticism_level || "--")})</span></div>
        <div class="agent-input-row"><span class="agent-input-key">Critic Recommended Action</span><span class="agent-input-val">${escHtml(critic.recommended_action || "--")}</span></div>
      </div>
      <div class="mt-16" style="font-size: 12px; color: var(--text-secondary);">
        <strong>Catalysts:</strong> ${escHtml((macro.catalysts || []).join(" · ") || "None")}
        <br/><strong>Critic Objections:</strong> ${escHtml((critic.objections || []).join(" · ") || "No structural objections")}
      </div>`;

    const news = t.news || {};
    const newsBadge = el("newsProvBadge");
    const nProv = news.provenance || "DEMO";
    newsBadge.textContent = nProv;
    newsBadge.className = `badge-prov prov-${nProv}`;

    const items = news.items || [];
    el("newsFeedList").innerHTML = items.length
      ? items
          .map(
            (n) => `
          <div class="news-item">
            <div class="news-title">${escHtml(n.title)}</div>
            <div class="news-meta">
              <span class="${votePillClass(n.sentiment)}">${escHtml(n.sentiment)}</span>
              <span>${escHtml(n.source || "RSS")}</span>
              <span>${escHtml(n.published || "")}</span>
              <span>Tags: ${escHtml((n.tags || []).join(", "))}</span>
            </div>
          </div>`
          )
          .join("")
      : `<div class="empty-state">No financial news headlines loaded.</div>`;
  }

  // ---- TAB 4: FORWARD LEDGER ----
  function renderLedger() {
    const ledger = (state.telemetry && state.telemetry.ledger) || {};
    el("lgTotal").textContent = String(ledger.total_signals ?? 0);
    el("lgOpenResolved").textContent = `Open: ${ledger.open_signals ?? 0} · Resolved: ${ledger.resolved_signals ?? 0}`;
    el("lgWinRate").textContent = `${fmtNum(ledger.win_rate_pct || 0, 1)}%`;
    el("lgPf").textContent = `Profit Factor: ${fmtNum(ledger.profit_factor || 0, 2)}`;
    el("lgExpR").textContent = `${fmtSigned(ledger.expectancy_r || 0, 2, "R")}`;
    el("lgEquity").textContent = `$${fmtNum(ledger.account_equity ?? 100000, 2)}`;
    el("lgPnl").textContent = `Realized P&L: $${fmtSigned(ledger.total_realized_pnl || 0, 2)}`;

    const trades = ledger.recent_trades || [];
    const wrap = el("ledgerTableContainer");
    if (!trades.length) {
      wrap.innerHTML = `<div class="empty-state">Forward paper ledger is empty. No fabricated history is displayed. Click <strong>Record Current Signal</strong> when an actionable trade plan is generated.</div>`;
      return;
    }
    wrap.innerHTML = `
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Signal ID</th>
              <th>Timestamp (ET)</th>
              <th>Inst</th>
              <th>Dir</th>
              <th>Entry</th>
              <th>Stop</th>
              <th>TP1</th>
              <th>Status</th>
              <th>Exit</th>
              <th>R</th>
              <th>P&amp;L ($)</th>
            </tr>
          </thead>
          <tbody>
            ${trades
              .map(
                (tr) => `
              <tr>
                <td class="mono">${escHtml(tr.trade_id)}</td>
                <td class="mono">${escHtml(`${tr.date || ""} ${tr.time || ""}`.trim())}</td>
                <td>${escHtml(tr.instrument)}</td>
                <td><span class="${votePillClass(tr.direction)}">${escHtml(tr.direction)}</span></td>
                <td class="mono">${fmtNum(tr.entry_price)}</td>
                <td class="mono">${fmtNum(tr.stop)}</td>
                <td class="mono">${fmtNum(tr.target)}</td>
                <td class="mono">${escHtml(tr.outcome)}</td>
                <td class="mono">${tr.exit_price != null ? fmtNum(tr.exit_price) : "--"}</td>
                <td class="mono">${fmtSigned(tr.return_r || 0, 2, "R")}</td>
                <td class="mono">${fmtSigned(tr.pnl_usd || 0, 2)}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>`;
  }

  window.addEventListener("resize", () => {
    if (state.charts.tvChart) {
      const box = el("tvChartContainer");
      if (box) state.charts.tvChart.resize(box.clientWidth, box.clientHeight);
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    initControls();
    loadAllData();
  });
})();

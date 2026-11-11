(() => {
  "use strict";

  const SESSION_TEL_KEY = "spx500_tel_cache_v2";
  const SESSION_BT_KEY = "spx500_bt_cache_v2";
  const SESSION_ADMIN_KEY = "spx500_admin_key";

  const state = {
    instrument: localStorage.getItem("spx500_inst") === "spy" ? "spy" : "spx",
    theme: localStorage.getItem("spx500_theme") === "light" ? "light" : "dark",
    autoRefreshMs: 0,
    autoTimer: null,
    telemetry: null,
    backtest: null,
    security: { admin_key_required: false, authorized: true },
    lastAgentFingerprint: "",
    lastBacktestFingerprint: "",
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

  function chartColors() {
    return {
      text: cssVar("--text-secondary"),
      grid: cssVar("--border-color"),
      bull: cssVar("--bull"),
      bear: cssVar("--bear"),
      accent: cssVar("--accent-primary"),
      ivory: cssVar("--accent-ivory"),
      vwap: cssVar("--chart-vwap"),
      warn: cssVar("--warn"),
      neutral: cssVar("--neutral"),
      bg: cssVar("--bg-surface-solid"),
    };
  }

  function showToast(message, tone = "info") {
    const stack = el("toastStack");
    if (!stack) return;
    const item = document.createElement("div");
    item.className = `toast ${tone}`;
    item.textContent = message;
    stack.appendChild(item);
    window.setTimeout(() => {
      item.style.opacity = "0";
      item.style.transform = "translate3d(0, 6px, 0)";
      item.style.transition = "opacity 0.2s ease, transform 0.2s ease";
      window.setTimeout(() => item.remove(), 220);
    }, 3200);
  }

  function flashElement(node) {
    if (!node) return;
    node.classList.remove("flash-tick");
    void node.offsetWidth;
    node.classList.add("flash-tick");
  }

  function setErrorBanner(msg) {
    const banner = el("globalErrorBanner");
    if (!banner) return;
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
    if (upper.includes("BULL") || upper.includes("LONG") || upper.includes("APPROV") || upper.includes("CLEAR") || upper.includes("TARGET")) {
      return "pill pill-bull";
    }
    if (upper.includes("BEAR") || upper.includes("SHORT") || upper.includes("VETO") || upper.includes("HALT") || upper.includes("REJECT") || upper.includes("STOP")) {
      return "pill pill-bear";
    }
    if (upper.includes("CAUTION") || upper.includes("WARN") || upper.includes("MIXED") || upper.includes("MANUAL")) {
      return "pill pill-warn";
    }
    return "pill pill-neutral";
  }

  // ---- Controls & Cursor Spotlight ----
  function initControls() {
    document.documentElement.setAttribute("data-theme", state.theme);
    el("btnThemeToggle").textContent = state.theme === "dark" ? "Light Theme" : "Dark Theme";
    el("btnInstrumentSpx").classList.toggle("active", state.instrument === "spx");
    el("btnInstrumentSpy").classList.toggle("active", state.instrument === "spy");

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
      try { localStorage.setItem("spx500_theme", state.theme); } catch (_) {}
      document.documentElement.setAttribute("data-theme", state.theme);
      el("btnThemeToggle").textContent = state.theme === "dark" ? "Light Theme" : "Dark Theme";
      state.lastBacktestFingerprint = "";
      renderAll();
    });

    el("btnRefreshAll").addEventListener("click", () => loadAllData(true));
    el("btnAutoRefresh").addEventListener("click", cycleAutoRefresh);

    el("btnResimulate").addEventListener("click", runBootstrapSimulation);
    el("btnRecordSignal").addEventListener("click", () => recordCurrentSignal(false));
    el("btnRecordBias").addEventListener("click", () => recordCurrentSignal(true));
    el("btnLedgerRecord").addEventListener("click", () => recordCurrentSignal(true));
    el("btnLedgerEvaluate").addEventListener("click", evaluateOpenLedger);
    el("btnLedgerClear").addEventListener("click", clearForwardLedger);
    const btnAdmin = el("btnAdminUnlock");
    if (btnAdmin) btnAdmin.addEventListener("click", promptAndVerifyAdminKey);

    const form = el("manualTradeForm");
    if (form) {
      form.addEventListener("submit", submitManualTrade);
    }

    // Delegated handler for per-row ledger update buttons (TP1 / Stop / Market / Delete)
    const ledgerWrap = el("ledgerTableContainer");
    if (ledgerWrap) {
      ledgerWrap.addEventListener("click", (evt) => {
        const btn = evt.target.closest("button[data-ledger-action]");
        if (!btn) return;
        const tradeId = btn.getAttribute("data-trade-id");
        const action = btn.getAttribute("data-ledger-action");
        const inst = btn.getAttribute("data-inst") || state.instrument.toUpperCase();
        updateLedgerTrade(tradeId, action, inst);
      });
    }

    // 2026 Bento cursor spotlight (rAF-throttled, zero layout thrash)
    let rafPending = false;
    document.addEventListener("pointermove", (e) => {
      if (rafPending) return;
      const card = e.target.closest && e.target.closest(".card");
      if (!card) return;
      rafPending = true;
      window.requestAnimationFrame(() => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${e.clientX - rect.left}px`);
        card.style.setProperty("--my", `${e.clientY - rect.top}px`);
        rafPending = false;
      });
    }, { passive: true });
  }

  function cycleAutoRefresh() {
    const seq = [0, 2000, 5000];
    const idx = seq.indexOf(state.autoRefreshMs);
    state.autoRefreshMs = seq[(idx + 1) % seq.length];
    if (state.autoTimer) {
      window.clearInterval(state.autoTimer);
      state.autoTimer = null;
    }
    if (state.autoRefreshMs > 0) {
      el("btnAutoRefresh").textContent = `Auto: ${state.autoRefreshMs / 1000}s`;
      state.autoTimer = window.setInterval(() => loadAllData(false), state.autoRefreshMs);
      showToast(`Live polling enabled (${state.autoRefreshMs / 1000}s interval)`, "success");
    } else {
      el("btnAutoRefresh").textContent = "Auto: Off";
    }
  }

  function switchInstrument(inst) {
    state.instrument = inst;
    try { localStorage.setItem("spx500_inst", inst); } catch (_) {}
    el("btnInstrumentSpx").classList.toggle("active", inst === "spx");
    el("btnInstrumentSpy").classList.toggle("active", inst === "spy");
    const mfInst = el("mfInst");
    if (mfInst) mfInst.value = inst.toUpperCase();
    renderTelemetry();
  }

  // ---- Fast Networking & Instant Session Cache ----
  function getAdminKey() {
    try { return sessionStorage.getItem(SESSION_ADMIN_KEY) || ""; } catch (_) { return ""; }
  }

  function updateAdminUi() {
    const btn = el("btnAdminUnlock");
    if (!btn) return;
    const req = Boolean(state.security && state.security.admin_key_required);
    btn.style.display = req ? "inline-flex" : "none";
    if (!req) return;
    const authed = Boolean(state.security && state.security.authorized);
    btn.textContent = authed ? "Admin Unlocked" : "Unlock Write Access";
    btn.classList.toggle("btn-bull", authed);
  }

  async function syncSecurityStatus() {
    try {
      const headers = {};
      const key = getAdminKey();
      if (key) headers["X-Admin-Key"] = key;
      const status = await fetchJson("/api/security/status", { headers });
      state.security = status;
      if (status.admin_key_required && !status.authorized && key) {
        try { sessionStorage.removeItem(SESSION_ADMIN_KEY); } catch (_) {}
      }
      updateAdminUi();
    } catch (_) {}
  }

  async function promptAndVerifyAdminKey() {
    if (state.security && state.security.authorized && getAdminKey()) {
      const lock = window.confirm("Admin write access is currently unlocked in this browser tab. Lock it now?");
      if (lock) {
        try { sessionStorage.removeItem(SESSION_ADMIN_KEY); } catch (_) {}
        state.security.authorized = false;
        updateAdminUi();
        showToast("Admin write access locked for this tab.", "warn");
      }
      return;
    }
    const candidate = window.prompt("Enter ALPHA_ADMIN_KEY (from Render Environment settings) to unlock live ledger edits:");
    if (!candidate || !candidate.trim()) return;
    try {
      const res = await fetch("/api/security/verify", {
        method: "POST",
        cache: "no-store",
        headers: { "X-Admin-Key": candidate.trim() },
      });
      if (!res.ok) {
        try { sessionStorage.removeItem(SESSION_ADMIN_KEY); } catch (_) {}
        state.security.authorized = false;
        updateAdminUi();
        showToast("Invalid Admin Key. Public read-only mode remains active.", "error");
        return;
      }
      try { sessionStorage.setItem(SESSION_ADMIN_KEY, candidate.trim()); } catch (_) {}
      state.security.authorized = true;
      updateAdminUi();
      showToast("Admin key verified for this session. Live ledger edits unlocked.", "success");
    } catch (err) {
      showToast(`Verification failed: ${err.message}`, "error");
    }
  }

  async function fetchJson(url, options = {}) {
    const mergedHeaders = { ...(options.headers || {}) };
    const method = (options.method || "GET").toUpperCase();
    const key = getAdminKey();
    if (key && !mergedHeaders["X-Admin-Key"]) {
      mergedHeaders["X-Admin-Key"] = key;
    }
    const resp = await fetch(url, { cache: "no-store", ...options, headers: mergedHeaders });
    if (!resp.ok) {
      const txt = await resp.text();
      if (resp.status === 401 && method === "POST") {
        try { sessionStorage.removeItem(SESSION_ADMIN_KEY); } catch (_) {}
        if (state.security) {
          state.security.admin_key_required = true;
          state.security.authorized = false;
          updateAdminUi();
        }
        throw new Error("Admin Key required. Click 'Unlock Write Access' in the header or Forward Ledger.");
      }
      throw new Error(`HTTP ${resp.status} from ${url}: ${txt.slice(0, 160)}`);
    }
    return resp.json();
  }

  function hydrateFromSessionCache() {
    try {
      const tRaw = sessionStorage.getItem(SESSION_TEL_KEY);
      const bRaw = sessionStorage.getItem(SESSION_BT_KEY);
      if (tRaw && bRaw) {
        const t0 = performance.now();
        state.telemetry = JSON.parse(tRaw);
        state.backtest = JSON.parse(bRaw);
        renderAll();
        const dt = performance.now() - t0;
        const pill = el("latencyPill");
        if (pill) pill.textContent = `${dt.toFixed(1)} ms (cache)`;
      }
    } catch (_) {}
  }

  async function loadAllData(forceBgRefresh = false) {
    setErrorBanner("");
    const t0 = performance.now();
    try {
      const telUrl = forceBgRefresh ? "/api/telemetry?refresh=1" : "/api/telemetry";
      const [telemetry, backtest] = await Promise.all([
        fetchJson(telUrl),
        fetchJson("/api/backtest"),
      ]);
      state.telemetry = telemetry;
      state.backtest = backtest;

      try {
        sessionStorage.setItem(SESSION_TEL_KEY, JSON.stringify(telemetry));
        sessionStorage.setItem(SESSION_BT_KEY, JSON.stringify(backtest));
      } catch (_) {}

      renderAll();
      const elapsed = performance.now() - t0;
      const pill = el("latencyPill");
      if (pill) {
        pill.textContent = `${elapsed.toFixed(1)} ms`;
        flashElement(pill);
      }
    } catch (err) {
      setErrorBanner(`Telemetry load failed: ${err.message}`);
    }
  }

  // ---- Interactive Backend Actions ----
  async function runBootstrapSimulation() {
    if (!state.telemetry) return;
    const t0 = performance.now();
    const msgBox = el("recordStatusMsg");
    msgBox.textContent = "Running empirical OHLC return bootstrap...";
    try {
      const inst = state.instrument;
      const instData = state.telemetry[inst];
      const plan = instData.trade_plan || {};
      const payload = {
        instrument: inst.toUpperCase(),
        direction: plan.raw_direction === "SHORT" || plan.signal === "SHORT" ? "SHORT" : "LONG",
        paths: 1500,
        seed: Math.floor(Date.now() % 100000),
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
      // Update Step 7 agent card in state cleanly so UI stays consistent
      const step7 = (state.telemetry.agent_reasoning || []).find((a) => Number(a.step) === 7);
      if (step7) {
        step7.vote = sim.verdict || step7.vote;
        step7.output_summary = `WinRate=${fmtNum(sim.win_rate, 1)}%, Exp=${fmtSigned(sim.expected_r, 3, "R")} (${sim.paths_simulated} ${sim.direction} paths)`;
        state.lastAgentFingerprint = "";
        renderAgentPipeline(state.telemetry.agent_reasoning || []);
      }
      renderFanChart();
      const dt = (performance.now() - t0).toFixed(1);
      const summary = `Bootstrap (${sim.direction}, ${sim.paths_simulated} paths): Win=${fmtNum(sim.win_rate, 1)}%, Exp=${fmtSigned(sim.expected_r, 3, "R")} (${dt} ms)`;
      msgBox.textContent = summary;
      showToast(summary, "success");
    } catch (err) {
      msgBox.textContent = `Simulation error: ${err.message}`;
      showToast(`Simulation error: ${err.message}`, "error");
    }
  }

  async function recordCurrentSignal(allowBias = false) {
    const t0 = performance.now();
    const msgBox = el("recordStatusMsg");
    const ledgerMsg = el("ledgerStatusMsg");
    try {
      const res = await fetchJson("/api/ledger/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instrument: state.instrument.toUpperCase(), allow_bias: allowBias }),
      });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      const dt = (performance.now() - t0).toFixed(1);
      if (res.recorded) {
        const text = `Recorded ${res.trade.trade_id}: ${res.trade.direction} ${res.trade.instrument} @ ${fmtNum(res.trade.entry_price)} (${dt} ms)`;
        if (msgBox) msgBox.textContent = text;
        if (ledgerMsg) ledgerMsg.textContent = text;
        showToast(text, "success");
      } else {
        const warnText = `${res.reason || "Blocked by Risk Guardian (NO_TRADE)."} Click "Log Bias Override" or use the Forward Ledger form to record manually.`;
        if (msgBox) msgBox.textContent = warnText;
        if (ledgerMsg) ledgerMsg.textContent = warnText;
        showToast(warnText, "warn");
      }
    } catch (err) {
      showToast(`Ledger error: ${err.message}`, "error");
    }
  }

  async function submitManualTrade(evt) {
    evt.preventDefault();
    const t0 = performance.now();
    const inst = (el("mfInst").value || state.instrument.toUpperCase()).toUpperCase();
    const direction = el("mfDir").value || "LONG";
    const entryVal = el("mfEntry").value ? Number(el("mfEntry").value) : null;
    const stopVal = el("mfStop").value ? Number(el("mfStop").value) : null;
    const targetVal = el("mfTarget").value ? Number(el("mfTarget").value) : null;

    try {
      const res = await fetchJson("/api/ledger/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instrument: inst,
          direction,
          entry_price: entryVal,
          stop: stopVal,
          target: targetVal,
        }),
      });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      const dt = (performance.now() - t0).toFixed(1);
      if (res.recorded) {
        const text = `Logged ${res.trade.trade_id} (${res.trade.direction} ${res.trade.instrument} @ ${fmtNum(res.trade.entry_price)}) in ${dt} ms`;
        el("ledgerStatusMsg").textContent = text;
        el("manualTradeForm").reset();
        el("mfInst").value = state.instrument.toUpperCase();
        showToast(text, "success");
      } else {
        showToast(res.reason || "Could not log paper trade.", "warn");
      }
    } catch (err) {
      showToast(`Manual trade failed: ${err.message}`, "error");
    }
  }

  async function updateLedgerTrade(tradeId, action, instrument) {
    const t0 = performance.now();
    try {
      const res = await fetchJson("/api/ledger/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trade_id: tradeId, action, instrument }),
      });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      const dt = (performance.now() - t0).toFixed(1);
      const text = res.updated
        ? `Ledger updated (${action} ${tradeId}) in ${dt} ms`
        : res.reason || "Update failed.";
      el("ledgerStatusMsg").textContent = text;
      showToast(text, res.updated ? "success" : "warn");
    } catch (err) {
      showToast(`Update failed: ${err.message}`, "error");
    }
  }

  async function evaluateOpenLedger() {
    const t0 = performance.now();
    try {
      const res = await fetchJson("/api/ledger/evaluate", { method: "POST" });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      const dt = (performance.now() - t0).toFixed(1);
      const msg = `Evaluated open trades vs latest OHLC bars (${dt} ms). Open=${res.ledger.open_signals}, Resolved=${res.ledger.resolved_signals}`;
      el("ledgerStatusMsg").textContent = msg;
      showToast(msg, "success");
    } catch (err) {
      showToast(`Evaluate error: ${err.message}`, "error");
    }
  }

  async function clearForwardLedger() {
    const t0 = performance.now();
    try {
      const res = await fetchJson("/api/ledger/clear", { method: "POST" });
      if (res.ledger && state.telemetry) {
        state.telemetry.ledger = res.ledger;
        renderLedger();
      }
      const dt = (performance.now() - t0).toFixed(1);
      const text = `Forward paper ledger cleared in ${dt} ms.`;
      if (el("recordStatusMsg")) el("recordStatusMsg").textContent = text;
      if (el("ledgerStatusMsg")) el("ledgerStatusMsg").textContent = text;
      showToast(text, "info");
    } catch (err) {
      showToast(`Clear error: ${err.message}`, "error");
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

    const chgColor = instData.change_pts >= 0 ? "var(--bull)" : "var(--bear)";
    const kpiLbl = el("kpiSpotLabel");
    if (kpiLbl) kpiLbl.textContent = `${instKey.toUpperCase()} SPOT`;
    const metaLbl = el("sectionMetaLabel");
    if (metaLbl) metaLbl.textContent = `${instKey.toUpperCase()} / 5 MIN`;

    const spotEl = el("stripSpot");
    const nextSpot = fmtNum(instData.spot);
    if (spotEl.textContent !== nextSpot) {
      spotEl.textContent = nextSpot;
      flashElement(spotEl);
    }
    spotEl.style.color = chgColor;
    el("stripChange").textContent = `${fmtSigned(instData.change_pts)} (${fmtSigned(instData.change_pct, 2, "%")})`;
    el("stripChange").style.color = chgColor;

    const macro = t.macro || {};
    el("stripVix").textContent = fmtNum(macro.vix);
    el("stripUs10y").textContent = `${fmtNum(macro.us10y, 3)}%`;
    const dxySub = el("stripDxySub");
    if (dxySub) dxySub.textContent = `Treasury yield · DXY ${fmtNum(macro.dxy, 3)}`;
    const gex = instData.gex || {};
    const gexEl = el("stripGex");
    gexEl.textContent = `${fmtSigned(gex.net_gex_total_b, 3, "B")}`;
    gexEl.style.color = Number(gex.net_gex_total_b || 0) >= 0 ? "var(--bull)" : "var(--bear)";
    const gexReg = el("stripGexRegime");
    if (gexReg) gexReg.textContent = gex.gamma_regime ? String(gex.gamma_regime).replaceAll("_", " ").toLowerCase() : "Option gamma regime";

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

  // ---- Zero-Thrash TradingView Lightweight Charts ----
  function renderTradingViewChart(instKey, instData) {
    const container = el("tvChartContainer");
    if (!container || typeof window.LightweightCharts === "undefined") return;

    const c = chartColors();
    if (!state.charts.tvChart) {
      const chart = window.LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: { background: { color: c.bg }, textColor: c.text },
        grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
        crosshair: { mode: 0 },
        timeScale: { borderColor: c.grid, timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderColor: c.grid },
      });

      state.charts.candleSeries = chart.addCandlestickSeries({
        upColor: c.bull,
        downColor: c.bear,
        borderVisible: false,
        wickUpColor: c.bull,
        wickDownColor: c.bear,
      });

      state.charts.vwapSeries = chart.addLineSeries({
        color: c.vwap,
        lineWidth: 2,
        title: "VWAP",
      });

      const volSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "",
        color: "rgba(155, 147, 160, 0.25)",
      });
      volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      state.charts.volumeSeries = volSeries;
      state.charts.tvChart = chart;
    } else {
      state.charts.tvChart.applyOptions({
        layout: { background: { color: c.bg }, textColor: c.text },
        grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
        timeScale: { borderColor: c.grid },
        rightPriceScale: { borderColor: c.grid },
      });
      state.charts.candleSeries.applyOptions({
        upColor: c.bull,
        downColor: c.bear,
        wickUpColor: c.bull,
        wickDownColor: c.bear,
      });
      state.charts.vwapSeries.applyOptions({ color: c.vwap });
    }

    const rawCandles = Array.isArray(instData.candles) ? instData.candles : [];
    const seen = new Set();
    const clean = [];
    for (const bar of rawCandles) {
      const ts = Number(bar.time);
      if (!Number.isFinite(ts) || seen.has(ts)) continue;
      seen.add(ts);
      clean.push(bar);
    }
    clean.sort((a, b) => a.time - b.time);

    state.charts.candleSeries.setData(
      clean.map((x) => ({ time: x.time, open: x.open, high: x.high, low: x.low, close: x.close }))
    );
    state.charts.vwapSeries.setData(
      clean.filter((x) => Number.isFinite(Number(x.vwap))).map((x) => ({ time: x.time, value: Number(x.vwap) }))
    );
    state.charts.volumeSeries.setData(
      clean.map((x) => ({
        time: x.time,
        value: Number(x.volume || 0),
        color: x.close >= x.open ? "rgba(163, 217, 189, 0.24)" : "rgba(239, 156, 130, 0.24)",
      }))
    );

    for (const pl of state.charts.priceLines) {
      try { state.charts.candleSeries.removePriceLine(pl); } catch (_) {}
    }
    state.charts.priceLines = [];

    const plan = instData.trade_plan || {};
    const levels = [
      { price: plan.entry, color: c.accent, title: `ENTRY ${instKey.toUpperCase()}`, style: 0 },
      { price: plan.stop_loss, color: c.bear, title: "STOP LOSS", style: 2 },
      { price: plan.tp1, color: c.bull, title: "TP1", style: 2 },
      { price: plan.tp2, color: c.bull, title: "TP2", style: 1 },
    ];
    for (const lv of levels) {
      if (Number.isFinite(Number(lv.price))) {
        const line = state.charts.candleSeries.createPriceLine({
          price: Number(lv.price),
          color: lv.color,
          lineWidth: 1,
          lineStyle: lv.style,
          axisLabelVisible: true,
          title: lv.title,
        });
        state.charts.priceLines.push(line);
      }
    }
    state.charts.tvChart.timeScale().fitContent();
  }

  function upsertLineOrBarChart(key, canvasId, config) {
    const canvas = el(canvasId);
    if (!canvas || typeof window.Chart === "undefined") return;
    const existing = state.charts[key];
    if (existing) {
      existing.data = config.data;
      existing.options = config.options;
      existing.update("none");
      return;
    }
    state.charts[key] = new window.Chart(canvas, config);
  }

  // ---- Chart.js Charts ----
  function renderFanChart() {
    if (!state.telemetry) return;
    const instKey = state.instrument;
    const instData = state.telemetry[instKey] || {};
    const fan = instData.fan_series || {};
    const tm = state.telemetry.timemachine_simulation || {};

    el("tmSummarySubtitle").textContent =
      `${tm.bootstrap_mode || "EMPIRICAL_BAR_BOOTSTRAP"} · Paths=${tm.paths_simulated || "--"} · Win=${fmtNum(tm.win_rate, 1)}% · Exp=${fmtSigned(tm.expected_r, 3, "R")} (${tm.verdict || "--"})`;

    const c = chartColors();
    const steps = (fan.steps || []).map((s) => `+${s * 5}m`);

    upsertLineOrBarChart("fan", "fanChartCanvas", {
      type: "line",
      data: {
        labels: steps,
        datasets: [
          { label: "P95", data: fan.p95 || [], borderColor: "rgba(163, 217, 189, 0.55)", borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0, fill: false },
          { label: "P75", data: fan.p75 || [], borderColor: c.bull, backgroundColor: "rgba(163, 217, 189, 0.14)", borderWidth: 1.5, pointRadius: 0, fill: "+1" },
          { label: "P50 (Median)", data: fan.p50 || [], borderColor: c.ivory, borderWidth: 2.5, pointRadius: 2, fill: false },
          { label: "P25", data: fan.p25 || [], borderColor: c.warn, backgroundColor: "rgba(239, 156, 130, 0.12)", borderWidth: 1.5, pointRadius: 0, fill: "-1" },
          { label: "P5", data: fan.p5 || [], borderColor: "rgba(239, 156, 130, 0.65)", borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0, fill: false },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 180 },
        plugins: { legend: { labels: { color: c.text, boxWidth: 12, font: { size: 11 } } } },
        scales: { x: { ticks: { color: c.text }, grid: { color: c.grid } }, y: { ticks: { color: c.text }, grid: { color: c.grid } } },
      },
    });
  }

  function renderShapChart() {
    if (!state.telemetry) return;
    const contribs = (state.telemetry.feature_contributions || []).slice(0, 8);
    const c = chartColors();
    upsertLineOrBarChart("shap", "shapChartCanvas", {
      type: "bar",
      data: {
        labels: contribs.map((x) => x.feature),
        datasets: [{
          label: "TreeSHAP Log-Odds Contribution (UP vs DOWN)",
          data: contribs.map((x) => Number(x.shap_contribution || 0)),
          backgroundColor: contribs.map((x) => (Number(x.shap_contribution) >= 0 ? c.bull : c.bear)),
          borderRadius: 5,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 180 },
        plugins: { legend: { labels: { color: c.text, font: { size: 11 } } } },
        scales: { x: { ticks: { color: c.text }, grid: { color: c.grid } }, y: { ticks: { color: c.text, font: { size: 11 } }, grid: { color: c.grid } } },
      },
    });
  }

  // ---- 8-Agent Planning Pipeline ----
  function renderAgentPipeline(agents) {
    const grid = el("agentFlowGrid");
    if (!grid) return;

    const fp = JSON.stringify(agents);
    if (fp === state.lastAgentFingerprint) return;
    state.lastAgentFingerprint = fp;

    if (!agents.length) {
      grid.innerHTML = `<div class="empty-state">No agent reasoning trace returned from telemetry.</div>`;
      return;
    }

    grid.innerHTML = agents
      .map((ag, idx) => {
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
        <article class="agent-card" style="--stagger: ${idx};">
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

    const fp = `${state.theme}|${bt.generated_at_utc || ""}|${bt.strategy?.bars_evaluated || 0}`;
    if (fp === state.lastBacktestFingerprint) return;
    state.lastBacktestFingerprint = fp;

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
    const c = chartColors();
    const labels = (strat.labels || []).map((s) => String(s).slice(5, 16));

    upsertLineOrBarChart("equity", "equityChartCanvas", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Strategy Equity ($)", data: strat.equity_curve || [], borderColor: c.bull, borderWidth: 2, pointRadius: 0 },
          { label: "SPY Buy & Hold ($)", data: strat.benchmark_curve || [], borderColor: c.text, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: c.text } } },
        scales: { x: { ticks: { color: c.text, maxTicksLimit: 8 }, grid: { color: c.grid } }, y: { ticks: { color: c.text }, grid: { color: c.grid } } },
      },
    });

    upsertLineOrBarChart("drawdown", "drawdownChartCanvas", {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "Drawdown (%)",
          data: strat.drawdown_curve || [],
          borderColor: c.bear,
          backgroundColor: "rgba(239, 156, 130, 0.16)",
          fill: true,
          borderWidth: 1.5,
          pointRadius: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: c.text } } },
        scales: { x: { ticks: { color: c.text, maxTicksLimit: 8 }, grid: { color: c.grid } }, y: { ticks: { color: c.text }, grid: { color: c.grid } } },
      },
    });
  }

  function renderCalibrationAndConfusion(ho, cal) {
    el("evBrierSubtitle").textContent =
      `Multiclass Brier Score: ${fmtNum(cal.brier_score, 4)} · ECE: ${fmtNum(cal.ece, 4)} · Accuracy: ${fmtNum((ho.accuracy || 0) * 100, 1)}%`;

    const c = chartColors();
    const bins = cal.bins || [];
    upsertLineOrBarChart("calibration", "calibrationChartCanvas", {
      type: "line",
      data: {
        labels: bins.map((b) => fmtNum(b.mean_confidence, 2)),
        datasets: [
          { label: "Empirical Accuracy", data: bins.map((b) => b.empirical_accuracy), borderColor: c.accent, backgroundColor: c.accent, borderWidth: 2, pointRadius: 4 },
          { label: "Perfect Calibration (y=x)", data: bins.map((b) => b.mean_confidence), borderColor: c.text, borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: c.text } } },
        scales: {
          x: { title: { display: true, text: "Predicted Confidence", color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
          y: { min: 0, max: 1, title: { display: true, text: "Observed Accuracy", color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
        },
      },
    });

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

    const c = chartColors();
    const strikes = gex.strikes || [];
    upsertLineOrBarChart("gex", "gexChartCanvas", {
      type: "bar",
      data: {
        labels: strikes.map((s) => String(s.strike)),
        datasets: [
          { label: "Call GEX ($M)", data: strikes.map((s) => s.call_gex), backgroundColor: c.bull },
          { label: "Put GEX ($M)", data: strikes.map((s) => s.put_gex), backgroundColor: c.bear },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 180 },
        plugins: { legend: { labels: { color: c.text } } },
        scales: { x: { ticks: { color: c.text }, grid: { color: c.grid } }, y: { ticks: { color: c.text }, grid: { color: c.grid } } },
      },
    });

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
      wrap.innerHTML = `<div class="empty-state">Forward paper ledger is empty. No fabricated history is displayed. Use <strong>Record Signal / Bias</strong> or <strong>+ Log Paper Trade</strong> above to track forward signals in real time.</div>`;
      return;
    }
    wrap.innerHTML = `
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Trade ID</th>
              <th>Timestamp (ET)</th>
              <th>Inst</th>
              <th>Dir</th>
              <th>Entry</th>
              <th>Stop</th>
              <th>TP1</th>
              <th>Outcome</th>
              <th>Exit</th>
              <th>R</th>
              <th>P&amp;L ($)</th>
              <th>Actions</th>
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
                <td class="mono">${fmtNum(tr.current_stop ?? tr.stop)}</td>
                <td class="mono">${fmtNum(tr.target)}</td>
                <td><span class="${votePillClass(tr.outcome)}">${escHtml(tr.outcome)}</span></td>
                <td class="mono">${tr.exit_price != null ? fmtNum(tr.exit_price) : "--"}</td>
                <td class="mono">${fmtSigned(tr.return_r || 0, 2, "R")}</td>
                <td class="mono">${fmtSigned(tr.pnl_usd || 0, 2)}</td>
                <td>
                  <div style="display:flex;gap:4px;">
                    ${
                      tr.outcome === "OPEN"
                        ? `<button type="button" class="btn btn-sm btn-bull" data-ledger-action="CLOSE_TARGET" data-trade-id="${escHtml(tr.trade_id)}" data-inst="${escHtml(tr.instrument)}" title="Settle at TP1">TP1</button>
                           <button type="button" class="btn btn-sm btn-danger" data-ledger-action="CLOSE_STOP" data-trade-id="${escHtml(tr.trade_id)}" data-inst="${escHtml(tr.instrument)}" title="Settle at Stop">SL</button>
                           <button type="button" class="btn btn-sm" data-ledger-action="CLOSE_MARKET" data-trade-id="${escHtml(tr.trade_id)}" data-inst="${escHtml(tr.instrument)}" title="Close at current spot">Mkt</button>`
                        : ""
                    }
                    <button type="button" class="btn btn-sm btn-danger" data-ledger-action="DELETE" data-trade-id="${escHtml(tr.trade_id)}" data-inst="${escHtml(tr.instrument)}" title="Delete row">Del</button>
                  </div>
                </td>
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
    hydrateFromSessionCache();
    syncSecurityStatus();
    loadAllData(false);
  });
})();

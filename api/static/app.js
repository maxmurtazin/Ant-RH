/* global window, document, fetch */

const $ = (id) => document.getElementById(id);

const allowedStages = new Set([
  "study",
  "analyze",
  "journal",
  "docs",
  "topo-eval",
  "topo-report",
  "gemma-health",
  "stability",
]);

function fmt(x) {
  if (x === null || x === undefined) return "—";
  if (typeof x === "number") {
    if (!Number.isFinite(x)) return String(x);
    return Math.abs(x) >= 1000 ? x.toFixed(2) : x.toPrecision(8).replace(/\.?0+$/, "");
  }
  return String(x);
}

function setBadge(el, kind, text) {
  el.classList.remove("badge-good", "badge-bad", "badge-warn", "badge-muted");
  el.classList.add(kind);
  el.textContent = text;
}

async function apiGet(path) {
  const res = await fetch(path, { method: "GET" });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`GET ${path} failed (${res.status}): ${txt}`);
  }
  return await res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let msg = "";
    try {
      const j = await res.json();
      msg = j?.detail ? j.detail : JSON.stringify(j);
    } catch {
      msg = await res.text();
    }
    throw new Error(`POST ${path} failed (${res.status}): ${msg}`);
  }
  return await res.json();
}

function deriveBottlenecksAndNextActions(status, acoMetrics, topoMetrics) {
  const bottlenecks = [];
  const next = [];

  if (Array.isArray(status?.missing) && status.missing.length > 0) {
    bottlenecks.push(`missing artifacts: ${status.missing.join(", ")}`);
    next.push("run: study → analyze-gemma → lab-journal to regenerate context");
  }

  if (status?.gemma_main_issue) {
    bottlenecks.push(`main_issue: ${status.gemma_main_issue}`);
  }

  if (acoMetrics?.trend === "increasing") {
    bottlenecks.push("ACO losses are trending up (non-learning / scaling issue)");
    next.push("run: analyze-gemma (diagnose scaling/alignment)");
  } else if (acoMetrics?.trend === "decreasing") {
    next.push("ACO improving: consider running topo-eval for cross-check");
  }

  if (typeof topoMetrics?.advantage_over_random === "number") {
    if (topoMetrics.advantage_over_random <= 0) {
      bottlenecks.push("TopologicalLM mean_reward is not beating random (advantage ≤ 0)");
      next.push("run: topo-eval after dataset/training tweaks");
    } else {
      next.push("TopologicalLM shows positive advantage: run topo-eval periodically");
    }
  }

  next.push("run: docs to update narrative outputs");

  return { bottlenecks, next };
}

function renderList(el, items) {
  el.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "—";
    el.appendChild(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  }
}

function nowStamp() {
  const d = new Date();
  return d.toLocaleString();
}

function fmtSci(x) {
  if (x === null || x === undefined) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "0.00e+00";
  return n.toExponential(2);
}

function setStatus(el, status) {
  const s = (status || "unknown").toString().toLowerCase();
  el.classList.remove(
    "status-ok",
    "status-approx",
    "status-broken",
    "status-unknown",
    "status-chaotic",
    "status-integrable",
    "status-intermediate"
  );
  const map = {
    ok: "status-ok",
    approx: "status-approx",
    broken: "status-broken",
    unknown: "status-unknown",
    degenerate: "status-approx",
    chaotic: "status-chaotic",
    integrable: "status-integrable",
    intermediate: "status-intermediate",
  };
  el.classList.add(map[s] || "status-unknown");
  el.textContent = (s || "unknown").toUpperCase();
}

async function refreshPhysics() {
  let phys = null;
  try {
    phys = await apiGet("/metrics/physics");
  } catch (e) {
    phys = { source: "error", self_adjoint_status: "unknown", spectral_status: "unknown", otoc_indicator: "unknown" };
  }

  $("physSource").textContent = phys?.source ? String(phys.source) : "—";

  setStatus($("physSelfAdjBadge"), phys?.self_adjoint_status || "unknown");
  $("physSelfAdjErr").textContent =
    phys?.self_adjoint_error !== null && phys?.self_adjoint_error !== undefined ? fmtSci(phys.self_adjoint_error) : "—";

  setStatus($("physSpecBadge"), phys?.spectral_status || "unknown");
  $("physSpacingStd").textContent =
    phys?.spacing_std !== null && phys?.spacing_std !== undefined ? fmt(phys.spacing_std) : "—";
  $("physSpectrumReal").textContent =
    phys?.spectrum_real === true ? "true" : phys?.spectrum_real === false ? "false" : "—";

  setStatus($("physOtocBadge"), phys?.otoc_indicator || "unknown");
  $("physRMean").textContent = phys?.r_mean !== null && phys?.r_mean !== undefined ? fmt(phys.r_mean) : "—";
}

async function refreshAll() {
  const btn = $("btnRefresh");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  try {
    const health = await apiGet("/health");
    setBadge($("healthBadge"), "badge-good", health?.status === "ok" ? "healthy" : "unknown");

    const [status, aco, topo, gemmaHealth] = await Promise.all([
      apiGet("/status"),
      apiGet("/metrics/aco"),
      apiGet("/metrics/topological-lm"),
      apiGet("/health/gemma").catch((e) => ({ status: "unknown", message: String(e?.message || e) })),
    ]);

    // Status badge
    if (Array.isArray(status?.missing) && status.missing.length === 0) {
      setBadge($("statusBadge"), "badge-good", "ok");
    } else {
      setBadge($("statusBadge"), "badge-warn", "partial");
    }

    // Status summary
    const summary = [
      `gemma_learning=${fmt(status?.gemma_learning)}`,
      `main_issue=${fmt(status?.gemma_main_issue)}`,
      `operator_total_loss=${fmt(status?.operator_total_loss)}`,
      `operator_spectral_loss=${fmt(status?.operator_spectral_loss)}`,
      `operator_spacing_loss=${fmt(status?.operator_spacing_loss)}`,
    ].join("\n");
    $("statusSummary").textContent = summary;

    // Missing
    $("missingList").textContent =
      Array.isArray(status?.missing) && status.missing.length ? status.missing.join("\n") : "none";

    // Project memory
    $("projectMemory").textContent = status?.project_memory_head ? status.project_memory_head : "—";

    // ACO metrics
    $("acoBest").textContent = fmt(aco?.best_loss);
    $("acoMean").textContent = fmt(aco?.mean_loss);
    $("acoTrend").textContent = fmt(aco?.trend);
    $("acoRows").textContent = fmt(aco?.n_rows);

    if (aco?.trend === "decreasing") setBadge($("acoTrendBadge"), "badge-good", "decreasing");
    else if (aco?.trend === "increasing") setBadge($("acoTrendBadge"), "badge-bad", "increasing");
    else setBadge($("acoTrendBadge"), "badge-muted", fmt(aco?.trend));

    // Topo metrics
    $("topoRandom").textContent = fmt(topo?.random_mean_reward);
    $("topoMean").textContent = fmt(topo?.topological_lm_mean_reward);
    $("topoAdv").textContent = fmt(topo?.advantage_over_random);
    $("topoUnique").textContent = fmt(topo?.unique_candidate_ratio);
    $("topoValid").textContent = fmt(topo?.valid_braid_ratio);

    if (typeof topo?.advantage_over_random === "number") {
      if (topo.advantage_over_random > 0) setBadge($("topoBadge"), "badge-good", "advantage > 0");
      else setBadge($("topoBadge"), "badge-warn", "advantage ≤ 0");
    } else {
      setBadge($("topoBadge"), "badge-muted", "unknown");
    }

    const { bottlenecks, next } = deriveBottlenecksAndNextActions(status, aco, topo);
    renderList($("bottlenecks"), bottlenecks);
    renderList($("nextActions"), next);

    // Gemma health card
    const overall = gemmaHealth?.overall_status || gemmaHealth?.status || "unknown";
    $("gemmaOverall").textContent = fmt(overall);
    $("gemmaHealthRaw").textContent = JSON.stringify(gemmaHealth, null, 2);

    const checks = Array.isArray(gemmaHealth?.checks) ? gemmaHealth.checks : [];
    const byName = new Map(checks.map((c) => [c?.name, c]));
    const s = (name) => {
      const item = byName.get(name);
      return item?.status ? String(item.status) : "—";
    };
    $("gemmaPlanner").textContent = s("planner");
    $("gemmaAnalyzer").textContent = s("analyzer");
    $("gemmaHelp").textContent = s("help");
    $("gemmaJournal").textContent = s("lab_journal");
    $("gemmaPaper").textContent = s("paper_writer");
    $("gemmaDocs").textContent = s("docs_builder");
    $("gemmaLiterature").textContent = s("literature_study");

    if (overall === "ok") setBadge($("gemmaHealthBadge"), "badge-good", "ok");
    else if (overall === "degraded") setBadge($("gemmaHealthBadge"), "badge-warn", "degraded");
    else if (overall === "failed") setBadge($("gemmaHealthBadge"), "badge-bad", "failed");
    else setBadge($("gemmaHealthBadge"), "badge-muted", "unknown");

    await refreshPhysics();

    $("lastUpdated").textContent = `last updated: ${nowStamp()}`;
  } catch (e) {
    setBadge($("healthBadge"), "badge-bad", "error");
    setBadge($("statusBadge"), "badge-bad", "error");
    $("statusSummary").textContent = String(e?.message || e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh";
  }
}

async function runStage(stage) {
  if (!allowedStages.has(stage)) {
    $("runMeta").textContent = `refused: stage not whitelisted (${stage})`;
    return;
  }
  const buttons = Array.from(document.querySelectorAll("button[data-stage]"));
  for (const b of buttons) b.disabled = true;

  $("runMeta").textContent = `running stage: ${stage} …`;
  $("runStdout").textContent = "";
  $("runStderr").textContent = "";

  try {
    const res = await apiPost("/run/stage", { stage });
    const meta = [`ok=${fmt(res?.ok)}`, `target=${fmt(res?.target)}`, `rc=${fmt(res?.returncode)}`, `t=${fmt(res?.duration_s)}s`, `timed_out=${fmt(res?.timed_out)}`].join(
      "\n"
    );
    $("runMeta").textContent = meta;
    $("runStdout").textContent = res?.stdout ? res.stdout : "—";
    $("runStderr").textContent = res?.stderr ? res.stderr : "—";
    if (stage === "topo-eval" || stage === "stability") {
      await refreshPhysics();
    }
  } catch (e) {
    $("runMeta").textContent = `error: ${String(e?.message || e)}`;
  } finally {
    for (const b of buttons) b.disabled = false;
  }
}

async function askGemma() {
  const btn = $("btnGemma");
  const q = $("gemmaQuestion").value.trim();
  const voice = $("gemmaVoice").checked;
  if (!q) return;

  btn.disabled = true;
  btn.textContent = "Asking…";
  $("gemmaAnswer").textContent = "working…";
  try {
    const res = await apiPost("/gemma/help", { question: q, voice });
    $("gemmaAnswer").textContent = res?.answer ? res.answer : "—";
  } catch (e) {
    $("gemmaAnswer").textContent = `error: ${String(e?.message || e)}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask Gemma";
  }
}

function init() {
  $("btnRefresh").addEventListener("click", refreshAll);
  $("btnRefreshPhysics").addEventListener("click", refreshPhysics);
  for (const b of document.querySelectorAll("button[data-stage]")) {
    b.addEventListener("click", () => runStage(b.getAttribute("data-stage")));
  }
  $("btnGemma").addEventListener("click", askGemma);
  $("gemmaQuestion").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      askGemma();
    }
  });

  refreshAll();
}

window.addEventListener("DOMContentLoaded", init);


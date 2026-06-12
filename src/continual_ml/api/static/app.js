"use strict";

function drawChart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  const cssH = canvas.clientHeight || 220;  // layout height (CSS-pinned), not the buffer
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 48, r: 12, t: 10, b: 22 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;

  const allPts = series.flatMap((s) => s.points);
  if (allPts.length === 0) {
    ctx.fillStyle = "#8b97b8";
    ctx.font = "12px sans-serif";
    ctx.fillText("collecting data…", pad.l, pad.t + h / 2);
    return;
  }
  let xmin = Math.min(...allPts.map((p) => p.x));
  let xmax = Math.max(...allPts.map((p) => p.x));
  let ymin = Math.min(...allPts.map((p) => p.y));
  let ymax = Math.max(...allPts.map((p) => p.y));
  if (xmin === xmax) xmax = xmin + 1;
  const yPad = (ymax - ymin) * 0.1 || 1;
  ymin = opts.yZero ? Math.min(0, ymin) : ymin - yPad;
  ymax = ymax + yPad;

  const X = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * w;
  const Y = (y) => pad.t + h - ((y - ymin) / (ymax - ymin)) * h;

  ctx.strokeStyle = "#25304f";
  ctx.fillStyle = "#8b97b8";
  ctx.font = "10px ui-monospace, monospace";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yv = ymin + ((ymax - ymin) * i) / 4;
    const py = Y(yv);
    ctx.beginPath();
    ctx.moveTo(pad.l, py);
    ctx.lineTo(pad.l + w, py);
    ctx.stroke();
    ctx.fillText(yv.toFixed(0), 4, py + 3);
  }

  for (const s of series) {
    if (s.points.length === 0) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (const p of s.points) {
      if (p.y === null || p.y === undefined) { started = false; continue; }
      const px = X(p.x), py = Y(p.y);
      if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
    }
    ctx.stroke();
  }
}

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString());

async function post(url) { try { await fetch(url, { method: "POST" }); } catch (e) {} }

let panelReady = false;
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

async function initPredictPanel(s) {
  if (panelReady) return;
  panelReady = true;
  try {
    const z = await (await fetch("/zones")).json();
    if (z.available && z.zones.length) {
      buildTripForm(z.zones);
      return;
    }
  } catch (e) { /* fall through to raw form */ }
  buildRawForm(s.schema);
}

function zoneOptions(zones, selectedId) {
  return zones.map((z) =>
    `<option value="${z.id}" ${z.id === selectedId ? "selected" : ""}>${z.zone} · ${z.borough}</option>`
  ).join("");
}

function buildTripForm(zones) {
  $("predict-title").textContent = "Trip ETA predictor · pick zones";
  const hours = Array.from({ length: 24 }, (_, h) =>
    `<option value="${h}" ${h === 18 ? "selected" : ""}>${String(h).padStart(2, "0")}:00</option>`).join("");
  const days = DAYS.map((d, i) =>
    `<option value="${i}" ${i === 2 ? "selected" : ""}>${d}</option>`).join("");
  $("predict-form").innerHTML = `
    <div class="field"><label>Pickup zone</label><select id="t-pu">${zoneOptions(zones, "132")}</select></div>
    <div class="field"><label>Dropoff zone</label><select id="t-do">${zoneOptions(zones, "161")}</select></div>
    <div class="field"><label>Hour</label><select id="t-hour">${hours}</select></div>
    <div class="field"><label>Day</label><select id="t-day">${days}</select></div>
    <div class="field"><label>Passengers</label><input id="t-pax" type="number" value="1" min="1" max="6" /></div>
    <div class="field"><label>&nbsp;</label><button class="primary" id="t-go">Estimate ETA</button></div>`;
  $("t-go").onclick = runTripPredict;
}

async function runTripPredict() {
  const body = {
    pu_zone: $("t-pu").value, do_zone: $("t-do").value,
    pickup_hour: parseInt($("t-hour").value, 10),
    pickup_dayofweek: parseInt($("t-day").value, 10),
    passenger_count: parseInt($("t-pax").value, 10) || 1,
  };
  $("predict-result").textContent = "estimating…";
  try {
    const j = await (await fetch("/predict_trip", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })).json();
    if (j.error) { $("predict-result").textContent = j.error; return; }
    const rush = j.is_rush_hour ? " · rush hour" : "";
    $("predict-result").innerHTML =
      `→ ETA <b>${j.prediction_min} min</b> (${j.prediction_s} s) · ${j.distance_km} km straight-line · ` +
      `${j.pu_borough}→${j.do_borough}${rush} · ${Number(j.latency_ms).toFixed(2)} ms`;
  } catch (e) {
    $("predict-result").textContent = "prediction failed";
  }
}

function buildRawForm(schema) {
  if (!schema) { panelReady = false; return; }
  const wrap = $("predict-form");
  wrap.innerHTML = "";
  const fields = [...(schema.numeric || []), ...(schema.categorical || [])];
  for (const f of fields) {
    const d = document.createElement("div");
    d.className = "field";
    const isCat = (schema.categorical || []).includes(f);
    d.innerHTML = `<label>${f}</label><input data-f="${f}" type="${isCat ? "text" : "number"}"
      value="${isCat ? "" : 0}" step="any" />`;
    wrap.appendChild(d);
  }
  const btn = document.createElement("button");
  btn.className = "primary";
  btn.textContent = "Predict";
  btn.onclick = runRawPredict;
  const d = document.createElement("div");
  d.className = "field";
  d.innerHTML = "<label>&nbsp;</label>";
  d.appendChild(btn);
  wrap.appendChild(d);
}

async function runRawPredict() {
  const features = {};
  document.querySelectorAll("#predict-form input").forEach((el) => {
    features[el.dataset.f] = el.type === "number" ? parseFloat(el.value) : el.value;
  });
  try {
    const j = await (await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    })).json();
    $("predict-result").textContent =
      `→ predicted ${Number(j.prediction).toFixed(1)}  (v${j.model_version}, ${Number(j.latency_ms).toFixed(2)} ms)`;
  } catch (e) {
    $("predict-result").textContent = "prediction failed";
  }
}

async function tick() {
  let s;
  try {
    s = await (await fetch("/stats")).json();
  } catch (e) { return; }
  if (s.status === "starting") return;

  // header
  $("source").textContent = s.source;
  $("model-type").textContent = s.model_type;
  $("target-name").textContent = s.target_name;
  $("mlflow").textContent = s.mlflow_enabled ? "on" : "off";
  const running = s.stream && s.stream.running;
  const dot = $("status-dot");
  dot.className = "dot " + (running ? "live" : "paused");
  if (s.stream) $("rate-val").textContent = Math.round(s.stream.rate_per_sec);

  // KPIs
  $("k-samples").textContent = fmt(s.samples_processed);
  $("k-labeled").textContent = fmt(s.labeled_samples);
  $("k-mae").innerHTML = `${(s.performance.rolling_mae || 0).toFixed(1)}<span class="unit"> s</span>`;
  const r2 = (s.performance.rolling_r2 !== undefined ? s.performance.rolling_r2 : s.performance.r2) || 0;
  $("k-r2").textContent = r2.toFixed(3);
  $("k-version").textContent = s.model_version;
  $("k-concept").textContent = s.drift.concept_events;
  $("k-spurious").textContent = s.drift.spurious_events;
  const retrain = s.drift.retrain_recommended;
  const rEl = $("k-retrain");
  rEl.textContent = retrain ? "YES" : "no";
  rEl.style.color = retrain ? "var(--bad)" : "var(--muted)";

  initPredictPanel(s);

  // perf chart
  const hist = s.history || [];
  drawChart($("perf-chart"), [
    { color: "#5b8cff", points: hist.map((d) => ({ x: d.index, y: d.rolling_mae })) },
    { color: "#fbbd23", points: hist.map((d) => ({ x: d.index, y: d.rolling_rmse !== undefined ? d.rolling_rmse : d.rmse })) },
  ], { yZero: true });

  // predicted vs actual
  const rec = s.recent || [];
  drawChart($("pred-chart"), [
    { color: "#36d399", points: rec.map((d) => ({ x: d.index, y: d.actual })) },
    { color: "#5b8cff", points: rec.map((d) => ({ x: d.index, y: d.predicted })) },
  ], { yZero: true });

  // PSI bars
  const psi = s.drift.feature_psi || {};
  const entries = Object.entries(psi).sort((a, b) => b[1] - a[1]);
  const bars = $("psi-bars");
  if (entries.length === 0) {
    bars.innerHTML = '<span class="muted">waiting for reference window…</span>';
  } else {
    bars.innerHTML = entries.map(([f, v]) => {
      const pct = Math.min(100, (v / 0.6) * 100);
      const color = v > 0.2 ? "#f87272" : v > 0.1 ? "#fbbd23" : "#36d399";
      return `<div class="bar-row"><span class="name">${f}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${pct}%;background:${color}"></span></span>
        <span class="num">${v.toFixed(3)}</span></div>`;
    }).join("");
  }

  // drift events
  const evs = s.drift_events || [];
  const ev = $("events");
  if (evs.length === 0) {
    ev.innerHTML = '<span class="muted">none yet</span>';
  } else {
    ev.innerHTML = evs.map((e) => {
      const detail = e.details ? JSON.stringify(e.details) : "";
      return `<div class="event ${e.type}">
        <span class="tag ${e.type}">${e.type}</span>
        @ sample ${fmt(e.index)} · ${e.detector}
        <div class="meta">${detail}${e.action ? " · " + e.action : ""}</div>
      </div>`;
    }).join("");
  }
}

$("btn-pause").onclick = () => post("/stream/pause");
$("btn-resume").onclick = () => post("/stream/resume");
$("rate").oninput = (e) => { $("rate-val").textContent = e.target.value; };
$("rate").onchange = (e) => post(`/stream/rate?rate_per_sec=${e.target.value}`);

setInterval(tick, 1000);
tick();

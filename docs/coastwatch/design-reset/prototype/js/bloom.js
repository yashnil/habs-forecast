/* Bloom Intelligence prototype. URL params: ?station=<id>&var=<variable>&range=3m|1y|3y|all */
(function () {
  const { D, icon, esc, fmtDate, sig, REGIONS, REGION } = CW;
  const { lin, log, timeTicks, MONTH } = CWChart;
  CW.mountChrome("bloom");

  const OBS = D.observations;
  const POLICY = { current: 14, stale: 45 }; // production OBS_FRESHNESS
  const VARS = {
    pDA: { short: "Particulate domoic acid", tab: "Particulate DA", unit: "ng/mL", scale: "log", domain: [1e-4, 100], edges: [0.01, 0.1, 1, 10], kind: "Toxin in seawater" },
    pn_seriata: { short: "Pseudo-nitzschia, seriata group", tab: "P-n seriata", unit: "cells/L", scale: "log", domain: [1, 1e7], edges: [1e3, 1e4, 1e5, 1e6], kind: "Larger cells" },
    pn_delicatissima: { short: "Pseudo-nitzschia, delicatissima group", tab: "P-n delicatissima", unit: "cells/L", scale: "log", domain: [1, 1e7], edges: [1e3, 1e4, 1e5, 1e6], kind: "Smaller cells" },
    chl_extracted: { short: "Chlorophyll-a", tab: "Chlorophyll", unit: "mg/m³", scale: "log", domain: [0.01, 1000], edges: [1, 3, 10, 30], kind: "Algae biomass, not toxin" },
    temp: { short: "Water temperature", tab: "Temperature", unit: "°C", scale: "linear", domain: [5, 30], edges: [12, 14, 16, 18], kind: "At the sampling point" },
  };
  const ORDER = Object.keys(VARS);
  const RANGES = { "3m": 92, "1y": 365, "3y": 1096, all: null };
  const HEAT = ["#e3f1ee", "#a9d9cf", "#5fb8a8", "#1f8a7c", "#0b5a52"]; // measured identity: teal, sequential
  const GAP_DAYS = 21;

  const q = new URLSearchParams(location.search);
  const stations = OBS.stations.slice().sort((a, b) => (b.lat ?? 0) - (a.lat ?? 0));
  const byId = Object.fromEntries(stations.map((s) => [s.station_id, s]));
  const state = {
    station: byId[q.get("station")] ? q.get("station") : "HABs-SantaCruzWharf",
    variable: VARS[q.get("var")] ? q.get("var") : "pDA",
    range: q.get("range") in RANGES ? q.get("range") : "1y",
  };
  const mobile = () => matchMedia("(max-width: 720px)").matches;

  const summary = (s, v) => s.summaries.find((x) => x.variable === v);
  const seriesOf = (s, v) => s.series.find((x) => x.variable === v);
  function samples(s, v) {
    const se = seriesOf(s, v);
    if (!se) return [];
    return s.sample_times.map((t, i) => ({ t: Date.parse(t), i, v: se.values[i], q: se.qualifiers[String(i)] ?? null })).filter((p) => p.v != null);
  }
  function fmtVal(v, unit) {
    if (v == null) return "—";
    if (unit === "cells/L") return `${Math.round(v).toLocaleString("en-US")} ${unit}`;
    if (unit === "°C") return `${v.toFixed(1)} °C`;
    // two significant figures, keeping trailing zeros (0.20, not 0.2)
    return `${v >= 100 ? Math.round(v).toLocaleString("en-US") : v.toPrecision(2)} ${unit}`;
  }
  const fmtAxis = (v) => (v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : v >= 1 ? String(v) : String(Number(v.toPrecision(1))));
  const lastSample = (s) => s.sample_times.at(-1);

  // ---------- official strip ----------
  function renderStrip() {
    const s = byId[state.station];
    const items = CW.noticesForPort(s.nearest_port_code);
    document.getElementById("strip").innerHTML = CW.officialStrip(items) + `
      <button class="m-strip" data-open-official>${icon("shield")}<span><b>${items.length} official notices</b> may apply near ${esc(s.name)}</span><span class="unverified">Not verified</span>${icon("chevron", "icon-s")}</button>`;
  }

  // ---------- station rail ----------
  function stationLine(s) {
    const pda = summary(s, "pDA");
    const f = CW.freshness(lastSample(s), POLICY);
    const toxin = !pda?.last_date
      ? "No particulate DA since 2014"
      : pda.last_qualifier === "reported_zero"
        ? `pDA reported 0 · ${fmtDate(pda.last_date, { month: "short", day: "numeric" })}`
        : CW.daysAgo(pda.last_date) > POLICY.stale
          ? `pDA last measured ${fmtDate(pda.last_date, { month: "short", year: "numeric" })}`
          : `pDA ${fmtVal(pda.last_value, "ng/mL")}`;
    return { f, toxin };
  }
  function renderRail() {
    const groups = REGIONS.map((r) => ({ r, ss: stations.filter((s) => s.region === r.id) })).filter((g) => g.ss.length);
    const s = byId[state.station];
    document.getElementById("rail").innerHTML = `
      <div class="rail-head">
        <h2>Monitoring stations</h2>
        <p class="fine">${stations.length} CalHABMAP shore stations, north to south</p>
        <div class="rail-key"><span class="fresh" data-state="current">≤ 14 days</span><span class="fresh" data-state="stale">15–45</span><span class="fresh" data-state="historical">older</span></div>
      </div>
      <div class="rail-list">${groups
        .map(({ r, ss }) => `<p class="rail-region">${r.label}</p>${ss
          .map((st) => {
            const { f, toxin } = stationLine(st);
            return `<button class="st-row" data-station="${st.station_id}" aria-pressed="${st.station_id === state.station}">
              <span class="st-name">${esc(st.name)}</span>
              <span class="fresh" data-state="${f.state}"><span class="sr-only">${f.label}</span></span>
              <span class="st-sub">${fmtDate(lastSample(st), { month: "short", day: "numeric", year: f.state === "historical" ? "numeric" : undefined })} · ${toxin}</span>
            </button>`;
          })
          .join("")}`)
        .join("")}</div>`;
    document.getElementById("m-picker")?.remove();
  }

  // ---------- sampling-freshness strip ----------
  function samplingStrip(s, w) {
    const days = 120;
    const t1 = CW.NOW, t0 = t1 - days * CW.DAY;
    const x = lin(t0, t1, 0, w);
    const ts = s.sample_times.map(Date.parse).filter((t) => t >= t0);
    const last = Date.parse(lastSample(s));
    const gapX = Math.max(0, x(Math.max(last, t0)));
    const ticks = timeTicks(t0, t1, 5);
    return `<svg viewBox="0 0 ${w} 46" width="${w}" height="46" aria-hidden="true">
      <rect x="${gapX}" y="6" width="${w - gapX}" height="20" fill="#f1ede5" />
      ${ts.map((t) => `<rect x="${x(t) - 1.5}" y="6" width="3" height="20" rx="1" fill="var(--measured)"/>`).join("")}
      <line x1="${w - 1}" x2="${w - 1}" y1="0" y2="32" stroke="var(--ink)" stroke-width="1.5"/>
      ${ticks.filter((k) => x(k.t) < w - 50).map((k) => `<text x="${x(k.t)}" y="42" text-anchor="middle">${k.label}</text>`).join("")}
      <text x="${w - 4}" y="42" text-anchor="end" style="font-weight:600;fill:var(--ink)">today</text>
    </svg>`;
  }

  // ---------- observation chart ----------
  function obsChart(s, v, { w, h, t0, t1, compact = false, id }) {
    const cfg = VARS[v];
    const pts = samples(s, v).filter((p) => p.t >= t0 && p.t <= t1);
    const visits = s.sample_times.map(Date.parse).filter((t) => t >= t0 && t <= t1);
    const m = compact ? { l: 40, r: 8, t: 8, b: 40 } : { l: 56, r: 16, t: 14, b: 58 };
    const zeroLane = h - m.b + (compact ? 14 : 20);
    const y = cfg.scale === "log" ? log(cfg.domain[0], cfg.domain[1], h - m.b, m.t) : lin(cfg.domain[0], cfg.domain[1], h - m.b, m.t);
    const x = lin(t0, t1, m.l, w - m.r);
    const yTicks = cfg.scale === "log" ? Array.from({ length: Math.round(Math.log10(cfg.domain[1] / cfg.domain[0])) + 1 }, (_, i) => cfg.domain[0] * 10 ** i).filter((_, i, a) => !compact || i % 2 === 0 || i === a.length - 1) : [5, 10, 15, 20, 25, 30];
    const xTicks = timeTicks(t0, t1, compact ? 4 : 8);
    const pos = pts.filter((p) => p.v > 0 || cfg.scale === "linear");
    const zeros = pts.filter((p) => p.v === 0 && cfg.scale === "log");
    const yc = (v) => y(Math.max(v, cfg.domain[0]));
    const segs = [];
    let cur = [];
    pos.forEach((p, i) => {
      if (i && p.t - pos[i - 1].t > GAP_DAYS * CW.DAY) { segs.push(cur); cur = []; }
      cur.push(p);
    });
    segs.push(cur);
    const peak = pos.reduce((a, p) => (!a || p.v > a.v ? p : a), null);
    const r = compact ? 2.2 : 3.4;
    const svg = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" data-chart="${id}" role="img" aria-label="${esc(cfg.short)} at ${esc(s.name)}, ${pts.length} measurements in view">
      <g class="grid">${yTicks.map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}"/>`).join("")}</g>
      ${yTicks.map((t) => `<text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${fmtAxis(t)}</text>`).join("")}
      ${xTicks.map((k) => `<line x1="${x(k.t)}" x2="${x(k.t)}" y1="${h - m.b}" y2="${h - m.b + 4}" stroke="var(--line-strong)"/><text x="${x(k.t)}" y="${h - 6}" text-anchor="middle" ${k.major ? 'style="font-weight:600;fill:var(--ink-2)"' : ""}>${k.label}</text>`).join("")}
      ${cfg.scale === "log" ? `<text x="${m.l - 8}" y="${zeroLane + 4}" text-anchor="end">0*</text><line x1="${m.l}" x2="${w - m.r}" y1="${zeroLane}" y2="${zeroLane}" stroke="var(--line)" stroke-dasharray="2 3"/>` : ""}
      ${visits.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${h - m.b + (cfg.scale === "log" ? (compact ? 22 : 30) : 6)}" y2="${h - m.b + (cfg.scale === "log" ? (compact ? 26 : 36) : 12)}" stroke="var(--ink-3)" stroke-opacity=".45"/>`).join("")}
      ${segs.filter((sg) => sg.length > 1).map((sg) => `<path d="${sg.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${yc(p.v).toFixed(1)}`).join("")}" fill="none" stroke="var(--measured)" stroke-width="${compact ? 1.2 : 1.6}" stroke-opacity=".7" stroke-linejoin="round"/>`).join("")}
      ${pos.map((p) => `<circle cx="${x(p.t).toFixed(1)}" cy="${yc(p.v).toFixed(1)}" r="${r}" fill="${p.q === "flag_high" ? "#fff" : "var(--measured)"}" stroke="${p.q === "flag_high" ? "var(--measured)" : "#fff"}" stroke-width="${compact ? 0.8 : 1.2}"/>`).join("")}
      ${zeros.map((p) => `<circle cx="${x(p.t).toFixed(1)}" cy="${zeroLane}" r="${r - 0.4}" fill="#fff" stroke="var(--ink-3)" stroke-width="1.2"/>`).join("")}
      ${!compact && peak ? (() => { const px = x(peak.t), left = px > w * 0.55, tx = left ? px - 8 : px + 8, a = left ? "end" : "start"; return `<g><line x1="${px}" x2="${px}" y1="${yc(peak.v) - 6}" y2="${m.t}" stroke="var(--ink)" stroke-width="1"/><text x="${tx}" y="${m.t + 10}" text-anchor="${a}" style="font-weight:600;fill:var(--ink)">Highest in view: ${fmtVal(peak.v, cfg.unit)}</text><text x="${tx}" y="${m.t + 25}" text-anchor="${a}">${fmtDate(peak.t, { month: "short", day: "numeric", year: "numeric" })}</text></g>`; })() : ""}
      ${!pts.length ? `<text x="${(m.l + w - m.r) / 2}" y="${(m.t + h - m.b) / 2}" text-anchor="middle" style="font:500 14px var(--font-sans);fill:var(--ink-2)">Not measured in this period</text>` : ""}
      <rect class="hit" x="${m.l}" y="0" width="${w - m.l - m.r}" height="${h}" fill="transparent"/>
      <line class="cross" x1="0" x2="0" y1="${m.t}" y2="${h - m.b}" stroke="var(--ink)" stroke-width="1" visibility="hidden"/>
    </svg>`;
    return { svg, pts, visits, x, zeros, m };
  }

  function attachTooltip(wrap, s, v, chart) {
    const svgEl = wrap.querySelector("svg");
    const tip = wrap.querySelector(".tooltip");
    const cross = svgEl.querySelector(".cross");
    const all = s.sample_times.map((t, i) => ({ t: Date.parse(t), i })).filter((p) => chart.visits.includes(p.t));
    const se = seriesOf(s, v);
    svgEl.addEventListener("mousemove", (e) => {
      const rect = svgEl.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * svgEl.viewBox.baseVal.width;
      const near = all.reduce((a, p) => (!a || Math.abs(chart.x(p.t) - px) < Math.abs(chart.x(a.t) - px) ? p : a), null);
      if (!near) return;
      const val = se.values[near.i];
      const qual = se.qualifiers[String(near.i)];
      cross.setAttribute("x1", chart.x(near.t)); cross.setAttribute("x2", chart.x(near.t)); cross.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.innerHTML = `${fmtDate(near.t, { month: "short", day: "numeric", year: "numeric" })}<br><b>${val == null ? "Not measured in this sample" : qual === "reported_zero" ? "Reported 0 (not quantified)" : fmtVal(val, VARS[v].unit)}</b>`;
      const left = (chart.x(near.t) / svgEl.viewBox.baseVal.width) * rect.width;
      tip.style.left = `${Math.min(left + 12, rect.width - 190)}px`;
      tip.style.top = `8px`;
    });
    svgEl.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  }

  // ---------- season heatmap ----------
  function heatmap(s, v, w) {
    const cfg = VARS[v];
    const pts = samples(s, v);
    if (!pts.length) return `<p class="fine">No ${esc(cfg.short.toLowerCase())} measurements at this station since 2014.</p>`;
    const y0 = new Date(Date.parse(s.sample_times[0])).getUTCFullYear();
    const y1 = new Date(CW.NOW).getUTCFullYear();
    const months = mobile();
    const cols = months ? 12 : 53;
    const cells = new Map();
    for (const p of pts) {
      const d = new Date(p.t);
      const c = months ? d.getUTCMonth() : Math.min(52, Math.floor((p.t - Date.UTC(d.getUTCFullYear(), 0, 1)) / (7 * CW.DAY)));
      const k = `${d.getUTCFullYear()}-${c}`;
      const prev = cells.get(k);
      cells.set(k, prev == null ? p.v : Math.max(prev, p.v));
    }
    const lab = 40, top = 18;
    const cw = (w - lab) / cols, ch = months ? 18 : 15;
    const rows = y1 - y0 + 1;
    const cls = (val) => cfg.edges.filter((e) => val >= e).length;
    let out = `<svg viewBox="0 0 ${w} ${top + rows * ch + 4}" width="${w}" height="${top + rows * ch + 4}" role="img" aria-label="Highest ${esc(cfg.short)} per ${months ? "month" : "week"}, ${y0}–${y1}">`;
    for (let mth = 0; mth < 12; mth++) {
      const cx = months ? lab + mth * cw + cw / 2 : lab + ((Date.UTC(2021, mth, 1) - Date.UTC(2021, 0, 1)) / (7 * CW.DAY)) * cw;
      out += `<text x="${cx}" y="11" ${months ? 'text-anchor="middle"' : ""}>${months ? MONTH[mth][0] : MONTH[mth]}</text>`;
    }
    for (let yr = y0; yr <= y1; yr++) {
      const ry = top + (yr - y0) * ch;
      out += `<text x="${lab - 6}" y="${ry + ch - 4}" text-anchor="end">${yr}</text>`;
      for (let c = 0; c < cols; c++) {
        const val = cells.get(`${yr}-${c}`);
        const cx = lab + c * cw;
        if (val == null) { out += `<rect x="${cx + 0.5}" y="${ry + 0.5}" width="${cw - 1}" height="${ch - 1}" fill="none" stroke="#ece8df" stroke-width=".6"/>`; continue; }
        if (val === 0 && cfg.scale === "log") { out += `<rect x="${cx + 0.5}" y="${ry + 0.5}" width="${cw - 1}" height="${ch - 1}" fill="#f2efe8"/><circle cx="${cx + cw / 2}" cy="${ry + ch / 2}" r="1.4" fill="#9aa3ad"/>`; continue; }
        out += `<rect x="${cx + 0.5}" y="${ry + 0.5}" width="${cw - 1}" height="${ch - 1}" rx="1.5" fill="${HEAT[cls(val)]}"/>`;
      }
    }
    return out + "</svg>";
  }
  const heatLegend = (v) => {
    const e = VARS[v].edges, u = VARS[v].unit;
    const labels = [`< ${fmtAxis(e[0])}`, `${fmtAxis(e[0])}–${fmtAxis(e[1])}`, `${fmtAxis(e[1])}–${fmtAxis(e[2])}`, `${fmtAxis(e[2])}–${fmtAxis(e[3])}`, `≥ ${fmtAxis(e[3])}`];
    return `<div class="legend">${labels.map((l, i) => `<span class="k"><i class="swatch" style="background:${HEAT[i]}"></i>${l}</span>`).join("")}<span class="muted">${u}</span>${VARS[v].scale === "log" ? `<span class="k"><i class="swatch" style="background:#f2efe8;box-shadow:inset 0 0 0 3px #f2efe8,inset 0 0 0 5px #9aa3ad"></i>reported 0 only</span>` : ""}<span class="k"><i class="swatch" style="background:#fff;box-shadow:inset 0 0 0 1px #e4e0d7"></i>not sampled</span></div>`;
  };

  // ---------- model chart ----------
  function modelChart(s, w, h, t0, t1) {
    const all = (s.charm?.history?.particulate_domoic ?? []).map((p) => ({ t: Date.parse(p.date), v: p.value, n: p.n }));
    // A run whose neighbourhood had far fewer valid cells than usual (cloud/edge gaps in the model
    // output) is drawn as a hollow marker, not joined into the line, so it cannot read as a collapse.
    const typical = all.map((p) => p.n).sort((a, b) => a - b)[Math.floor(all.length / 2)] ?? 0;
    const partial = all.filter((p) => p.n < 0.6 * typical);
    const hist = all.filter((p) => p.n >= 0.6 * typical);
    const m = { l: 56, r: 16, t: 10, b: 30 };
    const x = lin(t0, t1, m.l, w - m.r), y = lin(0, 1, h - m.b, m.t);
    const kept = hist.length ? hist[0].t : null;
    const segs = [];
    let cur = [];
    hist.forEach((p, i) => { if (i && p.t - hist[i - 1].t > 2.5 * CW.DAY) { segs.push(cur); cur = []; } cur.push(p); });
    segs.push(cur);
    const pathOf = (sg) => sg.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
    modelChart.nPartial = partial.filter((p) => p.t >= t0).length;
    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="C-HARM nowcast probability near ${esc(s.name)}">
      <g class="grid">${[0, 0.5, 1].map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}"/>`).join("")}</g>
      ${[0, 0.5, 1].map((t) => `<text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${t * 100}%</text>`).join("")}
      ${kept && kept > t0 ? `<rect x="${m.l}" y="${m.t}" width="${x(kept) - m.l}" height="${h - m.b - m.t}" fill="url(#hatch)"/>${x(kept) - m.l > 150 ? `<text x="${m.l + 10}" y="${m.t + 18}" style="fill:var(--ink-2)">CoastWatch keeps model history</text><text x="${m.l + 10}" y="${m.t + 33}" style="fill:var(--ink-2)">from ${fmtDate(kept, { month: "short", day: "numeric", year: "numeric" })}</text>` : ""}` : ""}
      <defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#ddd3ef" stroke-width="1.2"/></pattern></defs>
      ${segs.filter((sg) => sg.length > 1).map((sg) => `<path d="${pathOf(sg)}L${x(sg.at(-1).t)},${h - m.b}L${x(sg[0].t)},${h - m.b}Z" fill="var(--model)" opacity=".08"/><path d="${pathOf(sg)}" fill="none" stroke="var(--model)" stroke-width="1.6" stroke-linejoin="round"/>`).join("")}
      ${partial.filter((p) => p.t >= t0).map((p) => `<circle cx="${x(p.t)}" cy="${y(p.v)}" r="3" fill="#fff" stroke="var(--model)" stroke-width="1.2"/>`).join("")}
      ${timeTicks(t0, t1, 8).map((k) => `<text x="${x(k.t)}" y="${h - 8}" text-anchor="middle" ${k.major ? 'style="font-weight:600;fill:var(--ink-2)"' : ""}>${k.label}</text>`).join("")}
    </svg>`;
  }

  // ---------- main ----------
  function render() {
    const s = byId[state.station];
    const cfg = VARS[state.variable];
    const f = CW.freshness(lastSample(s), POLICY);
    const port = CW.portById[s.nearest_port_code];
    const t1 = CW.NOW;
    const first = Date.parse(s.sample_times[0]);
    const t0 = RANGES[state.range] ? t1 - RANGES[state.range] * CW.DAY : first;
    const main = document.getElementById("main");
    const W = Math.max(300, main.clientWidth - (mobile() ? 32 : 0));
    const pda = summary(s, "pDA");
    const cadence = s.summaries.filter((x) => x.n_last_365d).sort((a, b) => b.n_last_365d - a.n_last_365d)[0];

    const readout = (v) => {
      const c = VARS[v], sm = summary(s, v);
      if (!sm?.last_date) return `<button class="readout none" data-var="${v}" aria-pressed="${v === state.variable}"><span class="r-l">${c.short}</span><span class="r-v r-none">Not measured here since 2014</span><span class="r-d">${c.kind}</span></button>`;
      const rf = CW.freshness(sm.last_date, POLICY);
      // An old value never gets headline size: the date leads, the value follows.
      if (rf.state === "historical") return `<button class="readout old" data-var="${v}" aria-pressed="${v === state.variable}"><span class="r-l">${c.short}</span><span class="r-v r-none">Not measured since ${fmtDate(sm.last_date, { month: "short", year: "numeric" })}</span><span class="r-d">${CW.freshChip(rf, "Historical")}<span>last value ${sm.last_qualifier === "reported_zero" ? "reported 0" : fmtVal(sm.last_value, c.unit)}, ${fmtDate(sm.last_date)}</span></span></button>`;
      const val = sm.last_qualifier === "reported_zero" ? `Reported 0` : fmtVal(sm.last_value, c.unit);
      return `<button class="readout" data-var="${v}" aria-pressed="${v === state.variable}">
        <span class="r-l">${c.short}</span>
        <span class="r-v">${/^[\d.,]+ /.test(val) ? val.replace(/ (\S+)$/, '<span class="u"> $1</span>') : val}${sm.last_qualifier === "reported_zero" ? `<small> not quantified</small>` : ""}</span>
        <span class="r-d">${CW.freshChip(rf, fmtDate(sm.last_date, { month: "short", day: "numeric", year: rf.state === "historical" ? "numeric" : undefined }))}<span>${sm.n_last_365d} in 12 mo</span></span>
      </button>`;
    };

    const chartW = mobile() ? W : W - 48;
    const mainChart = obsChart(s, state.variable, { w: chartW, h: mobile() ? 260 : 340, t0, t1, id: "main" });
    const others = ORDER.filter((v) => v !== state.variable);
    const smW = mobile() ? W : Math.floor((W - 48 - 24) / 2);
    const smCharts = others.map((v) => ({ v, c: obsChart(s, v, { w: smW, h: 150, t0, t1, compact: true, id: v }) }));
    const nVisits = mainChart.visits.length;
    const nMeas = mainChart.pts.length;

    main.innerHTML = `
      <button class="m-picker" id="m-picker" data-open-rail>${icon("pin")}<span><b>${esc(s.name)}</b><span>${REGION[s.region]?.label ?? ""} · ${stations.length} stations</span></span>${icon("down", "icon-s")}</button>
      <header class="st-head" data-testid="station-detail" data-station="${s.station_id}">
        <div class="st-title">
          <p class="eyebrow measured">Shore station · ${REGION[s.region]?.label ?? "California"} · CalHABMAP</p>
          <h1>${esc(s.name)}</h1>
          <p class="st-meta"><span class="mono">${s.lat.toFixed(3)}°N ${Math.abs(s.lon).toFixed(3)}°W</span> · ${esc(s.location_code)} · ${s.nearest_port_km} km from ${esc(s.nearest_port_name)} · <a href="map.html?port=${s.nearest_port_code}&region=${s.region}">Open on map</a></p>
        </div>
        <div class="fresh-card" data-state="${f.state}">
          <p class="eyebrow">Latest sample</p>
          <p class="fc-date">${fmtDate(lastSample(s), { month: "long", day: "numeric", year: "numeric" })}</p>
          <p class="fc-age">${CW.freshChip(f)}<span>${CW.agoText(f.days)}</span></p>
          <div class="fc-strip">${samplingStrip(s, mobile() ? W - 34 : 360)}</div>
          <p class="fine">${cadence?.median_interval_days_365d ? `Sampled about every ${Math.round(cadence.median_interval_days_365d)} days (${cadence.n_last_365d} samples in 12 months).` : "No samples in the last 12 months."} Lab results arrive days to weeks after sampling.</p>
        </div>
      </header>
      ${f.state === "historical" ? `<div class="hist-banner">${icon("info")}<p><b>This station's latest sample is from ${fmtDate(lastSample(s))}.</b> Its values do not describe current conditions. No new sample is not the same as no toxin.</p></div>` : ""}

      <section class="readouts" aria-label="Latest measurement of each quantity">${ORDER.map(readout).join("")}</section>

      <section class="card chart-card" aria-labelledby="main-chart-title">
        <div class="cc-head">
          <div>
            <h2 id="main-chart-title">${cfg.short} <span class="muted">${cfg.unit} · ${cfg.scale} scale</span></h2>
            <p class="fine">${nMeas ? `Measured in ${nMeas} of ${nVisits} sampling visits in view${mainChart.zeros.length ? ` · ${mainChart.zeros.length} reported 0` : ""}` : "Not measured in this period. No measurement is not the same as no toxin."}</p>
          </div>
          <div class="seg" role="group" aria-label="Time range">${Object.keys(RANGES).map((r) => `<button data-range="${r}" aria-pressed="${r === state.range}">${{ "3m": "3 mo", "1y": "1 yr", "3y": "3 yr", all: "Since 2014" }[r]}</button>`).join("")}</div>
        </div>
        <div class="var-tabs" role="tablist" aria-label="Measurement">${ORDER.map((v) => `<button role="tab" data-var="${v}" aria-selected="${v === state.variable}">${VARS[v].tab}</button>`).join("")}</div>
        <div class="chart main-chart" style="position:relative">${mainChart.svg}<div class="tooltip" hidden></div></div>
        <div class="legend cc-legend">
          <span class="k"><i class="swatch dot" style="background:var(--measured)"></i>measured</span>
          ${cfg.scale === "log" ? `<span class="k"><i class="swatch ring"></i>0* reported 0 (not quantified, not absent)</span>` : ""}
          <span class="k"><i class="swatch" style="width:2px;height:10px;background:var(--ink-3)"></i>sampling visit</span>
          <span class="k muted">Line breaks where samples are more than ${GAP_DAYS} days apart. Gaps are not zeros.</span>
        </div>
      </section>

      <section aria-labelledby="sm-title">
        <div class="section-head"><h2 id="sm-title">Other measurements, same period</h2><p>Each panel has its own axis. Select one to make it the main chart.</p></div>
        <div class="small-multiples">${smCharts
          .map(({ v, c }) => {
            const sm = summary(s, v);
            return `<button class="card sm" data-var="${v}">
              <span class="sm-head"><span class="sm-title">${VARS[v].short}</span><span class="sm-last">${sm?.last_date ? (sm.last_qualifier === "reported_zero" ? "reported 0" : fmtVal(sm.last_value, VARS[v].unit)) : "not measured"}</span></span>
              ${c.pts.length ? `<span class="chart">${c.svg}</span>` : `<span class="sm-empty">${sm?.last_date ? `Not measured in this period. Last measured ${fmtDate(sm.last_date, { month: "short", year: "numeric" })}.` : "Not measured at this station since 2014."} No measurement is not the same as none present.</span>`}
              ${c.pts.length ? `<span class="fine">${c.pts.length} measured in view · ${VARS[v].unit}, ${VARS[v].scale}</span>` : ""}
            </button>`;
          })
          .join("")}</div>
        <p class="fine sm-note">Pseudo-nitzschia counts are cells of a size group; not every Pseudo-nitzschia produces toxin. Chlorophyll measures algae biomass, not toxin.</p>
      </section>

      <section class="card heat-card" aria-labelledby="heat-title">
        <div class="section-head"><div><p class="eyebrow measured">Historical context</p><h2 id="heat-title">Every ${mobile() ? "month" : "week"} since ${new Date(first).getUTCFullYear()}: ${cfg.short.toLowerCase()}</h2></div><p>Highest value each ${mobile() ? "month" : "week"}. Decade bins, not risk levels.</p></div>
        <div class="chart">${heatmap(s, state.variable, mobile() ? W - 34 : W - 48)}</div>
        ${heatLegend(state.variable)}
      </section>

      <section class="model-band" data-testid="model-section" aria-labelledby="model-title">
        <div class="mb-head">
          <div>
            <p class="eyebrow model">Agency forecast · model, not a measurement</p>
            <h2 id="model-title">What NOAA's C-HARM model estimated near this station</h2>
            <p class="mb-lede">Median probability that particulate domoic acid exceeds 500 ng/L in model cells within ${s.charm?.radius_km ?? 15} km (nearest cell ${s.charm?.nearest_cell_km ?? "—"} km away). A probability is a different quantity from the measurements above, so CoastWatch never compares them numerically.</p>
          </div>
          <span class="chip chip-model">C-HARM v3.1 · NOAA</span>
        </div>
        ${s.charm?.history?.particulate_domoic?.length ? `<div class="chart">${modelChart(s, mobile() ? W : W - 48, mobile() ? 170 : 190, Math.max(t0, CW.NOW - 365 * CW.DAY), t1)}</div>
          <div class="legend"><span class="k"><i class="swatch" style="height:2px;background:var(--model)"></i>daily nowcast median</span>${modelChart.nPartial ? `<span class="k"><i class="swatch ring" style="box-shadow:inset 0 0 0 1.5px var(--model)"></i>partial coverage (fewer than 60% of the usual model cells had values)</span>` : ""}<span class="k muted">Day-to-day swings are the model's own output and are not smoothed.</span></div>
          <p class="fine">Same time axis as the main chart${RANGES[state.range] && RANGES[state.range] <= 365 ? "" : " (last 12 months)"}. ${s.charm.history.particulate_domoic.length} daily nowcasts. Days without a model run are left blank. A probability is not a closure decision, and a low value does not mean it is safe to fish or harvest.</p>` : `<p class="mb-empty" data-testid="model-unavailable">Model history is unavailable for this station. That says nothing about bloom conditions.</p>`}
      </section>

      <section class="about" aria-labelledby="about-title">
        <h2 id="about-title" class="sr-only">About these data</h2>
        <div class="about-grid">
          <div><p class="eyebrow">What a value means</p><p>${esc(OBS.caveats[0])}</p></div>
          <div><p class="eyebrow">Blanks and zeros</p><p>${esc(OBS.caveats[2])} ${esc(OBS.caveats[3])}</p></div>
          <div><p class="eyebrow">Seawater is not seafood</p><p>${esc(OBS.caveats[1])}</p></div>
        </div>
        <details class="disclosure"><summary>What each quantity is</summary><div class="body"><ul>${OBS.variables.map((v) => `<li><b>${esc(v.label)}</b> (${esc(v.units)}). ${esc(v.method)} ${esc(v.detection_limit_note ?? "")}</li>`).join("")}</ul></div></details>
        <details class="disclosure"><summary>Quality checks for this station</summary><div class="body"><ul>${s.qc.map((c) => `<li>${c.passed ? "Passed" : "Flagged"} — ${esc(c.detail)}</li>`).join("")}</ul></div></details>
        <details class="disclosure"><summary>How CoastWatch processes these data</summary><div class="body"><ul>${Object.values(OBS.method).map((m) => `<li>${esc(m)}</li>`).join("")}</ul></div></details>
        <p class="fine source">Source: <a href="${s.source_url}">${esc(s.station_id)} on SCCOOS ERDDAP</a> · ${esc(OBS.program)} · retrieved ${CW.fmtTime(s.retrieved_at)}. ${esc(OBS.provenance?.[0]?.license ?? "")} These pages have not yet been reviewed by an independent HAB scientist.</p>
      </section>`;

    attachTooltip(main.querySelector(".main-chart"), s, state.variable, mainChart);
  }

  function sync() {
    const u = new URLSearchParams({ station: state.station });
    if (state.variable !== "pDA") u.set("var", state.variable);
    if (state.range !== "1y") u.set("range", state.range);
    history.replaceState(null, "", "?" + u.toString());
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-station],[data-var],[data-range],[data-open-rail]");
    if (!t) return;
    if (t.dataset.station) { state.station = t.dataset.station; document.body.classList.remove("rail-open"); renderStrip(); renderRail(); render(); scrollTo(0, 0); }
    else if (t.dataset.var) { state.variable = t.dataset.var; renderRail(); render(); }
    else if (t.dataset.range) { state.range = t.dataset.range; render(); }
    else if (t.dataset.openRail !== undefined) document.body.classList.toggle("rail-open");
    sync();
  });
  renderStrip();
  renderRail();
  render();
  let rw = innerWidth;
  addEventListener("resize", () => { if (innerWidth !== rw) { rw = innerWidth; render(); } });
})();

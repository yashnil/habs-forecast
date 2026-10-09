/* Ocean Map prototype. URL params: ?region=<id>&port=<code>&var=<variable>&lead=<0-3>
   The map is the page. One compact navigation card (official notices + places), one dock
   (quantity, day, legend), and a port inspector that only exists while a port is selected. */
(function () {
  const { D, icon, esc, fmtDay, fmtDate, pct, sig, REGIONS, REGION } = CW;
  const { pClass, sparkline, P_CLASSES } = CWChart;
  CW.mountChrome("map");

  const VARS = [
    { id: "particulate_domoic", short: "Particulate DA", long: "Particulate domoic acid" },
    { id: "pseudo_nitzschia", short: "Pseudo-nitzschia", long: "Pseudo-nitzschia bloom" },
    { id: "cellular_domoic", short: "Cellular DA", long: "Cellular domoic acid" },
  ];
  const charm = D.layers.filter((l) => l.group_id === "charm");
  const layerFor = (v, lead) => charm.find((l) => l.variable === v && l.time.lead_days === lead);
  const run = D.forecast_runs.find((r) => r.group_id === "charm");
  const charmFresh = CW.freshness(run.issued_date, { current: 1, stale: 7 });
  const today = CW.pacificDay(CW.NOW);

  const q = new URLSearchParams(location.search);
  const state = {
    variable: q.get("var") || "particulate_domoic",
    lead: Number(q.get("lead") ?? charm.find((l) => l.time.valid_date === today)?.time.lead_days ?? 1),
    region: q.get("region") || null,
    port: q.get("port") ? Number(q.get("port")) : null,
    search: "",
  };
  if (state.port && !state.region) state.region = CW.portById[state.port]?.region ?? null;
  const ports = D.port_intel.ports;
  const portLead = (p, lead = state.lead, v = state.variable) => p.charm?.leads.find((l) => l.lead_days === lead)?.variables[v];
  const isMobile = () => matchMedia("(max-width: 720px)").matches;
  const varInfo = () => VARS.find((v) => v.id === state.variable);

  // Region views: tight on the coast that matters. Monterey Bay runs Año Nuevo to Point Sur.
  const VIEW = {
    north_coast: [[-124.6, 40.0], [-123.75, 41.95]],
    mendocino_sonoma: [[-124.05, 38.25], [-122.85, 39.9]],
    sf_bay_farallones: [[-123.15, 37.35], [-122.3, 38.05]],
    monterey_bay: [[-122.38, 36.48], [-121.66, 37.13]],
    central_coast: [[-121.4, 34.42], [-120.45, 35.95]],
    southern_california: [[-120.55, 32.55], [-117.1, 34.5]],
  };
  const STATEWIDE = [[-125.9, 32.45], [-117.1, 42.05]];

  // ---------- map ----------
  const graticule = { type: "FeatureCollection", features: [] };
  for (let lat = 32; lat <= 42; lat++) graticule.features.push({ type: "Feature", properties: { label: `${lat}°N` }, geometry: { type: "LineString", coordinates: [[-130, lat], [-114, lat]] } });
  for (let lon = -126; lon <= -116; lon += 2) graticule.features.push({ type: "Feature", properties: { label: `${-lon}°W` }, geometry: { type: "LineString", coordinates: [[lon, 30], [lon, 44]] } });
  const stations = D.observations.stations.filter((s) => s.lon != null).map((s) => ({ type: "Feature", properties: { id: s.station_id, name: s.name }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } }));
  const FONT = ["Noto Sans Regular"];
  // Land is a lighter slate than the sea so the coastline reads before the forecast does.
  const SEA = "#0b1d33", LAND = "#2a3646", COAST = "#d3dfeb";
  const style = {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      omt: { type: "vector", url: "https://tiles.openfreemap.org/planet", attribution: "OpenFreeMap · © OpenMapTiles · © OpenStreetMap contributors" },
      grat: { type: "geojson", data: graticule },
      official: { type: "geojson", data: D.official.geometry },
      ports: { type: "geojson", data: D.ports },
      stations: { type: "geojson", data: { type: "FeatureCollection", features: stations } },
    },
    layers: [
      { id: "land", type: "background", paint: { "background-color": LAND } },
      { id: "landcover", type: "fill", source: "omt", "source-layer": "landcover", filter: ["in", ["get", "class"], ["literal", ["wood", "forest"]]], paint: { "fill-color": "#2f3d4f", "fill-opacity": 0.8 } },
      { id: "water", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-color": SEA } },
      // Hatching marks water with no model value. The forecast raster is opaque, so the
      // hatch only shows where there is no value: never confused with a low probability.
      { id: "water-nodata", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-pattern": "hatch" } },
      { id: "grat", type: "line", source: "grat", paint: { "line-color": "#8aa4c2", "line-opacity": 0.14, "line-width": 0.6 } },
      { id: "roads", type: "line", source: "omt", "source-layer": "transportation", minzoom: 7, filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk"]]], paint: { "line-color": "#3a4859", "line-width": 0.8 } },
      { id: "state", type: "line", source: "omt", "source-layer": "boundary", filter: ["all", ["<=", ["coalesce", ["get", "admin_level"], 99], 4], ["!=", ["coalesce", ["get", "maritime"], 0], 1]], paint: { "line-color": "#5a6b80", "line-width": 0.8, "line-dasharray": [3, 2] } },
      { id: "coastline", type: "line", source: "omt", "source-layer": "water", paint: { "line-color": COAST, "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 9, 1.2, 12, 1.8] } },
      { id: "official-fill", type: "fill", source: "official", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#f6bb5c", "fill-opacity": 0.08 } },
      { id: "official-poly-casing", type: "line", source: "official", filter: ["!=", ["get", "kind"], "lat_limit"], paint: { "line-color": "#06111e", "line-width": 3.4, "line-opacity": 0.55 } },
      { id: "official-poly", type: "line", source: "official", filter: ["!=", ["get", "kind"], "lat_limit"], paint: { "line-color": "#f6bb5c", "line-width": 1.4, "line-dasharray": [2, 1.5] } },
      { id: "official-lat-casing", type: "line", source: "official", filter: ["==", ["get", "kind"], "lat_limit"], paint: { "line-color": "#06111e", "line-width": 4, "line-opacity": 0.6 } },
      { id: "official-lat", type: "line", source: "official", filter: ["==", ["get", "kind"], "lat_limit"], paint: { "line-color": "#f6bb5c", "line-width": 2 } },
      { id: "official-lat-label", type: "symbol", source: "official", filter: ["==", ["get", "kind"], "lat_limit"], minzoom: 7.4, layout: { "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": FONT, "text-size": 11, "text-offset": [0, -0.8] }, paint: { "text-color": "#f6bb5c", "text-halo-color": "#06111e", "text-halo-width": 1.6 } },
      { id: "grat-label", type: "symbol", source: "grat", maxzoom: 7.5, layout: { "symbol-placement": "line", "symbol-spacing": 2000, "text-field": ["get", "label"], "text-font": FONT, "text-size": 10 }, paint: { "text-color": "#9db2cb", "text-opacity": 0.75, "text-halo-color": SEA, "text-halo-width": 1 } },
      { id: "city", type: "symbol", source: "omt", "source-layer": "place", minzoom: 7.2, filter: ["in", ["get", "class"], ["literal", ["city", "town"]]], layout: { "text-field": ["get", "name"], "text-font": FONT, "text-size": ["interpolate", ["linear"], ["zoom"], 7, 10.5, 10, 12], "symbol-sort-key": ["get", "rank"] }, paint: { "text-color": "#aab8c8", "text-halo-color": LAND, "text-halo-width": 1.2 } },
      { id: "stations", type: "circle", source: "stations", minzoom: 6.5, paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 6.5, 2.6, 10, 4.5], "circle-color": "#3fc1b0", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.2 } },
      { id: "ports-halo", type: "circle", source: "ports", paint: { "circle-radius": ["case", ["==", ["get", "port_code"], -1], 13, 0], "circle-color": "#ffffff", "circle-opacity": 0.22, "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5 } },
      { id: "ports", type: "circle", source: "ports", paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 3.2, 10, 6], "circle-color": "#ffffff", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.6 } },
      { id: "ports-label", type: "symbol", source: "ports", minzoom: 7.4, layout: { "text-field": ["get", "display_name"], "text-font": FONT, "text-size": ["interpolate", ["linear"], ["zoom"], 7.4, 11.5, 10, 13.5], "text-anchor": "left", "text-offset": [0.9, 0], "text-optional": true }, paint: { "text-color": "#ffffff", "text-halo-color": "#06111e", "text-halo-width": 1.8 } },
    ],
  };
  const map = new maplibregl.Map({ container: "map", style, bounds: STATEWIDE, attributionControl: { compact: false }, fadeDuration: 0, preserveDrawingBuffer: true });
  window.__cwMap = map; // screenshot script hook
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric", maxWidth: 90 }), "bottom-right");
  map.on("styleimagemissing", (e) => {
    if (e.id !== "hatch") return;
    const n = 8, c = document.createElement("canvas");
    c.width = c.height = n;
    const g = c.getContext("2d");
    g.strokeStyle = "rgba(150,178,210,0.26)";
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, n); g.lineTo(n, 0); g.moveTo(-1, 1); g.lineTo(1, -1); g.moveTo(n - 1, n + 1); g.lineTo(n + 1, n - 1); g.stroke();
    map.addImage("hatch", g.getImageData(0, 0, n, n), { pixelRatio: 1 });
  });

  // Opaque raster: the legend swatches are exactly the colours on the map.
  function setRaster() {
    const l = layerFor(state.variable, state.lead);
    if (map.getLayer("forecast")) map.removeLayer("forecast");
    if (map.getSource("forecast")) map.removeSource("forecast");
    map.addSource("forecast", { type: "image", url: l.raster.url, coordinates: l.raster.corners });
    map.addLayer({ id: "forecast", type: "raster", source: "forecast", paint: { "raster-opacity": 1, "raster-resampling": "nearest", "raster-fade-duration": 0 } }, "grat");
    loadGrid(l);
  }

  // ---------- exact values from the published grid (same logic as production lib/grid.ts) ----------
  const grids = new Map();
  function loadGrid(l) {
    if (grids.has(l.layer_id)) return grids.get(l.layer_id);
    const p = fetch(l.raster.grid.url)
      .then((r) => new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer())
      .then((b) => new Uint16Array(b));
    grids.set(l.layer_id, p);
    return p;
  }
  function sampleAt(g, codes, lat, lon) {
    const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5), col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
    if (row < 0 || row >= g.height || col < 0 || col >= g.width) return { kind: "outside" };
    const val = (r, c) => { const k = codes[r * g.width + c]; return k === g.nodata ? null : k * g.scale_factor + g.add_offset; };
    const v = val(row, col);
    if (v != null) return { kind: "value", value: v };
    let best = null;
    for (let dr = -3; dr <= 3; dr++) for (let dc = -3; dc <= 3; dc++) {
      const r = row + dr, c = col + dc;
      if (r < 0 || r >= g.height || c < 0 || c >= g.width) continue;
      const x = val(r, c);
      if (x == null) continue;
      const dy = (g.lat_first + r * g.lat_step - lat) * 111.32, dx = (g.lon_first + c * g.lon_step - lon) * 111.32 * Math.cos((lat * Math.PI) / 180);
      const d = Math.hypot(dx, dy);
      if (!best || d < best.d) best = { value: x, d };
    }
    return best ? { kind: "nearest", value: best.value, km: best.d } : { kind: "none" };
  }
  const exact = (v) => `${(v * 100).toFixed(1)}%`;
  const readout = document.getElementById("readout");
  async function showReadout(e, pinned) {
    const l = layerFor(state.variable, state.lead);
    const codes = await loadGrid(l);
    const { lat, lng } = e.lngLat;
    const s = sampleAt(l.raster.grid, codes, lat, lng);
    const where = `<span class="mono">${lat.toFixed(2)}°N ${Math.abs(lng).toFixed(2)}°W</span>`;
    const head = `<span class="ro-q">${esc(varInfo().short)} · ${fmtDay(l.time.valid_date)}</span>`;
    readout.innerHTML =
      s.kind === "value"
        ? `${head}<span class="ro-v"><i style="background:${pClass(s.value)}"></i>${exact(s.value)}</span><span class="ro-m">Model cell at ${where}</span>`
        : s.kind === "nearest"
          ? `${head}<span class="ro-none">No model value at this point</span><span class="ro-m">Nearest cell ${s.km.toFixed(0)} km away: <b>${exact(s.value)}</b></span>`
          : `${head}<span class="ro-none"><i class="hatch"></i>No model value</span><span class="ro-m">${s.kind === "outside" ? "Outside the model area" : "Land or masked nearshore water"} · ${where}</span>`;
    readout.hidden = false;
    readout.classList.toggle("pinned", !!pinned);
    if (!pinned) {
      const r = map.getContainer().getBoundingClientRect();
      const x = Math.min(e.point.x + 16, r.width - 230), y = Math.max(e.point.y - 72, 8);
      readout.style.transform = `translate(${x}px, ${y}px)`;
    } else readout.style.transform = "";
  }

  // Frame the coast inside whatever the panels leave visible (measured, not assumed).
  function padding() {
    const box = (id) => document.getElementById(id).getBoundingClientRect();
    if (isMobile()) return { top: 72, left: 12, right: state.region ? 84 : 20, bottom: (state.port ? box("inspector").height : box("dock").height) + 16 };
    const right = state.port ? box("inspector").width + 48 : 72;
    if (!state.region) return { top: 32, left: box("nav").right + 32, right, bottom: 32 };
    return { top: 32, left: box("nav").right + 32, right, bottom: box("dock").height + 40 };
  }
  function frame(animate = true) {
    const b = state.region ? VIEW[state.region] : STATEWIDE;
    map.fitBounds(b, { padding: padding(), duration: animate && !matchMedia("(prefers-reduced-motion: reduce)").matches ? 600 : 0 });
  }

  map.on("load", () => {
    setRaster();
    if (isMobile()) map.setLayoutProperty("grat-label", "visibility", "none");
    frame(false);
    haloPort();
    map.on("click", "ports", (e) => { e.preventDefault(); selectPort(e.features[0].properties.port_code); });
    map.on("mouseenter", "ports", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "ports", () => (map.getCanvas().style.cursor = ""));
    map.on("mousemove", (e) => { if (!isMobile() && !map.queryRenderedFeatures(e.point, { layers: ["ports"] }).length) showReadout(e); else if (!isMobile()) readout.hidden = true; });
    map.getCanvas().addEventListener("mouseleave", () => { if (!isMobile()) readout.hidden = true; });
    map.on("click", (e) => { if (isMobile() && !e.defaultPrevented) showReadout(e, true); });
    map.on("movestart", () => { if (isMobile()) readout.hidden = true; });
    map.once("idle", () => document.body.setAttribute("data-ready", "1"));
  });
  const haloPort = () => map.getLayer("ports-halo") && map.setPaintProperty("ports-halo", "circle-radius", ["case", ["==", ["get", "port_code"], state.port ?? -1], 13, 0]);

  // ---------- navigation card: official first, then places ----------
  const relText = { statewide: "Statewide", port_latitude_within_stated_range: "Port within the notice's latitudes", same_county: "Same county", named_area_nearby: "Named area nearby" };
  function regionRange(r) {
    const vals = ports.filter((p) => p.region === r.id).map((p) => portLead(p)?.median).filter((v) => v != null);
    return vals.length ? `${pct(Math.min(...vals))}–${pct(Math.max(...vals))}` : "—";
  }
  const portRow = (p) => {
    const v = portLead(p);
    return `<button class="port-row" data-port="${p.port_code}" aria-pressed="${state.port === p.port_code}">
      <span class="pn">${esc(p.display_name)}</span>
      <span class="bar" aria-hidden="true"><i style="width:${(v?.median ?? 0) * 100}%;background:${v ? pClass(v.median) : "transparent"}"></i></span>
      <span class="val">${pct(v?.median)}</span></button>`;
  };
  function renderNav() {
    const el = document.getElementById("nav");
    const scope = state.region ? CW.noticesForRegion(state.region) : CW.records.map((record) => ({ record }));
    const where = state.region ? `may apply in ${REGION[state.region].label}` : "active in California";
    const offRow = `<button class="nav-off" data-open-official data-testid="official-summary">
        ${icon("shield")}<span class="t"><b>${scope.length} official notice${scope.length === 1 ? "" : "s"}</b> ${where}</span>
        <span class="unverified">Not verified</span>${icon("chevron", "icon-s")}</button>`;
    const search = `<label class="nav-search">${icon("search", "icon-s")}<input type="search" placeholder="Find a port" value="${esc(state.search)}" aria-label="Find a port" /></label>`;
    let body;
    if (state.search) {
      const hits = ports.filter((p) => p.display_name.toLowerCase().includes(state.search.toLowerCase()));
      body = `<div class="nav-list">${hits.length ? hits.map(portRow).join("") : `<p class="fine pad">No port named “${esc(state.search)}”.</p>`}</div>`;
    } else if (state.port) {
      // A port is open: the card shrinks to a breadcrumb; the inspector carries the detail.
      const p = CW.portById[state.port];
      body = `<button class="crumb" data-region="${p.region}">${icon("chevron", "icon-s back")}${REGION[p.region].label}<span class="fine">${ports.filter((x) => x.region === p.region).length} ports</span></button>`;
    } else if (state.region) {
      body = `<button class="crumb" data-region="">${icon("chevron", "icon-s back")}All California</button>
        <h2 class="nav-title">${REGION[state.region].label}</h2>
        <div class="nav-list">${ports.filter((p) => p.region === state.region).map(portRow).join("")}</div>
        <p class="fine pad">Median of model cells within 15 km of each port, not conditions at the dock.</p>`;
    } else {
      body = `<p class="nav-label">Coast, north to south</p>
        <div class="nav-list">${REGIONS.map((r) => `<button class="region" data-region="${r.id}"><span class="name">${r.label}</span><span class="range">${regionRange(r)}</span>${icon("chevron", "icon-s")}</button>`).join("")}</div>
        <p class="fine pad">Range of port medians, ${fmtDay(layerFor(state.variable, state.lead).time.valid_date)}.</p>`;
    }
    el.innerHTML = `${state.port ? "" : offRow}${search}${body}${isMobile() ? `<button class="nav-close" data-close-nav aria-label="Close">${icon("close")}</button>` : ""}`;
    el.dataset.mode = state.search ? "search" : state.port ? "port" : state.region ? "region" : "state";
  }

  // ---------- dock: which quantity, which day, how to read it ----------
  function renderDock() {
    const L = layerFor(state.variable, state.lead);
    const leads = charm.filter((l) => l.variable === state.variable).sort((a, b) => a.time.lead_days - b.time.lead_days);
    const scope = state.region ? CW.noticesForRegion(state.region) : CW.records;
    const dayLabel = (l) => (l.time.valid_date === today ? "Today" : `${fmtDate(l.time.valid_date, { weekday: "short" })} ${fmtDate(l.time.valid_date, { day: "numeric" })}`);
    document.getElementById("mplace").innerHTML = `${icon("search", "icon-s")}<span>${state.port ? esc(CW.portById[state.port].display_name) : state.region ? REGION[state.region].label : "All California"}</span>${icon("down", "icon-s")}`;
    document.getElementById("dock").innerHTML = `
      <button class="m-official" data-open-official>${icon("shield", "icon-s")}<span><b>${scope.length}</b> official notice${scope.length === 1 ? "" : "s"} ${state.region ? "may apply here" : "in California"}</span><span class="unverified">Not verified</span>${icon("chevron", "icon-s")}</button>
      <div class="dock-row">
        <label class="var-select"><span class="sr-only">Forecast quantity</span><select data-var-select>${VARS.map((v) => `<option value="${v.id}" ${v.id === state.variable ? "selected" : ""}>${v.short}</option>`).join("")}</select>${icon("down", "icon-s")}</label>
        <div class="seg var-seg" role="group" aria-label="Forecast quantity">${VARS.map((v) => `<button data-var="${v.id}" aria-pressed="${v.id === state.variable}">${v.short}</button>`).join("")}</div>
        <div class="seg day-seg" role="group" aria-label="Forecast day">${leads.map((l) => `<button data-lead="${l.time.lead_days}" aria-pressed="${l.time.lead_days === state.lead}" title="${l.time.lead_days === 0 ? "Nowcast" : `Forecast +${l.time.lead_days} day${l.time.lead_days > 1 ? "s" : ""}`}">${dayLabel(l)}</button>`).join("")}</div>
      </div>
      <div class="legend-block">
        <p class="legend-title">${esc(L.threshold_text)} <span>· ${fmtDay(L.time.valid_date)}, ${L.time.lead_days === 0 ? "nowcast" : `forecast +${L.time.lead_days} d`}</span></p>
        <div class="ramp" aria-hidden="true">${P_CLASSES.map((c) => `<i style="background:${c}"></i>`).join("")}</div>
        <div class="ramp-labels" aria-hidden="true">${[0, 20, 40, 60, 80, 100].map((n) => `<span style="left:${n}%">${n}${n === 100 ? "%" : ""}</span>`).join("")}</div>
        <p class="legend-note"><span class="nodata"><i class="hatch"></i>No model value</span><span class="m-chips"><span class="chip chip-model">Model</span>${CW.freshChip(charmFresh)}</span><span class="bands"><span class="d-long">10-point colour steps for display, not risk levels. Point at the water for the exact value.</span><span class="d-short">Display steps, not risk levels. Tap water for exact value.</span></span></p>
      </div>
      <div class="dock-meta"><span class="chip chip-model">Model</span><span class="issued">C-HARM v3.1 · NOAA · issued ${fmtDate(run.issued_date, { month: "short", day: "numeric" })}</span>${CW.freshChip(charmFresh)}</div>`;
  }

  // ---------- inspector: opens only for a selected port; official always first ----------
  function renderInspector() {
    const el = document.getElementById("inspector");
    document.body.classList.toggle("inspecting", !!state.port);
    if (!state.port) { el.hidden = true; return; }
    const p = CW.portById[state.port];
    const L = layerFor(state.variable, state.lead);
    const cur = portLead(p);
    const notices = CW.noticesForPort(p.port_code);
    const st = D.observations.stations.filter((s) => s.nearest_port_code === p.port_code).sort((a, b) => a.nearest_port_km - b.nearest_port_km)[0];
    const pda = st?.summaries.find((x) => x.variable === "pDA");
    const hist = (p.charm.history[state.variable] || []).map((h) => ({ t: Date.parse(h.date), v: h.value }));
    const chl = p.chlorophyll;
    el.hidden = false;
    el.innerHTML = `
      <div class="ins-head">
        <p class="eyebrow">Port · ${REGION[p.region].label}</p>
        <h2>${esc(p.display_name)}</h2>
        <p class="sub">${esc(p.county)} County · <span class="mono">${p.lat.toFixed(3)}°N ${Math.abs(p.lon).toFixed(3)}°W</span></p>
        <button class="ins-close" data-close-port aria-label="Close port">${icon("close")}</button>
      </div>
      <section class="ins-sec official" data-testid="inspector-official">
        <div class="ins-sec-head"><span class="eyebrow official">${icon("shield", "icon-s")}Official · ${notices.length} may apply</span>${CW.verificationLine()}</div>
        ${notices.map(({ record: r, rel }) => `<button class="notice-row" data-open-official data-focus="${r.id}">${CW.agencyChip(r.agency)}<span><span class="t">${esc(r.title)}</span><span class="why">${relText[rel.relation] ?? rel.relation}</span></span>${icon("chevron", "icon-s")}</button>`).join("")}
        <p class="fine off-note">Agency notices decide what is open. This page does not.</p>
      </section>
      <section class="ins-sec" data-testid="inspector-model">
        <div class="ins-sec-head"><span class="eyebrow model">Model · C-HARM</span>${CW.freshChip(charmFresh, `${charmFresh.label} · issued ${fmtDate(run.issued_date, { month: "short", day: "numeric" })}`)}</div>
        <p class="q">${esc(L.threshold_text)}</p>
        <div class="big-prob"><span class="n">${pct(cur?.median)}</span><span class="d">median of ${cur?.n ?? 0} model cells within 15 km · ${fmtDay(L.time.valid_date)}</span></div>
        <div class="leads" role="group" aria-label="Forecast days">${p.charm.leads.map((l) => { const x = l.variables[state.variable]; return `<button class="lead" data-lead="${l.lead_days}" aria-pressed="${l.lead_days === state.lead}" title="Cells range ${pct(x?.min)}–${pct(x?.max)}"><span class="day">${l.lead_days === 0 ? "Nowcast" : fmtDate(l.valid_date, { weekday: "short" }) + " " + fmtDate(l.valid_date, { day: "numeric" })}</span><span class="v">${pct(x?.median)}</span><span class="rng"><i style="left:${x.min * 100}%;width:${(x.max - x.min) * 100}%;background:${pClass(x.median)}"></i><b style="left:calc(${x.median * 100}% - 1px)"></b></span></button>`; }).join("")}</div>
        <p class="fine">Bar: lowest to highest nearby cell. Tick: median.</p>
        <div class="spark-wrap">
          <div class="ins-sec-head" style="margin:12px 0 2px"><span class="fine">Last 30 days of nowcasts (median)</span><span class="fine">${hist.length} runs</span></div>
          ${sparkline(hist, { w: 320, h: 56, t0: CW.NOW - 31 * CW.DAY, t1: CW.NOW })}
        </div>
        <p class="caveat"><b>A probability for nearby water, not a measurement and not a closure decision.</b> A low value does not mean it is safe to fish or harvest.</p>
      </section>
      ${st ? `<section class="ins-sec">
        <div class="ins-sec-head"><span class="eyebrow measured">Measured nearby · CalHABMAP</span>${CW.freshChip(CW.freshness(pda.last_date, { current: 14, stale: 45 }), CW.agoText(CW.daysAgo(pda.last_date)))}</div>
        <div class="kv"><span class="k">${esc(st.name)}, ${st.nearest_port_km} km</span><span class="v"></span>
          <span class="k">Particulate domoic acid, ${fmtDate(pda.last_date, { month: "short", day: "numeric" })}</span><span class="v">${pda.last_qualifier === "reported_zero" ? "reported 0" : sig(pda.last_value) + " ng/mL"}</span></div>
        <p style="margin-top:10px"><a class="link-arrow" href="bloom.html?station=${st.station_id}">Open in Bloom Intelligence${icon("chevron", "icon-s")}</a></p>
      </section>` : ""}
      ${chl?.latest ? `<section class="ins-sec">
        <div class="ins-sec-head"><span class="eyebrow">Satellite · NOAA VIIRS</span><span class="fine">8 days centred ${fmtDate(chl.latest_center_date, { month: "short", day: "numeric" })}</span></div>
        <div class="kv"><span class="k">Chlorophyll-a, median within 15 km</span><span class="v">${sig(chl.latest.median)} mg/m³</span></div>
        <p class="caveat">Algae biomass, not toxin. ${Math.round(chl.latest_valid_fraction * 100)}% of nearby pixels were cloud-free.</p>
      </section>` : ""}
      <section class="ins-sec"><p class="fine">${esc(p.caveats[0])}</p></section>`;
  }

  function render() {
    renderNav(); renderDock(); renderInspector();
    const h = document.getElementById("dock").getBoundingClientRect().height;
    document.documentElement.style.setProperty("--dock-h", `${Math.round(h)}px`);
    document.documentElement.style.setProperty("--sheet-h", `${Math.round(h)}px`);
  }
  function sync() {
    const u = new URLSearchParams();
    if (state.region) u.set("region", state.region);
    if (state.port) u.set("port", state.port);
    history.replaceState(null, "", "?" + u.toString());
  }
  function selectPort(code) {
    state.port = code;
    state.region = CW.portById[code].region;
    state.search = "";
    document.body.classList.remove("nav-open");
    readout.hidden = true;
    haloPort(); render(); frame(); sync();
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-region],[data-port],[data-var],[data-lead],[data-close-port],[data-open-nav],[data-close-nav]");
    if (!t) return;
    if (t.dataset.openNav !== undefined) { document.body.classList.add("nav-open"); renderNav(); return; }
    if (t.dataset.closeNav !== undefined) { document.body.classList.remove("nav-open"); return; }
    if (t.dataset.region !== undefined) { state.region = t.dataset.region || null; state.port = null; haloPort(); render(); frame(); sync(); }
    else if (t.dataset.port) selectPort(Number(t.dataset.port));
    else if (t.dataset.var) { state.variable = t.dataset.var; setRaster(); render(); }
    else if (t.dataset.lead) { state.lead = Number(t.dataset.lead); setRaster(); render(); }
    else if (t.dataset.closePort !== undefined) { state.port = null; haloPort(); render(); frame(); sync(); }
  });
  document.addEventListener("change", (e) => {
    if (!e.target.matches("[data-var-select]")) return;
    state.variable = e.target.value; setRaster(); render();
  });
  document.addEventListener("input", (e) => {
    if (!e.target.matches(".nav-search input")) return;
    state.search = e.target.value;
    renderNav();
    const i = document.querySelector(".nav-search input");
    i.focus(); i.setSelectionRange(i.value.length, i.value.length);
  });
  render();
  addEventListener("resize", () => frame(false));
})();

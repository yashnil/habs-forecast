/* Ocean Map prototype. URL params: ?region=<id>&port=<code>&var=<variable>&lead=<0-3> */
(function () {
  const { D, icon, esc, fmtDay, fmtDate, pct, sig, REGIONS, REGION, CALIFORNIA } = CW;
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
  };
  const ports = D.port_intel.ports;
  const portLead = (p, lead = state.lead, v = state.variable) => p.charm?.leads.find((l) => l.lead_days === lead)?.variables[v];
  const isMobile = () => matchMedia("(max-width: 720px)").matches;

  // ---------- map ----------
  const graticule = { type: "FeatureCollection", features: [] };
  for (let lat = 32; lat <= 42; lat++) graticule.features.push({ type: "Feature", properties: { label: `${lat}°N` }, geometry: { type: "LineString", coordinates: [[-130, lat], [-114, lat]] } });
  for (let lon = -126; lon <= -116; lon += 2) graticule.features.push({ type: "Feature", properties: { label: `${-lon}°W` }, geometry: { type: "LineString", coordinates: [[lon, 30], [lon, 44]] } });

  const regionLabels = { type: "FeatureCollection", features: REGIONS.map((r) => ({ type: "Feature", properties: { label: r.label.toUpperCase() }, geometry: { type: "Point", coordinates: [r.bounds[0][0] - (r.id === "southern_california" ? -0.2 : 1.1), (r.bounds[0][1] + r.bounds[1][1]) / 2 - (r.id === "southern_california" ? 0.9 : 0)] } })) };
  const stations = D.observations.stations.map((s) => ({ type: "Feature", properties: { id: s.station_id, name: s.name }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } })).filter((f) => f.geometry.coordinates[0] != null);
  const FONT = ["Noto Sans Regular"];
  const style = {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      omt: { type: "vector", url: "https://tiles.openfreemap.org/planet", attribution: "OpenFreeMap · © OpenMapTiles · © OpenStreetMap contributors" },
      grat: { type: "geojson", data: graticule },
      official: { type: "geojson", data: D.official.geometry },
      ports: { type: "geojson", data: D.ports },
      regions: { type: "geojson", data: regionLabels },
      stations: { type: "geojson", data: { type: "FeatureCollection", features: stations } },
    },
    layers: [
      { id: "land", type: "background", paint: { "background-color": "#1c2a3c" } },
      { id: "landcover", type: "fill", source: "omt", "source-layer": "landcover", filter: ["in", ["get", "class"], ["literal", ["wood", "forest"]]], paint: { "fill-color": "#203046", "fill-opacity": 0.7 } },
      { id: "water", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-color": "#0b1d33" } },
      { id: "grat", type: "line", source: "grat", paint: { "line-color": "#6f8db0", "line-opacity": 0.16, "line-width": 0.6 } },
      { id: "coastline", type: "line", source: "omt", "source-layer": "water", paint: { "line-color": "#a8bfd6", "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.5, 9, 1.1, 12, 1.6], "line-opacity": 0.9 } },
      { id: "roads", type: "line", source: "omt", "source-layer": "transportation", minzoom: 6, filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk"]]], paint: { "line-color": "#2b3d55", "line-width": 0.8 } },
      { id: "state", type: "line", source: "omt", "source-layer": "boundary", filter: ["all", ["<=", ["coalesce", ["get", "admin_level"], 99], 4], ["!=", ["coalesce", ["get", "maritime"], 0], 1]], paint: { "line-color": "#3c5372", "line-width": 0.8, "line-dasharray": [3, 2] } },
      { id: "official-fill", type: "fill", source: "official", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#f6bb5c", "fill-opacity": 0.07 } },
      { id: "official-poly-casing", type: "line", source: "official", filter: ["!=", ["get", "kind"], "lat_limit"], paint: { "line-color": "#06111e", "line-width": 3.4, "line-opacity": 0.55 } },
      { id: "official-poly", type: "line", source: "official", filter: ["!=", ["get", "kind"], "lat_limit"], paint: { "line-color": "#f6bb5c", "line-width": 1.4, "line-dasharray": [2, 1.5] } },
      { id: "official-lat-casing", type: "line", source: "official", filter: ["==", ["get", "kind"], "lat_limit"], paint: { "line-color": "#06111e", "line-width": 4, "line-opacity": 0.6 } },
      { id: "official-lat", type: "line", source: "official", filter: ["==", ["get", "kind"], "lat_limit"], paint: { "line-color": "#f6bb5c", "line-width": 2 } },
      { id: "official-lat-label", type: "symbol", source: "official", filter: ["==", ["get", "kind"], "lat_limit"], minzoom: 6.2, layout: { "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": FONT, "text-size": 11, "text-offset": [0, -0.8] }, paint: { "text-color": "#f6bb5c", "text-halo-color": "#06111e", "text-halo-width": 1.6 } },
      { id: "grat-label", type: "symbol", source: "grat", layout: { "symbol-placement": "line", "symbol-spacing": 2000, "text-field": ["get", "label"], "text-font": FONT, "text-size": 10 }, paint: { "text-color": "#7f9bbd", "text-opacity": 0.7, "text-halo-color": "#0b1d33", "text-halo-width": 1 } },
      { id: "city", type: "symbol", source: "omt", "source-layer": "place", minzoom: 6.6, filter: ["in", ["get", "class"], ["literal", ["city", "town"]]], layout: { "text-field": ["get", "name"], "text-font": FONT, "text-size": ["interpolate", ["linear"], ["zoom"], 5, 10, 10, 12], "symbol-sort-key": ["get", "rank"] }, paint: { "text-color": "#8a9cb2", "text-halo-color": "#1c2a3c", "text-halo-width": 1.2 } },
      { id: "region-label", type: "symbol", source: "regions", maxzoom: 7.2, layout: { "text-field": ["get", "label"], "text-font": FONT, "text-size": 11, "text-letter-spacing": 0.18, "text-anchor": "right", "text-allow-overlap": true }, paint: { "text-color": "#dfe7f0", "text-opacity": 0.85, "text-halo-color": "#0b1d33", "text-halo-width": 1.2 } },
      { id: "stations", type: "circle", source: "stations", paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 2.6, 10, 5], "circle-color": "#3fc1b0", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.2 } },
      { id: "ports-halo", type: "circle", source: "ports", paint: { "circle-radius": ["case", ["==", ["get", "port_code"], -1], 11, 0], "circle-color": "#ffffff", "circle-opacity": 0.18 } },
      { id: "ports", type: "circle", source: "ports", paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 3.4, 10, 6], "circle-color": "#ffffff", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.6 } },
      { id: "ports-label", type: "symbol", source: "ports", minzoom: 6.6, layout: { "text-field": ["get", "display_name"], "text-font": FONT, "text-size": ["interpolate", ["linear"], ["zoom"], 5.4, 11, 10, 13.5], "text-anchor": "left", "text-offset": [0.8, 0], "text-optional": true }, paint: { "text-color": "#ffffff", "text-halo-color": "#06111e", "text-halo-width": 1.6 } },
    ],
  };
  const map = new maplibregl.Map({ container: "map", style, bounds: CALIFORNIA, attributionControl: { compact: false }, fadeDuration: 0, preserveDrawingBuffer: true });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric", maxWidth: 90 }), "bottom-right");

  function setRaster() {
    const l = layerFor(state.variable, state.lead);
    if (map.getLayer("forecast")) map.removeLayer("forecast");
    if (map.getSource("forecast")) map.removeSource("forecast");
    map.addSource("forecast", { type: "image", url: l.raster.url, coordinates: l.raster.corners });
    map.addLayer({ id: "forecast", type: "raster", source: "forecast", paint: { "raster-opacity": ["interpolate", ["linear"], ["zoom"], 6, 0.84, 8, 0.72, 10, 0.55], "raster-resampling": "nearest", "raster-fade-duration": 0 } }, "grat");
  }

  // Frame the coast inside whatever the panels leave visible (measured, not assumed).
  function padding() {
    const h = (id) => document.getElementById(id).getBoundingClientRect().height;
    if (isMobile()) return { top: 16, left: 16, right: 16, bottom: (state.port ? h("inspector") : h("dock")) + 12 };
    return { top: 24, left: 344 + 40, right: state.port ? 384 + 40 : 64, bottom: 24 };
  }
  function frame(animate = true) {
    // Regions are framed with ~0.5° of context so 3 km model cells stay small on screen.
    const r = state.port ? CW.portById[state.port].region : state.region;
    const b = r ? [[REGION[r].bounds[0][0] - 0.6, REGION[r].bounds[0][1] - 0.4], [REGION[r].bounds[1][0] + 0.3, REGION[r].bounds[1][1] + 0.4]] : [[-126.2, 32.45], [-117.1, 42.05]];
    map.fitBounds(b, { padding: padding(), duration: animate ? 600 : 0 });
  }

  map.on("load", () => {
    setRaster();
    if (isMobile()) map.setLayoutProperty("region-label", "visibility", "none");
    frame(false);
    map.setPaintProperty("ports-halo", "circle-radius", ["case", ["==", ["get", "port_code"], state.port ?? -1], 12, 0]);
    map.on("click", "ports", (e) => selectPort(e.features[0].properties.port_code));
    map.on("mouseenter", "ports", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "ports", () => (map.getCanvas().style.cursor = ""));
    map.once("idle", () => document.body.setAttribute("data-ready", "1"));
  });

  // ---------- left panel ----------
  function renderCoast() {
    const scope = state.region ? CW.noticesForRegion(state.region) : CW.records.map((record) => ({ record, rel: null }));
    const shown = scope.slice(0, 3);
    const where = state.region ? `may apply in ${REGION[state.region].label}` : "active in California";
    const regionRow = (r) => {
      const ps = ports.filter((p) => p.region === r.id);
      const vals = ps.map((p) => portLead(p)?.median).filter((v) => v != null);
      const open = state.region === r.id;
      return `<button class="region" aria-expanded="${open}" data-region="${r.id}">
          <span class="name">${r.label}</span><span class="range">${vals.length ? `${pct(Math.min(...vals))}–${pct(Math.max(...vals))}` : "—"}</span>
          <span class="ticks" aria-hidden="true">${ps.map((p) => `<i style="background:${portLead(p) ? pClass(portLead(p).median) : "#e4e0d7"}"></i>`).join("")}</span>
        </button>
        ${open ? `<div class="ports">${ps.map((p) => { const v = portLead(p); return `<button class="port-row" data-port="${p.port_code}" aria-pressed="${state.port === p.port_code}"><span>${esc(p.display_name)}</span><span class="bar"><i style="width:${(v?.median ?? 0) * 100}%;background:${v ? pClass(v.median) : "transparent"}"></i></span><span class="val">${pct(v?.median)}</span></button>`; }).join("")}</div>` : ""}`;
    };
    const L = layerFor(state.variable, state.lead);
    document.getElementById("coast").innerHTML = `
      <div class="off-block" data-testid="official-summary">
        <div class="off-head"><span class="off-title">${icon("shield")}${scope.length} official notice${scope.length === 1 ? "" : "s"} ${where}</span></div>
        <div class="off-rows">${shown.map(({ record: r, rel }) => `<button class="notice-row" data-open-official data-focus="${r.id}">${CW.agencyChip(r.agency)}<span class="t">${esc(r.title)}</span>${icon("chevron", "icon-s")}</button>`).join("")}</div>
        <div class="off-foot">${CW.verificationLine()}<button class="link-arrow" data-open-official>${scope.length > 3 ? `${scope.length - 3} more · ` : ""}All ${CW.records.length}${icon("chevron", "icon-s")}</button></div>
      </div>
      <div class="coast-head"><h2>Coast, north to south</h2><span class="fine">${fmtDay(L.time.valid_date)}</span></div>
      <div class="coast-list">
        <button class="region all" data-region="" aria-expanded="${!state.region}"><span class="name">All California</span><span class="range">${ports.length} ports</span></button>
        ${REGIONS.map(regionRow).join("")}
      </div>
      <p class="coast-note fine">Port values: median ${VARS.find((v) => v.id === state.variable).long.toLowerCase()} probability of model cells within 15 km. Not conditions at the dock.</p>`;
  }

  const revealPort = () => document.querySelector('.port-row[aria-pressed="true"]')?.scrollIntoView({ block: "nearest" });

  // ---------- dock ----------
  function renderDock() {
    const L = layerFor(state.variable, state.lead);
    const leads = charm.filter((l) => l.variable === state.variable).sort((a, b) => a.time.lead_days - b.time.lead_days);
    const scope = state.region ? CW.noticesForRegion(state.region) : CW.records;
    document.getElementById("dock").innerHTML = `
      <button class="m-official" data-open-official><span>${icon("shield")}${scope.length} official notices ${state.region ? "may apply here" : "in California"}</span><span class="unverified">Not verified ${icon("chevron", "icon-s")}</span></button>
      <div class="dock-row top">
        <div class="seg var-seg" role="group" aria-label="Forecast quantity">${VARS.map((v) => `<button data-var="${v.id}" aria-pressed="${v.id === state.variable}">${v.short}</button>`).join("")}</div>
        <button class="layer-btn">${icon("layers", "icon-s")}Layers</button>
      </div>
      <div class="m-regions" role="group" aria-label="Region">${[{ id: "", label: "All California" }, ...REGIONS].map((r) => `<button data-region="${r.id}" aria-pressed="${(state.region ?? "") === r.id}">${r.label}</button>`).join("")}</div>
      <div class="timeline" role="group" aria-label="Forecast day">${leads.map((l) => `<button class="tl-day" data-lead="${l.time.lead_days}" aria-pressed="${l.time.lead_days === state.lead}"><b>${l.time.valid_date === today ? "Today" : fmtDate(l.time.valid_date, { weekday: "short" })} ${fmtDate(l.time.valid_date, { month: "short", day: "numeric" })}</b><span class="lt">${l.time.lead_days === 0 ? "Nowcast" : `Forecast +${l.time.lead_days} d`}</span></button>`).join("")}</div>
      <div class="legend-row">
        <div>
          <p class="legend-title">${esc(L.threshold_text)}</p>
          <div class="ramp" aria-hidden="true">${P_CLASSES.map((c) => `<i style="background:${c}"></i>`).join("")}</div>
          <div class="ramp-labels" aria-hidden="true">${[0, 20, 40, 60, 80, 100].map((n) => `<span style="left:${n}%">${n}${n === 100 ? "%" : ""}</span>`).join("")}</div>
        </div>
        <div style="display:grid;gap:8px;justify-items:end">
          <span class="nodata"><i></i>No model value</span>
          <span class="tl-meta"><span class="chip chip-model">Model</span><span class="issued">C-HARM v3.1 · NOAA · issued ${fmtDate(run.issued_date, { month: "short", day: "numeric" })}</span>${CW.freshChip(charmFresh)}</span>
        </div>
      </div>`;
  }

  // ---------- inspector ----------
  function renderInspector() {
    const el = document.getElementById("inspector");
    document.body.classList.toggle("inspecting", !!state.port);
    if (!state.port) { el.hidden = true; return; }
    const p = CW.portById[state.port];
    const v = VARS.find((x) => x.id === state.variable);
    const L = layerFor(state.variable, state.lead);
    const cur = portLead(p);
    const notices = CW.noticesForPort(p.port_code);
    const st = D.observations.stations.filter((s) => s.nearest_port_code === p.port_code).sort((a, b) => a.nearest_port_km - b.nearest_port_km)[0];
    const pda = st?.summaries.find((x) => x.variable === "pDA");
    const hist = (p.charm.history[state.variable] || []).map((h) => ({ t: Date.parse(h.date), v: h.value }));
    const chl = p.chlorophyll;
    const relText = { statewide: "statewide", port_latitude_within_stated_range: "port within the notice's latitudes", same_county: "same county", named_area_nearby: "named area nearby" };
    el.hidden = false;
    el.innerHTML = `
      <div class="ins-head">
        <p class="eyebrow">Port · ${REGION[p.region].label}</p>
        <h2>${esc(p.display_name)}</h2>
        <p class="sub">${esc(p.county)} County · <span class="mono">${p.lat.toFixed(3)}°N ${Math.abs(p.lon).toFixed(3)}°W</span></p>
        <button class="ins-close" data-close-port aria-label="Close">${icon("close")}</button>
      </div>
      <section class="ins-sec official" data-testid="inspector-official">
        <div class="ins-sec-head"><span class="eyebrow official">Official · ${notices.length} may apply</span>${CW.verificationLine()}</div>
        ${notices.map(({ record: r, rel }) => `<button class="notice-row" data-open-official data-focus="${r.id}">${CW.agencyChip(r.agency)}<span><span class="t" style="display:block;white-space:normal">${esc(r.title)}</span><span class="why">${relText[rel.relation] ?? rel.relation}</span></span>${icon("chevron", "icon-s")}</button>`).join("")}
      </section>
      <section class="ins-sec" data-testid="inspector-model">
        <div class="ins-sec-head"><span class="eyebrow model">Model forecast · C-HARM</span>${CW.freshChip(charmFresh, `${charmFresh.label} · issued ${fmtDate(run.issued_date, { month: "short", day: "numeric" })}`)}</div>
        <p class="q">${esc(L.threshold_text)}</p>
        <div class="big-prob"><span class="n">${pct(cur?.median)}</span><span class="d">median of ${cur?.n ?? 0} model cells within 15 km · ${fmtDay(L.time.valid_date)}</span></div>
        <div class="leads" role="group" aria-label="Forecast days">${p.charm.leads.map((l) => { const x = l.variables[state.variable]; return `<button class="lead" data-lead="${l.lead_days}" aria-pressed="${l.lead_days === state.lead}"><span class="day">${l.lead_days === 0 ? "Nowcast" : fmtDate(l.valid_date, { weekday: "short" }) + " " + fmtDate(l.valid_date, { day: "numeric" })}</span><span class="v">${pct(x?.median)}</span><span class="rng" title="range across cells ${pct(x?.min)}–${pct(x?.max)}"><i style="left:${x.min * 100}%;width:${(x.max - x.min) * 100}%;background:${pClass(x.median)}"></i><b style="left:calc(${x.median * 100}% - 1px)"></b></span></button>`; }).join("")}</div>
        <p class="fine">Bars span the lowest to highest cell; the tick is the median.</p>
        <div class="spark-wrap">
          <div class="ins-sec-head" style="margin:10px 0 2px"><span class="fine">Last 30 days of nowcasts (median)</span><span class="fine">${hist.length} runs</span></div>
          ${sparkline(hist, { w: 344, h: 64, t0: CW.NOW - 31 * CW.DAY, t1: CW.NOW })}
        </div>
        <p class="caveat"><b>A probability for nearby water, not a measurement and not a closure decision.</b> A low value does not mean it is safe to fish or harvest.</p>
      </section>
      ${st ? `<section class="ins-sec">
        <div class="ins-sec-head"><span class="eyebrow measured">Measured nearby · CalHABMAP</span>${CW.freshChip(CW.freshness(pda.last_date, { current: 14, stale: 45 }), CW.agoText(CW.daysAgo(pda.last_date)))}</div>
        <div class="kv"><span class="k">${esc(st.name)} · ${st.nearest_port_km} km</span><span class="v"></span>
          <span class="k">Particulate domoic acid, ${fmtDate(pda.last_date, { month: "short", day: "numeric" })}</span><span class="v">${pda.last_qualifier === "reported_zero" ? "reported 0" : sig(pda.last_value) + " ng/mL"}</span></div>
        <p style="margin-top:10px"><a class="link-arrow" href="bloom.html?station=${st.station_id}">Open station in Bloom Intelligence${icon("chevron", "icon-s")}</a></p>
      </section>` : ""}
      ${chl?.latest ? `<section class="ins-sec">
        <div class="ins-sec-head"><span class="eyebrow">Satellite · NOAA VIIRS</span><span class="fine">8-day window centred ${fmtDate(chl.latest_center_date, { month: "short", day: "numeric" })}</span></div>
        <div class="kv"><span class="k">Chlorophyll-a, median within 15 km</span><span class="v">${sig(chl.latest.median)} mg/m³</span></div>
        <p class="caveat">Algae biomass, not toxin. ${Math.round(chl.latest_valid_fraction * 100)}% of nearby pixels were cloud-free.</p>
      </section>` : ""}
      <section class="ins-sec"><p class="fine">${esc(p.caveats[0])}</p></section>`;
  }

  function render() { renderCoast(); renderDock(); renderInspector(); revealPort(); }
  function sync() {
    const u = new URLSearchParams();
    if (state.region) u.set("region", state.region);
    if (state.port) u.set("port", state.port);
    history.replaceState(null, "", "?" + u.toString());
  }
  function selectPort(code) {
    state.port = code;
    state.region = CW.portById[code].region;
    if (map.getLayer("ports-halo")) map.setPaintProperty("ports-halo", "circle-radius", ["case", ["==", ["get", "port_code"], code ?? -1], 12, 0]);
    render(); frame(); sync();
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-region],[data-port],[data-var],[data-lead],[data-close-port]");
    if (!t) return;
    if (t.dataset.region !== undefined) { state.region = t.dataset.region || null; state.port = null; render(); frame(); sync(); }
    else if (t.dataset.port) selectPort(Number(t.dataset.port));
    else if (t.dataset.var) { state.variable = t.dataset.var; setRaster(); render(); }
    else if (t.dataset.lead) { state.lead = Number(t.dataset.lead); setRaster(); render(); }
    else if (t.dataset.closePort !== undefined) { state.port = null; map.setPaintProperty("ports-halo", "circle-radius", 0); render(); frame(); sync(); }
  });
  render();
  addEventListener("resize", () => frame(false));
})();

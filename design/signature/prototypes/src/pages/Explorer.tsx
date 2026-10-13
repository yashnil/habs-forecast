import type { Map as MLMap } from "maplibre-gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { hashParams, setHashParams } from "../App";
import { NET } from "../data/net";
import { useGrid, useManifest, useObservations, useOfficial, usePorts } from "../data/store";
import type { Layer } from "../data/types";
import { dms, km, type LngLat } from "../lib/geo";
import { modelReading, nearestPort, noticeGeometry, type Reading, satelliteReading, stationReading } from "../lib/provenance";
import { forecastFreshness, TODAY } from "../lib/time";
import type { FieldMode } from "../map/field";
import { coastPathFromPorts, journey, type Journey } from "../map/journey";
import { MapStage, type Show, type StageHandle } from "../map/MapStage";
import { highlightThread, setThreads, type Thread, type ThreadKind } from "../map/threads";
import { Ledger, OfficialActions, probWords } from "../ui/Ledger";
import { Masthead } from "../ui/Masthead";
import { type PlaceHit, PlaceSearch } from "../ui/PlaceSearch";
import { type Purpose, PURPOSES, PurposeCard, TOUR } from "../ui/Purpose";
import { Sheet, type Snap } from "../ui/Sheet";
import { TimeBar } from "../ui/TimeBar";
import "./explorer.css";

const VARS = [
  { id: "particulate_domoic", label: "Toxin in the water", sub: "particulate domoic acid > 500 ng/L" },
  { id: "pseudo_nitzschia", label: "Bloom-forming algae", sub: "Pseudo-nitzschia > 10,000 cells/L" },
  { id: "cellular_domoic", label: "Toxin per cell", sub: "cellular domoic acid" },
] as const;
const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const prefersReduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const useMedia = (q: string) => {
  const [m, setM] = useState(() => matchMedia(q).matches);
  useEffect(() => { const mq = matchMedia(q); const on = () => setM(mq.matches); mq.addEventListener("change", on); return () => mq.removeEventListener("change", on); }, [q]);
  return m;
};

export function Explorer() {
  const p0 = useMemo(() => hashParams(), []);
  const mobile = useMedia("(max-width: 760px)");
  const [mode, setModeState] = useState<"coastal" | "research">(() => (p0.get("mode") as "coastal" | "research") ?? (store.get("cw-mode-v1") as "coastal" | "research") ?? "coastal");
  const [show, setShow] = useState<Show>(() => (p0.get("show") as Show) ?? (mode === "research" ? "forecast" : "base"));
  const [variable, setVariable] = useState<string>(p0.get("v") ?? "particulate_domoic");
  const [lead, setLead] = useState<number>(() => Number(p0.get("lead") ?? 2));
  const [fieldMode, setFieldMode] = useState<FieldMode>((p0.get("field") as FieldMode) ?? "veil");
  const [sel, setSel] = useState<LngLat | null>(() => { const a = p0.get("at"); return a ? (a.split(",").map(Number) as LngLat) : null; });
  const [species, setSpecies] = useState<string | null>(p0.get("species"));
  const [snap, setSnap] = useState<Snap>("peek");
  const [sheetH, setSheetH] = useState(132);
  const [tiles, setTiles] = useState<"ok" | "failed">("ok");
  const [purpose, setPurpose] = useState<Purpose | null>(() => (store.get("cwsig-purpose") as Purpose) ?? null);
  const [askPurpose, setAskPurpose] = useState(() => !p0.get("at") && !store.get("cwsig-purpose") && p0.get("welcome") !== "0");
  const [caption, setCaption] = useState<string | null>(null);
  const [tour, setTour] = useState<number | null>(null);
  const [travelling, setTravelling] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [recent, setRecent] = useState<PlaceHit[]>(() => { try { return JSON.parse(store.get("cwsig-recent") ?? "[]"); } catch { return []; } });
  const stage = useRef<StageHandle>(null);
  const mapRef = useRef<MLMap | null>(null);
  const trip = useRef<Journey | null>(null);
  const research = mode === "research";

  /* -------------------------------------------------------------- data (each part independent) */
  const manifest = useManifest();
  const m = manifest.data;
  const official = useOfficial(m);
  const observations = useObservations(m);
  const ports = usePorts(m);
  const layerId = `charm_${variable}_lead${lead}`;
  const layer = m?.layers.find((l) => l.layer_id === layerId);
  const grid = useGrid(show === "forecast" || sel ? layer : undefined);
  const satLayer = m?.layers.find((l) => l.layer_id === "olci300_chl_latest");
  const run = m?.forecast_runs.find((r) => r.group_id === "charm");
  const charmSrc = m?.sources.find((s) => s.source_id === "charm");
  const fresh = run ? forecastFreshness(run.issued_date, charmSrc) : null;
  const leads = useMemo(() => (m?.layers ?? []).filter((l) => l.group_id === "charm" && l.variable === variable).map((l) => ({ lead: l.time.lead_days ?? 0, date: l.time.valid_date! })).sort((a, b) => a.lead - b.lead), [m, variable]);
  const coastPath = useMemo(() => (ports.data ? coastPathFromPorts(ports.data) : []), [ports.data]);

  /* -------------------------------------------------------------- mode & url */
  const setMode = (v: "coastal" | "research") => { setModeState(v); store.set("cw-mode-v1", v); if (v === "research" && show === "base") setShow("forecast"); };
  useEffect(() => {
    setHashParams({ mode, show, v: variable === "particulate_domoic" ? null : variable, lead: String(lead), field: fieldMode === "veil" ? null : fieldMode, at: sel ? `${sel[0].toFixed(4)},${sel[1].toFixed(4)}` : null, species });
  }, [mode, show, variable, lead, fieldMode, sel, species]);

  /* -------------------------------------------------------------- reading at the selection */
  const [satRead, setSatRead] = useState<Reading["satellite"]>({ status: "off" });
  useEffect(() => {
    if (!sel || show !== "satellite" || !satLayer) { setSatRead({ status: "off" }); return; }
    let alive = true;
    setSatRead({ status: "loading" });
    satelliteReading(satLayer, sel).then((r) => alive && setSatRead(r), () => alive && setSatRead({ status: "gap", reason: { kind: "source-unavailable", detail: "The satellite pixel could not be read just now." } }));
    return () => { alive = false; };
  }, [sel?.join(), show, satLayer]);

  const reading: Reading | null = useMemo(() => {
    if (!sel) return null;
    const np = ports.data ? nearestPort(ports.data, sel) : null;
    return {
      point: sel,
      place: { label: ports.data ? placeLabel(sel, np) : "This water point", port: np?.port, portKm: np?.km },
      model: modelReading(grid.data, layer, sel, m, grid.status === "failed"),
      satellite: satRead,
      station: stationReading(observations.data, sel),
      ...noticeGeometry(official.data, sel),
    };
  }, [sel?.join(), grid.data, grid.status, layer, observations.data, official.data, ports.data, satRead, m]);

  /* -------------------------------------------------------------- threads follow the reading */
  const threadKinds = useRef<ThreadKind[]>([]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!reading) { setThreads(map, []); threadKinds.current = []; return; }
    const t: Thread[] = [];
    const p = reading.point;
    if (reading.station && !reading.station.tooFar) t.push({ id: "station", from: p, to: [reading.station.station.lon, reading.station.station.lat], colour: "#4fd1ae" });
    else if (reading.station) t.push({ id: "station", from: p, to: [reading.station.station.lon, reading.station.station.lat], colour: "rgba(79,209,174,0.5)" });
    const mr = reading.model;
    if (mr.status === "value" && show === "forecast") t.push({ id: "model", from: p, to: mr.center, colour: "#c3a6f5" });
    if (mr.status === "gap" && mr.reason.kind === "nearshore-mask" && mr.reason.nearest && show === "forecast") t.push({ id: "model", from: p, to: mr.reason.nearest.center, colour: "#c3a6f5" });
    const sr = reading.satellite;
    if (sr.status === "value" && show === "satellite" && km(p, sr.center) > 0.05) t.push({ id: "satellite", from: p, to: sr.center, colour: "#7ae3c4" });
    if (sr.status === "gap" && sr.reason.kind === "cloud" && sr.reason.nearest) t.push({ id: "satellite", from: p, to: sr.reason.nearest.center, colour: "#7ae3c4" });
    const lim = reading.notices.filter((n) => n.nearestLimit && n.nearestLimit.km < 40).sort((a, b) => a.nearestLimit!.km - b.nearestLimit!.km)[0];
    if (lim) t.push({ id: "notice", from: p, to: [p[0], lim.nearestLimit!.lat], colour: "#f2b04c" });
    threadKinds.current = t.map((x) => x.id);
    setThreads(map, t);
  }, [reading, show, mapReady]);
  const onThread = useCallback((k: ThreadKind | null) => { if (mapRef.current) highlightThread(mapRef.current, k, threadKinds.current); }, []);

  /* -------------------------------------------------------------- selection marker (HTML, settles in) */
  const marker = useRef<HTMLDivElement | null>(null);
  const markerObj = useRef<import("maplibre-gl").Marker | null>(null);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!sel) { markerObj.current?.remove(); markerObj.current = null; return; }
    import("maplibre-gl").then(({ Marker }) => {
      if (!marker.current) { marker.current = document.createElement("div"); marker.current.className = "sel-marker"; marker.current.innerHTML = '<span class="sel-ring"></span><span class="sel-dot"></span>'; marker.current.setAttribute("aria-hidden", "true"); }
      if (!markerObj.current) markerObj.current = new Marker({ element: marker.current }).setLngLat(sel).addTo(map);
      else markerObj.current.setLngLat(sel);
      marker.current.classList.remove("settle"); void marker.current.offsetWidth; marker.current.classList.add("settle");
    });
  }, [sel?.join(), mapReady]);

  /* -------------------------------------------------------------- camera */
  const padding = mobile ? { top: 72, bottom: sheetH + 24, left: 24, right: 24 } : { top: 88, bottom: 120, left: 452, right: 72 };
  const pick = useCallback((p: LngLat) => {
    trip.current?.cancel();
    setSel(p);
    setAskPurpose(false);
    if (mobile && snap === "peek") setSnap("peek");
    const map = mapRef.current;
    if (!map) return;
    // keep the point in the clear part of the map (not under the panel or sheet)
    const pt = map.project(p);
    const c = map.getContainer();
    const pad = mobile ? { top: 72, bottom: sheetH + (show === "base" ? 90 : 190), left: 24, right: 24 } : { top: 88, bottom: 120, left: 452, right: 72 };
    if (pt.x < pad.left || pt.x > c.clientWidth - pad.right || pt.y < pad.top || pt.y > c.clientHeight - pad.bottom) {
      map.easeTo({ center: p, padding: pad, duration: prefersReduced() ? 0 : 420 });
    }
  }, [mobile, snap, sheetH, show]);

  const travel = useCallback((to: LngLat, zoom: number, label: string) => {
    const map = mapRef.current;
    if (!map) return;
    trip.current?.cancel();
    setTravelling(label);
    const j = journey(map, { center: to, zoom, padding }, coastPath);
    trip.current = j;
    j.done.then((how) => { if (trip.current === j) { setTravelling(null); trip.current = null; } void how; });
  }, [coastPath, padding.left, padding.bottom]);

  const onPlace = (h: PlaceHit) => {
    const sea = h.kind === "port" ? offshore(h.at, 4) : h.at;
    setSel(sea);
    setAskPurpose(false);
    travel(sea, h.kind === "station" ? 11 : 10, h.label);
    const next = [{ ...h, kind: "recent" as const, id: h.id }, ...recent.filter((r) => r.id !== h.id)].slice(0, 5);
    setRecent(next);
    store.set("cwsig-recent", JSON.stringify(next));
  };

  /* -------------------------------------------------------------- purpose → the map answers */
  const choosePurpose = (pp: Purpose) => {
    setPurpose(pp);
    store.set("cwsig-purpose", pp);
    setAskPurpose(false);
    const home: LngLat = [-121.94, 36.62]; // Monterey (default place in the redesign)
    if (pp === "commercial") { setShow("forecast"); setFieldMode("lens"); setModeState("coastal"); const at: LngLat = [-122.08, 36.72]; setSel(at); travel(at, 9, "Monterey Bay offshore"); }
    if (pp === "shellfish") { setShow("base"); setFieldMode("veil"); setModeState("coastal"); const at: LngLat = [-121.93, 36.63]; setSel(at); travel(at, 11.2, "Monterey shoreline"); }
    if (pp === "research") { setMode("research"); setShow("forecast"); setFieldMode("edges"); setSel(null); mapRef.current?.fitBounds([[-125.2, 32.3], [-116.8, 42.1]], { padding, duration: prefersReduced() ? 0 : 900 }); }
    if (pp === "explore") { setTour(0); return; }
    void home;
    setCaption(PURPOSES.find((x) => x.id === pp)!.caption);
  };
  useEffect(() => {
    if (tour == null) return;
    const s = TOUR[tour];
    setShow(s.show);
    setSel(null);
    travel(s.at, s.zoom, s.title);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tour]);
  useEffect(() => { if (!caption) return; const t = setTimeout(() => setCaption(null), 9000); return () => clearTimeout(t); }, [caption]);

  /* -------------------------------------------------------------- notices emphasised on the map */
  const hl = useMemo(() => {
    if (!reading) return null;
    return reading.notices.filter((n) => n.tier !== "outside").map((n) => n.record.id);
  }, [reading]);

  const activeNotices = official.data ? official.data.registry.records.filter((r) => r.status === "active" && !official.data!.conflicts.some((c) => c.endsWith(`record ${r.id}.`))).length : null;
  const speciesList = useMemo(() => [...new Set((official.data?.registry.records ?? []).flatMap((r) => r.species))].sort(), [official.data]);
  const failedForecast = grid.status === "failed";

  /* -------------------------------------------------------------- pieces */
  const head = reading ? (
    <div className="place-head">
      <p className="t-micro place-kicker">{research ? "Water point" : "Your waters"}</p>
      <h1 className="t-title place-title">{reading.place.label}</h1>
      <p className="t-coord">{dms(reading.point[1], reading.point[0])}</p>
    </div>
  ) : (
    <div className="place-head">
      <p className="t-micro place-kicker">California coast</p>
      <h1 className="t-title place-title">{travelling ?? "Choose your waters"}</h1>
      <p className="t-meta">Tap the water, or search a port or pier.</p>
    </div>
  );

  const summary = reading && !research ? <OneLine r={reading} species={species} showForecast={show === "forecast"} /> : null;

  const plan = (
    <div className="plan" aria-label="Your plan">
      <label className="plan-species">
        <span className="t-micro">Species</span>
        <select value={species ?? ""} onChange={(e) => setSpecies(e.target.value || null)}>
          <option value="">Any species</option>
          {speciesList.map((s) => <option key={s} value={s}>{s[0].toUpperCase() + s.slice(1)}</option>)}
        </select>
      </label>
    </div>
  );

  const fieldControlsInner = (
    <div className="fieldmode" role="radiogroup" aria-label="How loudly the forecast is drawn">
      {([["veil", "Whole field", "Field"], ["lens", "Around my point", "Near me"], ["edges", "Class lines", "Lines"]] as const).map(([id, l, short]) => (
        <button key={id} role="radio" aria-checked={fieldMode === id} onClick={() => setFieldMode(id)} disabled={id === "lens" && !sel} aria-label={l}><span className="fm-long">{l}</span><span className="fm-short" aria-hidden="true">{short}</span></button>
      ))}
    </div>
  );
  const body = (
    <>
      {!mobile && summary}
      {mobile && show === "forecast" && <div className="sheet-fieldmode">{fieldControlsInner}</div>}
      {reading && plan}
      {reading ? (
        <Ledger reading={reading} official={official} observations={observations} forecast={{ ...grid, failed: failedForecast }} manifest={m}
          research={research} species={species} onThread={onThread} satelliteOn={show === "satellite"} />
      ) : (
        <div className="empty-place">
          <p className="t-body">CoastWatch brings together what is <b>official</b>, what has been <b>measured</b>, what a <b>model</b> expects, and what <b>isn't known</b> — for any point on the water.</p>
          <ul className="legend-marks">
            <li><span className="mark mark-rule" aria-hidden="true" />Official notice limits</li>
            <li><span className="mark mark-sample" aria-hidden="true" />Shore samples</li>
            <li><span className="mark mark-field" aria-hidden="true" />NOAA forecast cells</li>
            <li><span className="mark mark-gap" aria-hidden="true" />No data</li>
          </ul>
          <OfficialActions />
        </div>
      )}
      {research && <ResearchTools variable={variable} setVariable={setVariable} fieldMode={fieldMode} setFieldMode={setFieldMode} m={m} reading={reading} />}
    </>
  );

  const layerSwitch = (
    <div className="layers" role="radiogroup" aria-label="What the sea shows">
      {([["base", "Seafloor", "mark-depth"], ["forecast", "Forecast", "mark-field"], ["satellite", "Satellite", "mark-pixel"]] as const).map(([id, label, mk]) => (
        <button key={id} role="radio" aria-checked={show === id} tabIndex={show === id ? 0 : -1} onClick={() => setShow(id)}
          onKeyDown={(e) => { const order: Show[] = ["base", "forecast", "satellite"]; const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (d) { e.preventDefault(); setShow(order[(order.indexOf(show) + d + 3) % 3]); } }}>
          <span className={`mark ${mk}`} aria-hidden="true" />{label}
        </button>
      ))}
    </div>
  );

  const fieldControls = show === "forecast" && !research && !mobile && (
    <div className="fieldmode" role="radiogroup" aria-label="How loudly the forecast is drawn">
      {([["veil", "Whole field"], ["lens", "Around my point"], ["edges", "Class lines"]] as const).map(([id, l]) => (
        <button key={id} role="radio" aria-checked={fieldMode === id} onClick={() => setFieldMode(id)} disabled={id === "lens" && !sel} title={id === "lens" && !sel ? "Choose a point first" : undefined}>{l}</button>
      ))}
    </div>
  );

  const satLegend = show === "satellite" && satLayer?.palette && (
    <div className="legend enter-fade">
      <p className="t-micro"><span className="mark mark-pixel" aria-hidden="true" /> Chlorophyll-a seen from space, mg/m³ (log scale). Algae, not toxin.</p>
      <div className="legend-ramp legend-ramp-smooth" aria-hidden="true" style={{ background: `linear-gradient(90deg, ${satLayer.palette.stops.map((s) => s.color).join(", ")})` }} />
      <p className="legend-scale t-micro"><span>0.05</span><span>1</span><span>50</span></p>
      <p className="t-micro"><span className="mark mark-gap" aria-hidden="true" /> No clear view (cloud, fog, no pass)</p>
    </div>
  );
  const legend = show === "satellite" ? satLegend : show === "forecast" && layer && (
    <div className="legend enter-fade">
      <p className="t-micro"><span className="mark mark-field" aria-hidden="true" /> {layer.threshold_text}</p>
      <div className="legend-ramp" aria-hidden="true">{Array.from({ length: 10 }, (_, i) => <span key={i} style={{ background: `var(--cw-p${i})` }} />)}</div>
      <p className="legend-scale t-micro"><span>0 %</span><span>chance</span><span>100 %</span></p>
      <p className="t-micro"><span className="mark mark-gap" aria-hidden="true" /> No value (model leaves it out)</p>
    </div>
  );

  const network = (NET.profile !== "fast" || NET.fail.size > 0) && (
    <p className="netflag t-micro">Simulated network: {NET.profile}{NET.fail.size ? ` · failing ${[...NET.fail].join(", ")}` : ""}{NET.cache ? " · device cache on" : ""}</p>
  );

  return (
    <div className="ex" data-mobile={mobile || undefined} data-research={research || undefined}>
      <Masthead mode={mode} onMode={setMode} notices={activeNotices} unverified onNotices={() => {}} compact={mobile} />
      <main className="ex-main">
        <MapStage ref={stage} show={show === "forecast" && !grid.data ? "base" /* calm basemap until the field exists: no hatch flash while loading or failed */ : show} grid={grid.data} layer={layer}
          field={{ mode: fieldMode, lens: fieldMode === "lens" && sel ? { lon: sel[0], lat: sel[1], km: 16 } : null }}
          satellite={satLayer} official={official.data} observations={observations.data} ports={ports.data}
          highlightNotices={hl} selection={sel} researchGrid={research}
          cells={{
            model: show === "forecast" && reading && reading.model.status !== "loading" ? ("cell" in reading.model ? reading.model.cell ?? null : null) : null,
            modelGap: reading?.model.status === "gap",
            pixel: show === "satellite" && reading && (reading.satellite.status === "value" || reading.satellite.status === "gap") ? (reading.satellite.status === "value" ? reading.satellite.pixel : reading.satellite.pixel ?? null) : null,
            pixelGap: reading?.satellite.status === "gap",
          }}
          initial={sel ? { center: sel, zoom: 9.5 } : { center: [-122.55, 36.78], zoom: 8.3 }}
          onReady={(map) => { mapRef.current = map; setMapReady(true); }}
          onPick={(p) => pick(p)} onTiles={setTiles} padding={padding} />

        {!mobile && (
          <aside className="panel" aria-label="Your waters">
            <div className="panel-scroll">
              {head}
              {body}
            </div>
          </aside>
        )}

        <div className="topbar">
          <PlaceSearch ports={ports.data} stations={observations.data?.stations} recent={recent} current={reading?.place.label ?? ""} onPick={onPlace} />
          {!mobile && layerSwitch}
          <button className="btn btn-float centre-pick" onClick={() => { const c = mapRef.current?.getCenter(); if (c) pick([c.lng, c.lat]); }} aria-label="Inspect the water at the centre of the map">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M8 0v4M8 12v4M0 8h4M12 8h4" stroke="currentColor" strokeWidth="1.4" /></svg>
            {!mobile && "Centre"}
          </button>
        </div>

        <div className="bottombar" style={mobile ? { bottom: sheetH + 76 } : undefined} data-hidden={(mobile && snap !== "peek") || undefined}>
          {mobile && layerSwitch}
          {fieldControls}
          <TimeBar compact={mobile} show={show} leads={leads} lead={lead} onLead={setLead} issued={run?.issued_date} freshness={fresh} satellite={satLayer} loading={!m} />
          {show === "forecast" && grid.status === "loading" && <div className="progress-line" role="progressbar" aria-label="Loading the forecast run" />}
          {failedForecast && show === "forecast" && <p className="inline-error t-label" role="status"><span className="mark mark-gap" aria-hidden="true" /> NOAA's forecast could not be loaded. Nothing is drawn in its place. <button className="btn btn-quiet" onClick={() => location.reload()}>Try again</button></p>}
        </div>
        {!mobile && legend && <div className="legend-wrap">{legend}</div>}

        {tiles === "failed" && <p className="tiles-note t-micro" role="status">Street map unavailable — the coast is drawn from CoastWatch's bundled outline. All data layers still work.</p>}
        {network}
        {caption && <p className="caption enter" role="status">{caption} <button className="btn btn-quiet" onClick={() => setCaption(null)}>OK</button></p>}
        {travelling && <p className="travel enter-fade t-micro" role="status">Moving along the coast to {travelling} — touch the map to stop</p>}

        {askPurpose && !mobile && <PurposeCard onChoose={choosePurpose} onSkip={() => { setAskPurpose(false); store.set("cwsig-purpose", "skipped"); }} />}
        {tour != null && (
          <section className="tourcard enter" aria-live="polite">
            <p className="t-micro">Stop {tour + 1} of {TOUR.length}</p>
            <h2 className="t-headline">{TOUR[tour].title}</h2>
            <p className="t-body">{TOUR[tour].text}</p>
            <div className="tour-actions">
              <button className="btn btn-quiet" onClick={() => setTour(null)}>End tour</button>
              {tour < TOUR.length - 1 ? <button className="btn btn-primary" onClick={() => setTour(tour + 1)}>Next stop →</button> : <button className="btn btn-primary" onClick={() => setTour(null)}>Explore on my own</button>}
            </div>
          </section>
        )}

        {mobile && (
          <Sheet snap={snap} onSnap={setSnap} onHeight={setSheetH} bottom={64} label="Your waters"
            peek={askPurpose ? 340 : reading ? 168 : 132}
            header={<div className="sheet-head">{head}{askPurpose ? null : summary}</div>}>
            {askPurpose ? <PurposeCard onChoose={choosePurpose} onSkip={() => { setAskPurpose(false); store.set("cwsig-purpose", "skipped"); }} /> : body}
          </Sheet>
        )}
      </main>
      {mobile && (
        <nav className="tabbar" aria-label="Primary">
          {["Map", "My Coast", "Blooms", "Fisheries", "Notices"].map((n) => <a key={n} href={n === "Map" ? "#/explorer" : n === "Blooms" ? "#/station" : "#/"} aria-current={n === "Map" ? "page" : undefined}><TabIcon name={n} />{n}</a>)}
        </nav>
      )}
      <p className="sr-only" aria-live="polite">{reading ? `Selected ${reading.place.label}.` : ""}</p>
      {void purpose}
      {void TODAY}
    </div>
  );
}

/** Level 1: one sentence that answers "what applies, what's known" — each clause led by its mark. */
function OneLine({ r, species, showForecast }: { r: Reading; species: string | null; showForecast: boolean }) {
  const covers = r.notices.filter((n) => n.tier === "geographic" && (!species || n.record.species.some((s) => s.includes(species) || species.includes(s)))).length;
  const may = r.notices.filter((n) => n.tier === "unresolved").length;
  const st = r.station;
  return (
    <p className="oneline">
      <span className="ol-part"><span className="mark mark-rule" aria-hidden="true" />{covers ? `${covers} official ${covers === 1 ? "notice covers" : "notices cover"} these waters` : may ? `${may} ${may === 1 ? "notice" : "notices"} may apply` : "No notice's limits include this point"}.</span>{" "}
      {st && <span className="ol-part"><span className="mark mark-sample" aria-hidden="true" />{st.tooFar ? "No recent sampling nearby" : `Nearest shore sample ${st.ageDays === 0 ? "today" : st.ageDays! < 60 ? `${st.ageDays} days ago` : "months ago"}, ${Math.round(st.km)} km away`}.</span>}{" "}
      {showForecast && r.model.status === "value" && <span className="ol-part"><span className="mark mark-field" aria-hidden="true" />Forecast {probWords(r.model.value)}.</span>}
      {showForecast && r.model.status === "gap" && <span className="ol-part"><span className="mark mark-gap" aria-hidden="true" />No forecast for this cell.</span>}
    </p>
  );
}

function ResearchTools({ variable, setVariable, fieldMode, setFieldMode, m, reading }: {
  variable: string; setVariable: (v: string) => void; fieldMode: FieldMode; setFieldMode: (f: FieldMode) => void; m?: import("../data/types").Manifest; reading: Reading | null;
}) {
  const leadsFor = (m?.layers ?? []).filter((l) => l.group_id === "charm" && l.variable === variable).sort((a, b) => (a.time.lead_days ?? 0) - (b.time.lead_days ?? 0));
  return (
    <section className="rtools" aria-label="Research tools">
      <h2 className="t-label rtools-h">Model layer</h2>
      <div className="rtools-vars" role="radiogroup" aria-label="C-HARM variable">
        {VARS.map((v) => (
          <button key={v.id} role="radio" aria-checked={variable === v.id} onClick={() => setVariable(v.id)} className="rvar">
            <span className="t-ui">{v.label}</span><span className="t-meta">{v.sub}</span>
          </button>
        ))}
      </div>
      <div className="fieldmode" role="radiogroup" aria-label="Field drawing">
        {([["veil", "Field"], ["lens", "Lens"], ["edges", "Class lines"]] as const).map(([id, l]) => <button key={id} role="radio" aria-checked={fieldMode === id} onClick={() => setFieldMode(id)}>{l}</button>)}
      </div>
      {reading && <LeadSeries layers={leadsFor} at={reading.point} />}
    </section>
  );
}

/** Exact values of the selected cell for each lead day of the one run (no interpolation). */
function LeadSeries({ layers, at }: { layers: Layer[]; at: LngLat }) {
  const g0 = useGrid(layers[0]), g1 = useGrid(layers[1]), g2 = useGrid(layers[2]), g3 = useGrid(layers[3]);
  const gs = [g0, g1, g2, g3];
  return (
    <table className="leadtable">
      <caption className="t-micro">This cell in each day of the same run</caption>
      <thead><tr><th scope="col">For</th><th scope="col">Value</th><th scope="col">Class</th></tr></thead>
      <tbody>
        {layers.map((l, i) => {
          const g = gs[i].data;
          const c = g ? cellValue(g, at) : undefined;
          return (
            <tr key={l.layer_id}>
              <th scope="row">{l.time.valid_date}</th>
              <td className="num">{c === undefined ? "…" : c == null ? <span className="t-meta">no value</span> : c.toFixed(4)}</td>
              <td>{c == null ? "—" : <span className="cls" style={{ background: `var(--cw-p${Math.min(9, Math.floor(c * 10))})` }}>{Math.min(9, Math.floor(c * 10)) * 10}–{Math.min(9, Math.floor(c * 10)) * 10 + 10} %</span>}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
function cellValue(g: import("../data/store").Grid, at: LngLat) {
  const s = g.spec;
  const c = Math.round((at[0] - s.lon_first) / s.lon_step), r = Math.round((at[1] - s.lat_first) / s.lat_step);
  if (c < 0 || r < 0 || c >= s.width || r >= s.height) return null;
  const v = g.values[r * s.width + c];
  return Number.isNaN(v) ? null : v;
}

function placeLabel(_p: LngLat, np: ReturnType<typeof nearestPort>) {
  if (!np) return "Open water";
  const d = np.km;
  const name = np.port.display_name;
  if (d < 3) return `${name} waters`;
  if (d < 25) return `Off ${name}`;
  return `${Math.round(d)} km from ${name}`;
}
/** A port's waters: a point a few km offshore (west) — ports are on land. */
function offshore(at: LngLat, kmOut: number): LngLat {
  return [at[0] - kmOut / (111.32 * Math.cos((at[1] * Math.PI) / 180)), at[1]];
}

function TabIcon({ name }: { name: string }) {
  const c = { width: 22, height: 22, viewBox: "0 0 22 22", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (name === "Map") return <svg {...c}><path d="M3 6l5-2 6 2 5-2v12l-5 2-6-2-5 2z" /><path d="M8 4v12M14 6v12" /></svg>;
  if (name === "My Coast") return <svg {...c}><path d="M11 19s6-5.2 6-10a6 6 0 0 0-12 0c0 4.8 6 10 6 10z" /><circle cx="11" cy="9" r="2.2" /></svg>;
  if (name === "Blooms") return <svg {...c}><circle cx="6" cy="14" r="2" /><circle cx="11" cy="8" r="2" /><circle cx="16" cy="12" r="2" /><path d="M3 19h16" /></svg>;
  if (name === "Fisheries") return <svg {...c}><path d="M3 11c3-4 8-5 12-2l4-3v10l-4-3c-4 3-9 2-12-2z" /><circle cx="13" cy="10" r="0.6" fill="currentColor" /></svg>;
  return <svg {...c}><path d="M4 7h14M4 11h14M4 15h9" stroke="#d9922e" /></svg>;
}

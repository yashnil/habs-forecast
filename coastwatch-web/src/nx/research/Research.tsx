"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { resolvePlace } from "@/lib/coast";
import { Brush } from "@/nx/ds/series/Brush";
import { Button } from "@/nx/ds/Button";
import { day, dayTime, age, latLon, range } from "@/nx/ds/format";
import { MarginRow, MarginRows } from "@/nx/ds/Margin";
import { Segmented } from "@/nx/ds/worksheet/Inputs";
import { Stamp, type Provenance } from "@/nx/ds/Stamp";
import { Unavailable } from "@/nx/ds/States";
import { useMedia } from "@/nx/ds/useMedia";
import { ChartScene, boundsAround, type ChartScene as Scene } from "@/nx/chart/scene";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import { download, svgFigure } from "./dom";
import { citation, modelCsv, windowCsv } from "./exports";
import {
  buildStrips,
  chooseStation,
  clampWin,
  dayKey,
  DEFAULT_STATION,
  extentOf,
  located,
  LABEL,
  modelVarFor,
  nearestStation,
  northToSouth,
  openingWindow,
  parseQuery,
  presetOf,
  presetWindow,
  queryFor,
  RANGES,
  valueWords,
  windowStats,
  inWin,
  type RangeId,
} from "./model";
import { RecordStrips, LEFT, layoutStrips, type RecordStripsHandle } from "./RecordStrips";
import { SampleLog } from "./SampleLog";
import { StationMarks } from "./StationMarks";
import { useSettledWindow } from "./useSettledWindow";
import { CHARM_ASSESSMENT, CHARM_SHORE, MODEL_WORDS, modelPlace } from "./words";
import type { ObsVariableId } from "@/lib/observations";
import s from "./research.module.css";

const NativeGrid = dynamic(() => import("./grid/NativeGrid").then((m) => m.NativeGrid), { ssr: false, loading: () => <p className={s.quiet}>Opening the grid…</p> });

export type ResearchProps = {
  observations: ObservationDataset;
  ports: PortsCollection | null;
  manifest: Manifest;
  official: OfficialDataset | null;
  baseUrl: string;
  /** the clock, ISO (the server's; CW_FIXED_NOW in tests) */
  now: string;
};

/** Desktop wide enough for the record beside the chart; anything narrower or shorter reads it in the margin. */
const WIDE = "(min-width: 1180px) and (min-height: 620px)";

/** The address, rewritten in place (no navigation, no scroll): the view is always shareable. */
function writeAddress(qs: string) {
  const url = `${window.location.pathname}?${qs}${window.location.hash}`;
  if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) window.history.replaceState(null, "", url);
}

export function Research({ observations, ports, manifest, official, baseUrl, now }: ResearchProps) {
  const sp = useSearchParams();
  const nowMs = useMemo(() => Date.parse(now), [now]);
  const [hash, setHash] = useState("");
  useEffect(() => setHash(window.location.hash), []);
  const query = useMemo(() => parseQuery(sp, hash), [sp, hash]);
  const wide = useMedia(WIDE);

  // ------------------------------------------------------------ the station
  const stations = useMemo(() => northToSouth(located(observations)), [observations]);
  const { port, key: portKey, fromUrl } = useCurrentPort();
  useEffect(() => {
    if (fromUrl && portKey) savePort(portKey);
  }, [fromUrl, portKey]);
  const place = useMemo(() => resolvePlace(query.place, ports), [query.place, ports]);
  const choice = useMemo(() => chooseStation(stations, query, place, port), [stations, query, place, port]);
  const st = choice.station;

  // ------------------------------------------------------------ the quantity in focus and the strips
  const measuredVars = useMemo(() => buildStrips(st, "pDA").filter((x) => x.kind === "measured").map((x) => x.v as ObsVariableId), [st]);
  const focus: ObsVariableId = query.v && measuredVars.includes(query.v) ? query.v : (measuredVars.find((v) => v === "pDA") ?? measuredVars[0] ?? "pDA");
  const askedMissing = query.v && !measuredVars.includes(query.v) ? query.v : null;
  const strips = useMemo(() => buildStrips(st, focus), [st, focus]);
  const focusStrip = strips.find((x) => x.v === focus) ?? strips[0];
  const modelStrip = strips.find((x) => x.kind === "model") ?? null;
  const extent = useMemo(() => extentOf(strips, nowMs), [strips, nowMs]);

  // ------------------------------------------------------------ the window: the address holds it; a drag previews it
  const committed = useMemo<[number, number]>(
    () => openingWindow({ extent, from: query.from, to: query.to, range: query.range, day: query.day, focusTimes: focusStrip?.points.filter((p) => p.cls !== "rejected").map((p) => p.t) ?? [] }),
    [extent, query.from, query.to, query.range, query.day, focusStrip],
  );
  const [drag, setDragState] = useState<[number, number] | null>(null);
  const dragRef = useRef<[number, number] | null>(null);
  const setDrag = (w: [number, number] | null) => {
    dragRef.current = w;
    setDragState(w);
  };
  const dragging = useRef(false);
  const target = drag ?? committed;
  const win = useSettledWindow(target, drag != null);

  const commit = useCallback(
    (w: [number, number], patch: { station?: string; v?: ObsVariableId; day?: number | null } = {}) => {
      const cw = clampWin(w, extent);
      writeAddress(queryFor({ station: patch.station ?? st.station_id, v: patch.v ?? focus, win: cw, place: query.place, back: query.back, day: patch.day === undefined ? query.day : patch.day }));
    },
    [extent, st.station_id, focus, query.place, query.back, query.day],
  );
  // the address is left as it arrived (a tab's `?place=`, a v11 link) until the reader chooses something; then
  // it is rewritten in v12's terms. The citation and the export always name the full view.

  const onBrush = (w: [number, number]) => {
    if (dragging.current) setDrag(w);
    else commit(w);
  };
  useEffect(() => {
    const up = () => {
      if (!dragging.current) return;
      dragging.current = false;
      const d = dragRef.current;
      if (d) commit(d);
      dragRef.current = null;
      setDragState(null);
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [commit]);
  const preset = presetOf(committed, extent);
  const choosePreset = (r: RangeId) => commit(presetWindow(r, extent));

  // ------------------------------------------------------------ the crosshair: pointer, else a chosen day, else the newest visit in view
  const [hover, setHover] = useState<number | null>(null);
  const visitsInWin = useMemo(() => {
    const ts = new Set<number>();
    for (const x of strips) if (x.kind === "measured") for (const p of x.points) if (inWin(p.t, committed)) ts.add(p.t);
    return [...ts].sort((a, b) => a - b);
  }, [strips, committed]);
  const pinned = query.day != null && inWin(query.day, committed) ? (visitsInWin.find((t) => Math.abs(t - query.day!) < 1.01 * 86_400_000) ?? query.day) : null;
  const cursor = hover ?? pinned ?? visitsInWin.at(-1) ?? null;
  const pin = (t: number) => {
    setHover(null);
    commit(committed, { day: t });
  };

  // ------------------------------------------------------------ choosing
  const chooseStationId = (id: string) => {
    setHover(null);
    const next = stations.find((x) => x.station_id === id);
    const nextVars = next ? buildStrips(next, focus).filter((x) => x.kind === "measured").map((x) => x.v) : [];
    commit(committed, { station: id, v: nextVars.includes(focus) ? focus : undefined, day: null });
  };
  const chooseFocus = (id: string) => commit(committed, { v: id as ObsVariableId });

  // ------------------------------------------------------------ the record's size (wide: fill the panel's height)
  const stripsBox = useRef<HTMLDivElement>(null);
  const [room, setRoom] = useState<number | null>(null);
  useEffect(() => {
    const el = stripsBox.current;
    if (!el || !wide) return setRoom(null);
    const ro = new ResizeObserver(([e]) => setRoom(Math.round(e.contentRect.height)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [wide]);
  // a strip with nothing in the window folds to its header line (decided on the settled window, so a drag never reflows)
  const folded = useMemo(() => new Set(strips.filter((x) => !x.points.some((p) => inWin(p.t, committed))).map((x) => x.id)), [strips, committed]);
  const plotH = useMemo(() => {
    if (!wide || room == null) return 76;
    // the tallest plots that fit the panel without scrolling; below 34 px the panel scrolls instead
    let h = 34;
    while (h < 160 && layoutStrips(strips, h + 2, folded).height <= room - 4) h += 2;
    return h;
  }, [wide, room, strips, folded]);

  // ------------------------------------------------------------ the chart: the station in the free part of the sheet, its neighbours around it
  const panelRef = useRef<HTMLElement>(null);
  const [frame, setFrame] = useState<{ W: number; H: number; free: number } | null>(null);
  useEffect(() => {
    if (!wide) return setFrame(null);
    const fr = document.querySelector<HTMLElement>(".n12-frame");
    const pn = panelRef.current;
    if (!fr || !pn) return;
    const measure = () => {
      const a = fr.getBoundingClientRect();
      const b = pn.getBoundingClientRect();
      setFrame((o) => {
        const n = { W: Math.round(a.width), H: Math.round(a.height), free: Math.round(Math.max(0, b.left - a.left)) };
        return o && o.W === n.W && o.H === n.H && o.free === n.free ? o : n;
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(fr);
    ro.observe(pn);
    return () => ro.disconnect();
  }, [wide]);
  const homeStation = port ? nearestStation(stations, port.lat, port.lon, port.key)?.station.station_id : DEFAULT_STATION;
  const sceneBase = port ? `research:${port.key}` : "research";
  const scene: Scene = useMemo(() => {
    const id = st.station_id === homeStation ? sceneBase : `${sceneBase}:${st.station_id}`;
    const focusPort = port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null;
    if (frame && frame.free > 120) {
      const pad = frame.W < 560 ? 20 : 44;
      const K = 70;
      const k = (2 * K) / Math.max(1, frame.H - 2 * pad);
      const kmLon = 111.32 * Math.cos((st.lat * Math.PI) / 180);
      const xs = Math.max(pad + 10, frame.free * 0.6);
      return {
        id,
        focus: focusPort,
        view: {
          bounds: [
            [st.lon - ((xs - pad) * k) / kmLon, st.lat - K / 111.32],
            [st.lon + ((frame.W - pad - xs) * k) / kmLon, st.lat + K / 111.32],
          ],
          maxZoom: 12,
        },
      };
    }
    return { id, focus: focusPort, view: { bounds: boundsAround([st.lon, st.lat], 32, 24), maxZoom: 12 } };
  }, [st, frame, port, sceneBase, homeStation]);

  // ------------------------------------------------------------ exports
  const figureRef = useRef<RecordStripsHandle>(null);
  const [origin, setOrigin] = useState("");
  useEffect(() => setOrigin(window.location.origin), []);
  const shareHref = `${origin}/research?${queryFor({ station: st.station_id, v: focus, win: committed })}`;
  const cite = citation(observations, st, strips, committed, shareHref);
  const [copied, setCopied] = useState(false);
  const fileBase = `neritic-${st.station_id}-${dayKey(committed[0])}-${dayKey(committed[1])}`;
  const exportCsv = () => download(`${fileBase}.csv`, windowCsv(observations, st, strips, committed), "text/csv");
  const exportModel = () => modelStrip && download(`${fileBase}-charm-${modelVarFor(focus)}.csv`, modelCsv(st, modelStrip, committed, modelVarFor(focus)), "text/csv");
  const exportSvg = () => {
    const el = figureRef.current?.svg();
    if (el) download(`${fileBase}.svg`, svgFigure(el, { title: `${st.name} (${st.station_id}), ${range(committed[0], committed[1], { year: true })}`, strips, plotH, folded, caption: cite }), "image/svg+xml");
  };
  const copyCitation = async () => {
    try {
      await navigator.clipboard.writeText(cite);
      setCopied(true);
      setTimeout(() => setCopied(false), 2400);
    } catch {
      setCopied(false);
    }
  };

  // ------------------------------------------------------------ words
  const first = st.sample_times[0];
  const last = st.sample_times.at(-1);
  const windowSamples = strips.filter((x) => x.kind === "measured").reduce((a, x) => a + x.points.filter((p) => inWin(p.t, committed)).length, 0);
  const stats = windowStats(focusStrip?.points ?? [], committed);
  const qcPassed = st.qc.filter((q) => q.passed).length;
  const vmeta = observations.variables.find((v) => v.id === focus);
  const provenance: Provenance = {
    kind: "measured",
    title: `Water samples, ${st.name}`,
    source: "CalHABMAP via SCCOOS",
    sourceLong: `${observations.provenance.source_name}${observations.provenance.institution ? `; ${observations.provenance.institution}` : ""}`,
    sourceUrl: st.source_url,
    date: last ?? observations.generated_at,
    dateRole: "sampled",
    retrieved: st.retrieved_at ?? observations.provenance.retrieved_at,
    place: `${latLon(st.lat, st.lon)}${st.nearest_port_name ? `, ${kmWords(st.nearest_port_km ?? 0)} from ${st.nearest_port_name}` : ""}`,
    method: vmeta?.method,
    caveats: observations.caveats,
  };
  const radius = st.charm?.radius_km ?? 15;
  const openSection = query.section;

  const legend = legendFor(strips, committed);

  return (
    <section className={s.page} data-testid="nx-research" data-station={st.station_id} data-layout={wide ? "wide" : "inline"}>
      <ChartScene scene={scene} />
      <StationMarks stations={stations.map((x) => ({ id: x.station_id, name: x.name, lat: x.lat, lon: x.lon }))} selected={st.station_id} radiusKm={modelStrip ? radius : null} onPick={chooseStationId} leanLeft={wide} />

      <p className="n12-kicker">Research</p>
      <h1 className={s.title} data-testid="research-title">
        {st.name}
      </h1>
      <p className={s.lead}>
        Every water sample here since {first ? day(first, { year: true }) : "the first"}, with C-HARM&rsquo;s model beside it. The window under the strips sets what the log and the export hold.
      </p>
      {choice.arrival || askedMissing || query.askedVar || query.back ? (
        <p className={s.arrived} data-testid="research-arrived">
          {choice.arrival ? (
            choice.arrival.near ? (
              <span>
                Nearest station to {choice.arrival.name}, {kmWords(choice.arrival.km)} away.
              </span>
            ) : (
              <span>
                No station is near {choice.arrival.name}: the nearest is {kmWords(choice.arrival.km)} away, so this record says nothing about that water.
              </span>
            )
          ) : null}
          {askedMissing ? <span> {LABEL[askedMissing]} isn&rsquo;t measured at {st.name}; the strips show what is.</span> : null}
          {query.askedVar ? <span> &ldquo;{query.askedVar}&rdquo; isn&rsquo;t a quantity a shore station measures; the strips show what this one does.</span> : null}
          {query.back ? (
            <Link href={query.back} className={s.back} data-testid="research-back">
              Back
            </Link>
          ) : null}
        </p>
      ) : null}
      <div className={s.stamp}>
        <Stamp provenance={provenance} now={now} />
      </div>

      {/* the record: in the margin on a phone; on a wide screen, a panel on the sheet beside the chart */}
      <section ref={panelRef} className={s.panel} aria-label={`The record at ${st.name}`} data-testid="research-record">
        <header className={s.panelHead}>
          <p className={s.windowLine} data-testid="research-window">
            <span className={s.windowDates}>{range(win[0], win[1], { year: true })}</span>
          </p>
          <Segmented<RangeId | "custom"> label="Window" hideLabel className={s.presets} value={preset ?? "custom"} onChange={(r) => r !== "custom" && choosePreset(r)} options={RANGES.map((r) => ({ value: r.id, label: r.label }))} />
        </header>
        <div ref={stripsBox} className={s.stripsBox}>
          {strips.length ? (
            <RecordStrips ref={figureRef} strips={strips} win={win} cursor={cursor} onCursor={setHover} focus={focus} onFocus={chooseFocus} plotH={plotH} folded={folded} inkKey={st.station_id} now={nowMs} />
          ) : (
            <Unavailable what={`${st.name} has published no samples`} why={st.error ?? "The station's last request returned nothing Neritic could read."} />
          )}
        </div>
        {strips.length ? (
          <div
            className={s.brushRow}
            onPointerDownCapture={() => {
              dragging.current = true;
            }}
            data-testid="research-brush"
          >
            <span className={s.brushLabel} style={{ width: LEFT - 12 }}>
              Window
            </span>
            <Brush extent={extent} value={win} onChange={onBrush} points={(focusStrip?.points ?? []).filter((p) => p.cls === "value").map((p) => ({ t: p.time, v: p.v }))} scale={focusStrip?.scale ?? "log"} label="every strip" height={44} />
          </div>
        ) : null}
        <div className={s.foot}>
          <ul className={s.legend} aria-label="Legend">
            {legend.map((l) => (
              <li key={l.id}>
                <LegendGlyph id={l.id} />
                <span>{l.words}</span>
              </li>
            ))}
          </ul>
          <p className={s.attrib}>Chart: depth NOAA ETOPO, CRM · coast © OpenStreetMap</p>
        </div>
      </section>

      <div className={s.primary}>
        <Button variant="primary" icon="share" onClick={exportCsv} full aria-describedby="rx-export-what">
          Export the window as CSV
        </Button>
        <p id="rx-export-what" className={s.small}>
          {windowSamples.toLocaleString("en-US")} samples, {range(committed[0], committed[1], { year: true })}, every quantity, with source columns.
        </p>
      </div>

      <MarginRows label="The record's details" className={s.rows}>
        <MarginRow
          key={`station-${openSection}`}
          label="Station"
          defaultOpen={openSection === "station"}
          value={<span data-testid="row-station">{st.nearest_port_name ? `${kmWords(st.nearest_port_km ?? 0)} from ${st.nearest_port_name}` : latLon(st.lat, st.lon)}</span>}
          detail={
            <nav aria-label="Shore stations, north to south" className={s.bleed}>
              <ol className={s.stationList} data-testid="research-stations">
                {stations.map((x) => (
                  <li key={x.station_id}>
                    <button type="button" aria-current={x.station_id === st.station_id ? "true" : undefined} onClick={() => chooseStationId(x.station_id)} data-testid={`list-station-${x.station_id}`}>
                      <span className={s.stationListName}>{x.name}</span>
                      <span className={s.quiet}>{x.sample_times.at(-1) ? day(x.sample_times.at(-1)!, { ref: now }) : "no samples"}</span>
                    </button>
                  </li>
                ))}
              </ol>
              <p className={s.small}>North to south, with each station&rsquo;s newest sample. The chart shows the same stations where they sample.</p>
              <p className={s.links}>
                <Link href={`/explore?sel=station:${encodeURIComponent(st.station_id)}`} data-testid="research-explore">
                  {st.name} in Explore
                </Link>
                {st.nearest_port_code != null ? (
                  <Link href={`/coast?place=port-${st.nearest_port_code}`} data-testid="research-coast">
                    Plan a trip from {st.nearest_port_name}
                  </Link>
                ) : null}
              </p>
            </nav>
          }
        />
        <MarginRow
          key={`window-${openSection}`}
          label="Window"
          defaultOpen={openSection === "window"}
          value={<span data-testid="row-window">{stats.max ? `Highest ${valueWords(focusStrip, stats.max)}, ${day(stats.max.t, { ref: now })}` : `No ${shortLabel(focus)} value`}</span>}
          detail={
            <div className={s.detailText} data-testid="research-stats">
              <dl className={s.facts}>
                <div>
                  <dt>Highest</dt>
                  <dd>{stats.max ? `${valueWords(focusStrip, stats.max)}, ${day(stats.max.t, { year: true })}` : "none in the window"}</dd>
                </div>
                <div>
                  <dt>Median</dt>
                  <dd>{stats.median != null ? valueWords(focusStrip, { ...stats.max!, v: stats.median, flagged: false }) : "none"}</dd>
                </div>
                <div>
                  <dt>Reported 0</dt>
                  <dd>{stats.zeros}</dd>
                </div>
                {stats.rejected ? (
                  <div>
                    <dt>Rejected</dt>
                    <dd>{stats.rejected}, by QC: no value, not counted</dd>
                  </div>
                ) : null}
                {stats.flagged ? (
                  <div>
                    <dt>Flagged</dt>
                    <dd>{stats.flagged} high, kept as published</dd>
                  </div>
                ) : null}
              </dl>
              <p className={s.small}>Of the samples of {LABEL[focus].toLowerCase()} in the window. A visit that did not measure it is not counted and is never a zero. Choose another quantity by its name on the record.</p>
            </div>
          }
        />
        <MarginRow
          key={`samples-${openSection}`}
          label="Samples"
          defaultOpen={openSection === "samples"}
          value={<span data-testid="row-samples">{(focusStrip?.points.filter((p) => inWin(p.t, committed)).length ?? 0).toLocaleString("en-US")} dated, {focusStrip ? shortLabel(focusStrip.v) : ""}</span>}
          detail={focusStrip ? <div className={s.bleed}><SampleLog strip={focusStrip} win={committed} cursor={cursor} onCursor={pin} /></div> : null}
        />
        <MarginRow
          key={`checks-${openSection}`}
          label="Checks"
          defaultOpen={openSection === "checks"}
          value={<span data-testid="row-checks">{qcPassed} of {st.qc.length} passed</span>}
          detail={
            <div className={s.detailText}>
              <ul className={s.checks} data-testid="research-qc">
                {st.qc.map((q) => (
                  <li key={q.name} data-passed={q.passed ? "1" : "0"}>
                    <span className={s.checkWord}>{q.passed ? "Passed" : "Failed"}</span>
                    <span>
                      <b>{checkName(q.name)}.</b> {q.detail}
                    </span>
                  </li>
                ))}
              </ul>
              {observations.method.qc ? <p className={s.small}>{observations.method.qc}</p> : null}
              {st.status !== "updated" ? <p className={s.small}>Last request: {st.status.replace(/_/g, " ")}{st.error ? `. ${st.error}` : ""}</p> : null}
            </div>
          }
        />
        <MarginRow
          key={`source-${openSection}`}
          label="Source"
          defaultOpen={openSection === "source"}
          value={<span>CalHABMAP, read {st.retrieved_at ? day(st.retrieved_at, { ref: now }) : "—"}</span>}
          detail={
            <div className={s.detailText} data-testid="research-provenance">
              <dl className={s.facts}>
                <div>
                  <dt>Program</dt>
                  <dd>{observations.provenance.source_name}</dd>
                </div>
                <div>
                  <dt>Dataset</dt>
                  <dd>
                    {st.station_id}
                    {st.location_code ? `, ${st.location_code}` : ""}
                  </dd>
                </div>
                {st.request_url ? (
                  <div>
                    <dt>Request</dt>
                    <dd className={s.url}>
                      <a href={st.request_url} target="_blank" rel="noreferrer" data-testid="research-request-url">
                        {st.request_url}
                      </a>
                    </dd>
                  </div>
                ) : null}
                <div>
                  <dt>Retrieved</dt>
                  <dd>{st.retrieved_at ? `${dayTime(st.retrieved_at, { ref: now })}, ${age(st.retrieved_at, now)}` : "—"}</dd>
                </div>
                <div>
                  <dt>Method</dt>
                  <dd>{vmeta?.method}</dd>
                </div>
                <div>
                  <dt>Matrix</dt>
                  <dd>{vmeta?.matrix}{vmeta?.fraction ? `, ${vmeta.fraction}` : ""}</dd>
                </div>
                <div>
                  <dt>Zeros</dt>
                  <dd>{vmeta?.zero_policy}</dd>
                </div>
                {vmeta?.detection_limit_note ? (
                  <div>
                    <dt>Limit</dt>
                    <dd>{vmeta.detection_limit_note}</dd>
                  </div>
                ) : null}
                <div>
                  <dt>Run</dt>
                  <dd>
                    {observations.provenance.pipeline_run_id}, version {observations.provenance.pipeline_version}
                  </dd>
                </div>
                <div>
                  <dt>Licence</dt>
                  <dd className={s.small}>{observations.provenance.license}</dd>
                </div>
              </dl>
              <ul className={s.caveats}>
                {observations.caveats.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </div>
          }
        />
        <MarginRow
          key={`model-${openSection}`}
          label="Model"
          defaultOpen={openSection === "model"}
          value={<span>{modelStrip ? `C-HARM, ${radius} km median` : "No C-HARM history here"}</span>}
          detail={
            !modelStrip ? (
              <div className={s.detailText} data-testid="model-panel-none">
                <p>No C-HARM nowcast history is published for this station, so the record has no model strip.</p>
                {st.charm?.error ? <p className={`${s.small} ${s.url}`}>What the pipeline reported: {st.charm.error}</p> : null}
              </div>
            ) : (
              <div className={s.detailText} data-testid="research-model">
                <p>
                  {modelStrip.label}. {modelPlace(radius)} {MODEL_WORDS}
                </p>
                <p>
                  Neritic keeps each day&rsquo;s nowcast, from {modelStrip.historyStart ? day(modelStrip.historyStart, { year: true }) : "—"}; open circles are days with fewer than 60 % of the usual cells, not joined.
                </p>
                <p className={s.small}>{CHARM_ASSESSMENT}</p>
                <p className={s.small}>{CHARM_SHORE}</p>
              </div>
            )
          }
        />
        <MarginRow
          key={`export-${openSection}`}
          label="Export"
          defaultOpen={openSection === "export"}
          value={<span>CSV, model, figure, citation</span>}
          detail={
            <div className={s.detailText} data-testid="research-exports">
              <div className={s.exportButtons}>
                <Button variant="secondary" onClick={exportCsv} full>
                  Samples in the window (CSV)
                </Button>
                {modelStrip ? (
                  <Button variant="secondary" onClick={exportModel} full>
                    Model days in the window (CSV)
                  </Button>
                ) : null}
                <Button variant="secondary" onClick={exportSvg} full>
                  The strips as a figure (SVG)
                </Button>
                {st.request_url ? (
                  <Button variant="text" href={st.request_url} external>
                    The station&rsquo;s own file at SCCOOS
                  </Button>
                ) : null}
              </div>
              <figure className={s.cite}>
                <figcaption>Cite this window</figcaption>
                <blockquote data-testid="research-citation">{cite}</blockquote>
                <Button variant="text" onClick={copyCitation}>
                  {copied ? "Copied" : "Copy the citation"}
                </Button>
                <span className={s.vh} aria-live="polite">
                  {copied ? "Citation copied" : ""}
                </span>
              </figure>
              <p className={s.small}>Every file opens with lines starting # that say what it holds, its source, request, licence and pipeline run. An empty value is never a zero.</p>
            </div>
          }
        />
        <MarginRow
          key={`grid-${openSection}`}
          label="Grid"
          defaultOpen={openSection === "grid"}
          value={<span>Cells, transects, areas</span>}
          detail={<GridOpener manifest={manifest} ports={ports} official={official} baseUrl={baseUrl} station={st} now={now} />}
        />
        <MarginRow
          label="Landings"
          value={<span>Statewide, CDFW</span>}
          detail={
            <div className={s.detailText}>
              <p>Commercial landings and their value by species, statewide, from CDFW. They have no place on the coast, so they are not on the chart.</p>
              <Button variant="text" href="/fisheries">
                Open the landings history
              </Button>
            </div>
          }
        />
      </MarginRows>
      <p className={s.imprint}>
        Dataset generated {day(observations.generated_at, { ref: now })}. Water samples, not seafood tests. Not for navigation.
      </p>
      <span hidden data-testid="research-window-ms" data-from={committed[0]} data-to={committed[1]} />
    </section>
  );
}

function GridOpener(props: { manifest: Manifest; ports: PortsCollection | null; official: OfficialDataset | null; baseUrl: string; station: { station_id: string; name: string; lat: number; lon: number }; now: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={s.detailText}>
      <p>The published C-HARM, satellite and HF-radar grids near {props.station.name}, cell by cell: read a cell, draw a transect or count an area, and export each as CSV.</p>
      <Button variant="secondary" onClick={() => setOpen(true)} full>
        Open the native grid
      </Button>
      {open ? <NativeGrid {...props} open={open} onOpenChange={setOpen} /> : null}
    </div>
  );
}

const kmWords = (km: number) => (km < 1 ? `${Math.round(km * 1000)} m` : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`);

function shortLabel(v: ObsVariableId | null): string {
  if (!v) return "";
  return { pDA: "pDA", dDA: "dDA", tDA: "tDA", pn_seriata: "seriata group", pn_delicatissima: "delicatissima group", chl_extracted: "chlorophyll", temp: "temperature" }[v];
}

function checkName(n: string): string {
  return { units: "Units", duplicates: "Duplicates", time: "Sample times" }[n] ?? n.charAt(0).toUpperCase() + n.slice(1);
}

type LegendId = "measured" | "zero" | "rejected" | "flagged" | "model" | "partial" | "threshold" | "now";

/** One legend for the view: only the marks the window actually shows. */
function legendFor(strips: ReturnType<typeof buildStrips>, win: [number, number]): { id: LegendId; words: string }[] {
  const pts = strips.filter((x) => x.kind === "measured").flatMap((x) => x.points.filter((p) => inWin(p.t, win)));
  const m = strips.find((x) => x.kind === "model");
  const out: { id: LegendId; words: string }[] = [{ id: "measured", words: "Measured" }];
  if (pts.some((p) => p.cls === "zero")) out.push({ id: "zero", words: "Reported 0" });
  if (pts.some((p) => p.cls === "rejected")) out.push({ id: "rejected", words: "Rejected by QC" });
  if (pts.some((p) => p.flagged)) out.push({ id: "flagged", words: "Flagged high, kept" });
  if (m) out.push({ id: "model", words: "Model, hatched where none kept" });
  if (m?.partial && [...m.partial].some((t) => inWin(t, win))) out.push({ id: "partial", words: "Model day, few cells" });
  if (strips.some((x) => x.threshold)) out.push({ id: "threshold", words: "C-HARM threshold" });
  out.push({ id: "now", words: "Today" });
  return out;
}

function LegendGlyph({ id }: { id: LegendId }) {
  return (
    <svg width="18" height="14" viewBox="0 0 18 14" aria-hidden className={s.glyph}>
      {id === "measured" ? <rect x="6" y="4" width="6" height="6" className={s.pt} /> : null}
      {id === "zero" ? <rect x="6" y="4" width="6" height="6" className={s.ptZero} /> : null}
      {id === "rejected" ? <path d="M5.5 3.5L12.5 10.5M12.5 3.5L5.5 10.5" className={s.cross} /> : null}
      {id === "flagged" ? (
        <>
          <rect x="6.5" y="4.5" width="5" height="5" className={s.pt} />
          <circle cx="9" cy="7" r="6" className={s.ring} />
        </>
      ) : null}
      {id === "model" ? (
        <>
          <path d="M1 10L5 4M5 10L9 4M9 10L13 4M13 10L17 4" className={s.hatch} />
          <rect x="6" y="4" width="6" height="6" className={s.ptHollow} />
        </>
      ) : null}
      {id === "partial" ? <circle cx="9" cy="7" r="3.5" className={s.ptHollow} /> : null}
      {id === "threshold" ? <line x1="1" x2="17" y1="7" y2="7" className={s.threshold} /> : null}
      {id === "now" ? <line x1="9" x2="9" y1="1" y2="13" className={s.now} /> : null}
    </svg>
  );
}

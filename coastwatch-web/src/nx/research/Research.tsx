"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { resolvePlace } from "@/lib/coast";
import type { ObsVariableId } from "@/lib/observations";
import {
  Brush,
  Button,
  ButtonGroup,
  ChartToggle,
  DataStateMark,
  Desk,
  DeskBody,
  DeskFoot,
  DeskHeader,
  DeskMain,
  Disclosure,
  PageLayout,
  PlotState,
  PlotSync,
  PlotTable,
  RangeControl,
  Stamp,
  Unavailable,
  useMedia,
  windowAround,
  type DataStateKind,
  type PlotSyncValue,
  type Provenance,
} from "@/nx/ds";
import { age, day, dayTime, latLon, range } from "@/nx/ds/format";
import { setChartHighlight } from "@/nx/chart/highlight";
import { ChartScene, boundsAround, tallBoundsAround, type ChartScene as Scene } from "@/nx/chart/scene";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import {
  availabilityAll,
  chooseFocus,
  copyStart,
  emptyRange,
  exportCounts,
  inspectSample,
  latestWindow,
  modelCollapsed,
  nearestWithRecent,
  stationLatest,
  stepSample,
  visitTicks,
} from "./availability";
import { download, svgFigure } from "./dom";
import { citation, modelCsv, windowCsv } from "./exports";
import { MeasuredPlot, ModelPlot } from "./FocusChart";
import { Inspector } from "./Inspector";
import {
  buildStrips,
  chooseStation,
  clampWin,
  DAY,
  dayKey,
  DEFAULT_STATION,
  extentOf,
  inWin,
  LABEL,
  located,
  logRows,
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
  type RangeId,
  type RecPoint,
} from "./model";
import { ageWords, plotPoints, type SamplePoint } from "./plotData";
import { RecordStrips, type RecordStripsHandle } from "./RecordStrips";
import { StationMarks } from "./StationMarks";
import { VariableRail, type RailRow } from "./VariableRail";
import { CHARM_ASSESSMENT, CHARM_SHORE, MODEL_WORDS, modelPlace } from "./words";
import s from "./research.module.css";
import v from "./v13.module.css";

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

/** The chart pane stands beside the desk (v13 README §1): tall and narrow, so the station's view runs along the coast. */
const WIDE = "(min-width: 1100px) and (min-height: 520px)";
/** The plot's axis room, shared with the brush so the two line up. */
const INSET = { left: 56, right: 12 };
const PRESET_LABEL: Record<RangeId, string> = { "90d": "90 d", "1y": "1 y", "3y": "3 y", all: "Whole record" };

/** What each mark on these plots is, in this page's own terms (the source publishes no detection limit, so a 0 is never "none"). */
const LEGEND_WORDS: Partial<Record<DataStateKind, string>> = {
  measured: "Measured",
  "reported-zero": "Reported 0: at or below an unpublished limit, not absent",
  missing: "Rejected by QC: no value",
  forecast: "Model: computed, not measured",
};

/** The address, rewritten in place (no navigation, no scroll): the view is always shareable. */
function writeAddress(qs: string) {
  const url = `${window.location.pathname}?${qs}${window.location.hash}`;
  if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) window.history.replaceState(null, "", url);
}

const SHORT: Record<ObsVariableId, string> = { pDA: "pDA", dDA: "dDA", tDA: "tDA", pn_seriata: "seriata group", pn_delicatissima: "delicatissima group", chl_extracted: "chlorophyll", temp: "temperature" };
const kmWords = (km: number) => (km < 1 ? `${Math.round(km * 1000)} m` : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`);
const checkName = (n: string) => ({ units: "Units", duplicates: "Duplicates", time: "Sample times" })[n] ?? n.charAt(0).toUpperCase() + n.slice(1).replace(/_/g, " ");
const count = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

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

  // ------------------------------------------------------------ the variable: the link's `var` whenever this station has ever measured it
  const { focus, askedMissing, measured } = useMemo(() => chooseFocus(st, query.v), [st, query.v]);
  const strips = useMemo(() => buildStrips(st, focus), [st, focus]);
  const focusStrip = strips.find((x) => x.v === focus) ?? null;
  const modelStrip = strips.find((x) => x.kind === "model") ?? null;
  const cmp = query.cmp && query.cmp !== focus && measured.includes(query.cmp) ? query.cmp : null;
  const cmpStrip = cmp ? (strips.find((x) => x.v === cmp) ?? null) : null;
  const extent = useMemo(() => extentOf(strips, nowMs), [strips, nowMs]);
  const focusPoints = useMemo(() => focusStrip?.points ?? [], [focusStrip]);

  // ------------------------------------------------------------ the window: the address holds it; a drag previews it
  const committed = useMemo<[number, number]>(
    () => openingWindow({ extent, from: query.from, to: query.to, range: query.range, day: query.day, focusTimes: focusPoints.filter((p) => p.cls !== "rejected").map((p) => p.t) }),
    [extent, query.from, query.to, query.range, query.day, focusPoints],
  );
  const [drag, setDragState] = useState<[number, number] | null>(null);
  const dragRef = useRef<[number, number] | null>(null);
  const setDrag = (w: [number, number] | null) => {
    dragRef.current = w;
    setDragState(w);
  };
  const dragging = useRef(false);
  // a new window is a cut (v13 README §5: data never tweens)
  const win = drag ?? committed;

  /**
   * Writes the view to the address. What is not being changed is read from the ADDRESS as it is now, not from
   * this render: the address is ahead of React for a moment after each change, and a second choice made in
   * that moment (a variable, then a comparison) must not undo the first. `w: null` keeps the window.
   */
  const commit = useCallback(
    (w: [number, number] | null, patch: { station?: string; v?: ObsVariableId; day?: number | null; cmp?: ObsVariableId | null } = {}) => {
      const cur = parseQuery(new URLSearchParams(window.location.search));
      const sameStation = cur.station == null || cur.station === st.station_id;
      const curV = sameStation && cur.v && measured.includes(cur.v) ? cur.v : focus;
      const curWin: [number, number] = cur.from != null && cur.to != null && sameStation ? [cur.from, cur.to] : committed;
      const v = patch.v ?? curV;
      const cmpNext = patch.cmp === undefined ? (cur.station != null ? cur.cmp : cmp) : patch.cmp;
      writeAddress(
        queryFor({
          station: patch.station ?? st.station_id,
          v,
          win: clampWin(w ?? curWin, extent),
          place: query.place,
          back: query.back,
          day: patch.day === undefined ? (cur.station != null ? cur.day : query.day) : patch.day,
          cmp: cmpNext === v ? null : cmpNext,
        }),
      );
    },
    [extent, st.station_id, focus, measured, committed, query.place, query.back, query.day, cmp],
  );
  // the address is left as it arrived (a tab's `?place=`, a v11 link) until the reader chooses something

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

  // ------------------------------------------------------------ per-variable truth for the window
  const availability = useMemo(() => availabilityAll(st, committed), [st, committed]);
  const avail = availability.find((a) => a.v === focus) ?? null;
  const empty = useMemo(() => (focusStrip ? emptyRange(st, focus, committed) : null), [st, focus, committed, focusStrip]);
  const latest = useMemo(() => latestWindow(focusPoints, extent), [focusPoints, extent]);
  const neighbour = useMemo(() => (empty ? nearestWithRecent(stations, st, focus, nowMs) : null), [empty, stations, st, focus, nowMs]);
  const copy = copyStart(observations, st);
  const railRows: RailRow[] = useMemo(
    () =>
      availability.map((a) => {
        const strip = strips.find((x) => x.v === a.v)!;
        return { a, unit: strip.unit, unitShort: strip.unitShort, ticks: visitTicks(st, a.v) };
      }),
    [availability, strips, st],
  );

  // ------------------------------------------------------------ the selected sample: the address's `day`, read on the selected variable
  const selected = useMemo<RecPoint | null>(() => {
    if (query.day == null) return null;
    return focusPoints.find((p) => Math.abs(p.t - query.day!) < 1.01 * DAY && inWin(p.t, committed)) ?? null;
  }, [query.day, focusPoints, committed]);
  const record = useMemo(() => (selected ? inspectSample(observations, st, focus, selected.i) : null), [selected, observations, st, focus]);
  const select = (p: RecPoint | null) => {
    if (!p) return commit(null, { day: null });
    // a sample outside the window brings the window to it (same width)
    commit(inWin(p.t, committed) ? null : windowAround(extent, committed, p.t), { day: p.t });
  };
  const prev = selected ? stepSample(focusPoints, selected.t, -1) : null;
  const next = selected ? stepSample(focusPoints, selected.t, 1) : null;
  const position = selected ? `${(focusPoints.findIndex((p) => p.i === selected.i) + 1).toLocaleString("en-US")} of ${count(focusPoints.length, `${SHORT[focus]} sample`)} in the record` : null;

  // ------------------------------------------------------------ the plots' shared crosshair; the station lit on the chart while a sample is read
  const [sync, setSync] = useState<PlotSyncValue>({ hoverT: null, pinned: null });
  const onSync = (n: PlotSyncValue) => setSync((o) => (o.hoverT === n.hoverT ? o : { hoverT: n.hoverT, pinned: null }));
  const reading = sync.hoverT != null || selected != null;
  useEffect(() => {
    setChartHighlight(reading ? { lon: st.lon, lat: st.lat } : null);
    return () => setChartHighlight(null);
  }, [reading, st.lon, st.lat]);

  const place2: [number, number] = useMemo(() => [st.lon, st.lat], [st.lon, st.lat]);
  const focusData = useMemo<SamplePoint[]>(() => (focusStrip ? plotPoints(focusStrip, place2) : []), [focusStrip, place2]);
  const cmpData = useMemo<SamplePoint[]>(() => (cmpStrip ? plotPoints(cmpStrip, place2) : []), [cmpStrip, place2]);
  const modelData = useMemo<SamplePoint[]>(() => (modelStrip ? plotPoints(modelStrip, place2) : []), [modelStrip, place2]);
  const selectedPoint = selected ? (focusData.find((p) => p.meta.i === selected.i) ?? null) : null;

  // ------------------------------------------------------------ the model panel: collapsed when the window has no measured value to read it beside
  const collapsed = modelStrip != null && modelCollapsed(focusPoints, committed);
  const [modelAnyway, setModelAnyway] = useState<string | null>(null);
  const modelKey = `${st.station_id}:${focus}`;
  const showModel = modelStrip != null && (!collapsed || modelAnyway === modelKey);

  // ------------------------------------------------------------ choosing
  const chooseStationId = (id: string, w: [number, number] | null = null) => {
    const nextSt = stations.find((x) => x.station_id === id);
    const nextVars = nextSt ? chooseFocus(nextSt, focus).measured : [];
    commit(w, { station: id, v: nextVars.includes(focus) ? focus : undefined, day: null, cmp: null });
  };
  const chooseVariable = (id: ObsVariableId) => commit(null, { v: id, day: null });
  const choosePreset = (r: RangeId) => commit(presetWindow(r, extent));
  const jumpLatest = () => latest && commit(latest.win, { day: latest.day });
  const openNeighbour = () => {
    if (!neighbour) return;
    // open the neighbour on a year that holds its newest sample of this variable
    const w: [number, number] = [neighbour.last ? Date.parse(neighbour.last.time) - 335 * DAY : committed[0], neighbour.last ? Date.parse(neighbour.last.time) + 30 * DAY : committed[1]];
    writeAddress(queryFor({ station: neighbour.station.station_id, v: focus, win: w, place: query.place, back: query.back }));
  };

  // ------------------------------------------------------------ the chart pane: the station, its neighbours up and down the coast
  const homeStation = port ? nearestStation(stations, port.lat, port.lon, port.key)?.station.station_id : DEFAULT_STATION;
  const sceneBase = port ? `research:${port.key}` : "research";
  const scene: Scene = useMemo(() => {
    const id = st.station_id === homeStation ? sceneBase : `${sceneBase}:${st.station_id}`;
    const focusPort = port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null;
    // the pane beside the desk is tall and narrow: 70 km up and down the coast, the station centred
    return {
      id,
      focus: focusPort,
      // the pane beside the desk is tall and narrow: 70 km up and down the coast when it is, a wider box when not
      view: { bounds: boundsAround([st.lon, st.lat], 32, 24), tall: tallBoundsAround([st.lon, st.lat], 70, 14), maxZoom: 12 },
      named: stations.map((x) => ({ name: x.name, lon: x.lon, lat: x.lat })),
    };
  }, [st, port, sceneBase, homeStation, stations]);

  // ------------------------------------------------------------ exports
  const figureRef = useRef<RecordStripsHandle>(null);
  const [stripsOpen, setStripsOpen] = useState(false);
  const folded = useMemo(() => new Set(strips.filter((x) => !x.points.some((p) => inWin(p.t, committed))).map((x) => x.id)), [strips, committed]);
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
    if (el) download(`${fileBase}.svg`, svgFigure(el, { title: `${st.name} (${st.station_id}), ${range(committed[0], committed[1], { year: true })}`, strips, plotH: 76, folded, caption: cite }), "image/svg+xml");
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
  const counts = exportCounts(strips, committed);
  const stats = windowStats(focusPoints, committed);
  const qcPassed = st.qc.filter((q) => q.passed).length;
  const vmeta = observations.variables.find((x) => x.id === focus);
  const lastVisit = st.sample_times.at(-1) ?? null;
  const lastOf = avail?.last ?? null;
  const latestByStation = useMemo(() => new Map(stationLatest(stations, focus).map((x) => [x.station.station_id, x])), [stations, focus]);
  const provenance: Provenance = {
    kind: "measured",
    title: `${LABEL[focus]}, ${st.name}`,
    source: "CalHABMAP via SCCOOS",
    sourceLong: `${observations.provenance.source_name}${observations.provenance.institution ? `; ${observations.provenance.institution}` : ""}`,
    sourceUrl: st.source_url,
    // the newest sample of the selected variable, never the station's newest visit (which may have measured something else)
    date: lastOf?.time ?? lastVisit ?? observations.generated_at,
    dateRole: "sampled",
    retrieved: st.retrieved_at ?? observations.provenance.retrieved_at,
    place: `${latLon(st.lat, st.lon)}${st.nearest_port_name ? `, ${kmWords(st.nearest_port_km ?? 0)} from ${st.nearest_port_name}` : ""}`,
    method: vmeta?.method,
    caveats: observations.caveats,
  };
  const radius = st.charm?.radius_km ?? 15;
  const open = query.section;
  const stationSelect = useId();
  const stationSelectPhone = useId();
  const variableSelect = useId();
  const cmpSelect = useId();

  const stationPicker = (id: string, testid: string) => (
    <>
      <label htmlFor={id}>Station</label>
      <span className={v.selectWrap}>
        <select id={id} className={v.select} value={st.station_id} onChange={(e) => chooseStationId(e.target.value)} data-testid={testid}>
          {stations.map((x) => (
            <option key={x.station_id} value={x.station_id}>
              {x.name}
            </option>
          ))}
        </select>
      </span>
    </>
  );
  const insteadWords = empty?.instead.length ? `${empty.instead.map((x) => SHORT[x]).join(", ")} only` : "none of the variables shown";
  const emptyReason = empty ? (
    <>
      {empty.kind === "never"
        ? `No ${LABEL[focus].toLowerCase()} has been published for ${st.name}.`
        : empty.last
          ? `Last measured ${day(empty.last.t, { year: true })} (${ageWords(Math.floor((nowMs - empty.last.t) / DAY))}): ${valueWords(focusStrip!, empty.last)}.`
          : `First measured ${day(empty.next!.t, { year: true })}, after this window.`}{" "}
      {empty.visits ? `The ${count(empty.visits, "sampling visit")} in this window recorded ${insteadWords}.` : "The station was not visited in this window."} Not measured is not zero.
      {neighbour ? (
        <span data-testid="research-empty-neighbour">
          {" "}
          The nearest station with a recent sample is {neighbour.station.name}, {kmWords(neighbour.km)} away: {SHORT[focus]} measured {day(neighbour.last.time, { year: true })}. It is a different pier, not this water.
        </span>
      ) : null}
    </>
  ) : null;

  // the states the plots actually draw, for one legend
  const legendKinds = useMemo<DataStateKind[]>(() => {
    const k = new Set<DataStateKind>();
    for (const d of [focusData, cmpData]) for (const p of d) if (inWin(p.t, committed)) k.add(p.kind ?? "measured");
    if (showModel && modelData.some((p) => inWin(p.t, committed))) k.add("forecast");
    return (["measured", "reported-zero", "missing", "forecast"] as DataStateKind[]).filter((x) => k.has(x));
  }, [focusData, cmpData, modelData, committed, showModel]);

  const tableRows = useMemo(() => logRows(focusPoints, committed, "date", "desc"), [focusPoints, committed]);
  const depthOf = (p: RecPoint) => st.depths_m[p.i];

  const statusWords = focusStrip ? `${LABEL[focus]}: ${lastOf ? `last measured ${day(lastOf.t, { year: true })} (${ageWords(Math.floor((nowMs - lastOf.t) / DAY))}), ${valueWords(focusStrip, lastOf)}.` : "never measured here."}` : `${st.name} has published no samples Neritic can read.`;
  // on a phone the sentence is under the plot (so the plot is in the first screen); the header keeps it on a wide screen
  const status = (
    <span data-testid="research-status" className={focusStrip ? v.onlyWide : undefined}>
      {statusWords}
    </span>
  );

  return (
    <section className={s.page} data-testid="nx-research" data-station={st.station_id} data-var={focus} data-layout="desk">
      <PageLayout mode="desk" paneKey="research" defaultPane="context" />
      <ChartScene scene={scene} />
      <StationMarks stations={stations.map((x) => ({ id: x.station_id, name: x.name, lat: x.lat, lon: x.lon }))} selected={st.station_id} radiusKm={showModel ? radius : null} onPick={(id) => chooseStationId(id)} leanLeft={wide} />

      <Desk>
        <DeskHeader
          context="Research · shore-station water samples"
          title={st.name}
          titleProps={{ "data-testid": "research-title" }}
          status={status}
          actions={
            <>
              {/* in the title's row on a wide screen; on a phone the picker is under the plot, so the plot is in the first screen */}
              <div className={`${v.stationPick} ${v.onlyWide}`}>{stationPicker(stationSelect, "research-station-select")}</div>
              <ChartToggle />
            </>
          }
        >
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
              {askedMissing ? <span> {LABEL[askedMissing]} isn&rsquo;t measured at {st.name}; the list shows what is.</span> : null}
              {query.askedVar ? <span> &ldquo;{query.askedVar}&rdquo; isn&rsquo;t a quantity a shore station measures; the list shows what this one does.</span> : null}
              {query.back ? (
                <Link href={query.back} className={s.back} data-testid="research-back">
                  Back
                </Link>
              ) : null}
            </p>
          ) : null}
        </DeskHeader>

        <DeskBody layout="main-full">
          <DeskMain>
            {!focusStrip ? (
              <Unavailable what={`${st.name} has published no samples`} why={st.error ?? "The station's last request returned nothing Neritic could read."} />
            ) : (
              <>
                <div data-testid="research-record" className={v.stack}>
                <section className={v.railBlock} aria-label="Variable">
                  <VariableRail rows={railRows} selected={focus} onSelect={chooseVariable} extent={extent} win={win} now={nowMs} stationName={st.name} />
                </section>

                <section className={v.record} aria-label={`${LABEL[focus]} at ${st.name}`} data-testid="research-chart">
                    {/* a phone's compact selector: the plot is in the first screen, the full list with its ticks under it */}
                    <div className={`${v.pickCell} ${v.cmpPick}`}>
                      <label htmlFor={variableSelect}>Variable</label>
                      <span className={v.selectWrap}>
                        <select id={variableSelect} className={v.select} value={focus} onChange={(e) => chooseVariable(e.target.value as ObsVariableId)} data-testid="research-variable-select">
                          {availability.map((a) => (
                            <option key={a.v} value={a.v}>
                              {LABEL[a.v]} ({a.withValue} in window)
                            </option>
                          ))}
                        </select>
                      </span>
                    </div>
                    <div className={v.titleCell}>
                      <h2 className={v.h2} data-testid="research-variable">
                        {LABEL[focus]} <span className={v.unitWords}>{focusStrip.unit}</span>
                      </h2>
                      <p className={v.windowLine}>
                        <span data-testid="research-window">{range(win[0], win[1], { year: true })}</span>
                        {avail ? (
                          <span data-testid="research-window-count">
                            {" "}
                            · {count(avail.withValue, "sample")}
                            {avail.zeros ? `, ${avail.zeros} reported 0` : ""}
                            {avail.absent ? `; ${avail.absent} of ${count(avail.visits, "visit")} did not measure it` : ""}
                          </span>
                        ) : null}
                      </p>
                      <p className={`${v.windowLine} ${v.onlyPhone}`}>{statusWords}</p>
                    </div>
                    <div className={v.rangeCell}>
                    <RangeControl<RangeId>
                      label="Window"
                      presets={RANGES.map((r) => ({ value: r.id, label: PRESET_LABEL[r.id] }))}
                      value={preset}
                      onChange={choosePreset}
                      onJumpLatest={latest ? jumpLatest : undefined}
                      latestLabel={latest ? `Latest sample: ${day(latest.day, { year: true })}` : undefined}
                    />
                    </div>

                  <div className={v.plots}>
                    <div
                      className={v.brush}
                      onPointerDownCapture={() => {
                        dragging.current = true;
                      }}
                      data-testid="research-brush"
                    >
                      <Brush extent={extent} value={win} onChange={onBrush} points={focusPoints.filter((p) => p.cls === "value").map((p) => ({ t: p.time, v: p.v }))} scale={focusStrip.scale} label={`${SHORT[focus]} and every plot`} height={44} inset={INSET} />
                    </div>

                    <PlotSync value={sync} onChange={onSync}>
                      <div className={v.main} data-testid="research-main">
                        {empty ? (
                          <PlotState
                            kind="not-collected"
                            what={`${LABEL[focus]} at ${st.name}`}
                            reason={<span data-testid="research-empty">{emptyReason}</span>}
                            height={wide ? 360 : 240}
                            action={
                              <ButtonGroup data-testid="research-empty-actions" className={v.emptyActions}>
                                {latest ? (
                                  <Button variant="primary" onClick={jumpLatest} data-testid="empty-latest" className={v.wrapBtn}>
                                    Latest {SHORT[focus]} sample: {day(latest.day, { year: true })}
                                  </Button>
                                ) : null}
                                {preset !== "all" ? (
                                  <Button variant="secondary" onClick={() => choosePreset("all")} data-testid="empty-all">
                                    Whole record
                                  </Button>
                                ) : null}
                                {neighbour ? (
                                  <Button variant="secondary" onClick={openNeighbour} data-testid="empty-neighbour" className={v.wrapBtn}>
                                    Open {neighbour.station.name}
                                  </Button>
                                ) : null}
                                <Button variant="quiet" href={st.source_url} external data-testid="empty-source">
                                  The record at SCCOOS ERDDAP
                                </Button>
                              </ButtonGroup>
                            }
                          />
                        ) : (
                          <MeasuredPlot
                            strip={focusStrip}
                            data={focusData}
                            win={win}
                            now={nowMs}
                            label={`${LABEL[focus]} at ${st.name}, ${range(win[0], win[1], { year: true })}`}
                            height={{ desktop: 360, phone: 240 }}
                            selected={selectedPoint}
                            onSelect={(p) => select(p ? p.meta : null)}
                            testid="research-plot"
                          />
                        )}
                      </div>

                      {cmpStrip ? (
                        <div className={v.second} data-testid="research-compare-plot">
                          <p className={v.plotTitle}>
                            {LABEL[cmp!]} <span className={v.unitWords}>{cmpStrip.unit}</span>
                          </p>
                          <MeasuredPlot strip={cmpStrip} data={cmpData} win={win} now={nowMs} label={`${LABEL[cmp!]} at ${st.name}, for comparison`} height={{ desktop: 150, phone: 120 }} testid="research-plot-cmp" />
                        </div>
                      ) : null}

                      {modelStrip ? (
                        <div className={v.second} data-testid="strip-model" data-collapsed={!showModel || undefined}>
                          {showModel ? (
                            <p className={v.plotTitle}>
                              Model · {modelStrip.label} <span className={v.unitWords}>{modelStrip.unit}</span>
                            </p>
                          ) : null}
                          {showModel ? (
                            <ModelPlot strip={modelStrip} data={modelData} win={win} now={nowMs} label={`Model: ${modelStrip.label}, near ${st.name}`} height={{ desktop: 120, phone: 100 }} testid="research-plot-model" />
                          ) : (
                            <p className={v.collapsed} data-testid="research-model-collapsed">
                              <span>
                                <b>Model · {modelStrip.label}</b> is not drawn: this window has no measured {SHORT[focus]} to read it beside, and a model probability is not a measurement.
                              </span>
                              <Button variant="quiet" onClick={() => setModelAnyway(modelKey)} data-testid="model-show">
                                Show the model alone
                              </Button>
                            </p>
                          )}
                          {/* T10: the mandatory words sit with the model, always visible, never behind a tap */}
                          <p className={v.modelWords} data-testid="research-model-words">
                            C-HARM: {MODEL_WORDS.charAt(0).toLowerCase()}
                            {MODEL_WORDS.slice(1)} No demonstrated skill at the shore; a high reading here is near-constant background, not an alarm.
                          </p>
                          {modelStrip.historyStart ? <p className={s.small}>No model history kept before {day(modelStrip.historyStart, { year: true })}.</p> : null}
                        </div>
                      ) : null}
                    </PlotSync>
                  </div>

                  <div className={v.under}>
                    <div className={`${v.stationPick} ${v.onlyPhone}`}>{stationPicker(stationSelectPhone, "research-station-select-phone")}</div>
                    <Stamp provenance={provenance} now={now} />
                    <ul className={v.legend} aria-label="Legend" data-testid="research-legend">
                      {legendKinds.map((k) => (
                        <li key={k}>
                          <DataStateMark kind={k} />
                          <span>{LEGEND_WORDS[k]}</span>
                        </li>
                      ))}
                    </ul>
                    {measured.length > 1 ? (
                      <div className={v.cmpPick}>
                        <label htmlFor={cmpSelect}>Compare with</label>
                        <span className={v.selectWrap}>
                          <select id={cmpSelect} className={v.select} value={cmp ?? ""} onChange={(e) => commit(null, { cmp: (e.target.value || null) as ObsVariableId | null })} data-testid="research-compare">
                            <option value="">Nothing</option>
                            {measured
                              .filter((x) => x !== focus)
                              .map((x) => (
                                <option key={x} value={x}>
                                  {LABEL[x]}
                                </option>
                              ))}
                          </select>
                        </span>
                      </div>
                    ) : null}
                  </div>
                  <p className={`${s.small} ${v.copyCell}`} data-testid="research-copy-start">
                    {copy.mayBeCut ? `Neritic's copy of this station starts ${day(`${copy.windowStart}T12:00:00Z`, { year: true })}; earlier samples, if any, are in ` : `Neritic asks the source for samples from ${day(`${copy.windowStart}T12:00:00Z`, { year: true })}; the first here is ${copy.firstSample ? day(`${copy.firstSample}T12:00:00Z`, { year: true }) : "none"}. The whole record is in `}
                    <a href={copy.sourceUrl} target="_blank" rel="noreferrer">
                      the source record
                    </a>
                    . Each axis spans the station&rsquo;s whole record and does not rescale with the window. In the list of variables each tick is a visit: tall for a value, short at the foot for a reported 0, short at the head for one rejected by QC, a dot where the station was visited and the variable was not measured; the frame is the window.
                  </p>
                </section>
                </div>

                {/* a direct child of the desk's main: on a phone the card pins above the destinations while the plot scrolls */}
                <Inspector record={record} onPrev={prev ? () => select(prev) : null} onNext={next ? () => select(next) : null} onClose={() => select(null)} position={position} />

                {/* ------------------------------------------------ this window */}
                <section className={v.group} aria-labelledby="rx-g-window">
                  <h2 id="rx-g-window" className={v.h2}>
                    This window
                  </h2>
                  <Disclosure
                    key={`samples-${open}`}
                    summary="Samples as a table"
                    defaultOpen={open === "samples"}
                    value={<span data-testid="row-samples">{count(tableRows.length, `${SHORT[focus]} sample`)}</span>}
                    data-testid="disc-samples"
                  >
                    <PlotTable<RecPoint>
                      caption={`${LABEL[focus]} at ${st.name}, every sample from ${day(committed[0], { year: true })} to ${day(committed[1], { year: true })}, newest first, exactly as published`}
                      columns={[
                        { id: "date", head: "Sampled (UTC)", cell: (p) => `${p.time.slice(0, 10)} ${p.time.slice(11, 16)}` },
                        { id: "value", head: focusStrip.unitShort, numeric: true, cell: (p) => (p.cls === "rejected" ? "" : String(p.v)) },
                        { id: "state", head: "State", cell: (p) => (p.cls === "zero" ? "Reported 0" : p.cls === "rejected" ? "Rejected by QC" : p.flagged ? "Measured, flagged high" : "Measured") },
                        { id: "depth", head: "Depth, m", numeric: true, cell: (p) => depthOf(p) ?? "" },
                      ]}
                      rows={tableRows}
                      rowKey={(p) => String(p.i)}
                      selected={selected ? String(selected.i) : null}
                      onSelect={(p) => select(p)}
                      empty={`No ${LABEL[focus].toLowerCase()} sample in this window. A visit without this measurement is not a zero.`}
                    />
                  </Disclosure>
                  <Disclosure
                    key={`window-${open}`}
                    summary="Statistics"
                    defaultOpen={open === "window"}
                    value={<span data-testid="row-window">{stats.max ? `Highest ${valueWords(focusStrip, stats.max)}, ${day(stats.max.t, { ref: now })}` : `No ${SHORT[focus]} value`}</span>}
                  >
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
                          <dt>Quantified</dt>
                          <dd>{avail?.quantified ?? 0}</dd>
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
                        <div>
                          <dt>Visits without it</dt>
                          <dd data-testid="research-absent">
                            {avail?.absent ?? 0} of {avail?.visits ?? 0}
                          </dd>
                        </div>
                      </dl>
                      <p className={s.small}>Of the samples of {LABEL[focus].toLowerCase()} in the window. A visit that did not measure it is not counted and is never a zero. The median is of the quantified values only.</p>
                    </div>
                  </Disclosure>
                  <Disclosure key={`export-${open}`} summary="Export and cite" defaultOpen={open === "export"} value={<span data-testid="row-export">{count(counts.rows, "row")}</span>}>
                    <div className={s.detailText} data-testid="research-exports">
                      <p id="rx-export-what" data-testid="research-export-what">
                        {count(counts.rows, "row")}, {range(committed[0], committed[1], { year: true })}
                        {counts.rows ? `: ${counts.by.filter((x) => x.n).map((x) => `${SHORT[x.v]} ${x.n.toLocaleString("en-US")}`).join(", ")}` : ""}. One row per sample and variable, with source columns.
                      </p>
                      <ButtonGroup>
                        <Button variant="secondary" icon="share" onClick={exportCsv} aria-describedby="rx-export-what">
                          Export the window as CSV
                        </Button>
                        {modelStrip ? (
                          <Button variant="secondary" onClick={exportModel}>
                            Model days in the window (CSV)
                          </Button>
                        ) : null}
                        {st.request_url ? (
                          <Button variant="quiet" href={st.request_url} external>
                            The station&rsquo;s own file at SCCOOS
                          </Button>
                        ) : null}
                      </ButtonGroup>
                      <figure className={s.cite}>
                        <figcaption>Cite this window</figcaption>
                        <blockquote data-testid="research-citation">{cite}</blockquote>
                        <Button variant="quiet" onClick={copyCitation}>
                          {copied ? "Copied" : "Copy the citation"}
                        </Button>
                        <span className={s.vh} aria-live="polite">
                          {copied ? "Citation copied" : ""}
                        </span>
                      </figure>
                      <p className={s.small}>Every file opens with lines starting # that say what it holds, its source, request, licence and pipeline run. An empty value is never a zero.</p>
                    </div>
                  </Disclosure>
                </section>

                {/* ------------------------------------------------ this record */}
                <section className={v.group} aria-labelledby="rx-g-record">
                  <h2 id="rx-g-record" className={v.h2}>
                    This record
                  </h2>
                  <Disclosure
                    key={`station-${open}`}
                    summary="Stations"
                    defaultOpen={open === "station"}
                    value={<span data-testid="row-station">{st.nearest_port_name ? `${kmWords(st.nearest_port_km ?? 0)} from ${st.nearest_port_name}` : latLon(st.lat, st.lon)}</span>}
                    data-testid="disc-station"
                  >
                    <nav aria-label="Shore stations, north to south">
                      <ol className={s.stationList} data-testid="research-stations">
                        {stations.map((x) => {
                          const l = latestByStation.get(x.station_id)?.last ?? null;
                          return (
                            <li key={x.station_id}>
                              <button type="button" aria-current={x.station_id === st.station_id ? "true" : undefined} onClick={() => chooseStationId(x.station_id)} data-testid={`list-station-${x.station_id}`}>
                                <span className={s.stationListName}>{x.name}</span>
                                <span className={s.quiet} data-testid={`list-latest-${x.station_id}`}>
                                  {l ? day(l.time, { year: true }) : `no ${SHORT[focus]}`}
                                </span>
                              </button>
                            </li>
                          );
                        })}
                      </ol>
                      <p className={s.small}>
                        North to south, with each station&rsquo;s newest {SHORT[focus]} sample (not its newest visit{lastVisit ? `: ${st.name} was last visited ${day(lastVisit, { year: true })}` : ""}). The chart shows the same stations where they sample.
                      </p>
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
                  </Disclosure>
                  <Disclosure
                    key={`checks-${open}`}
                    summary="Checks"
                    defaultOpen={open === "checks"}
                    value={
                      <span data-testid="row-checks">
                        {qcPassed} of {st.qc.length} passed
                      </span>
                    }
                  >
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
                      {st.status !== "updated" ? (
                        <p className={s.small}>
                          Last request: {st.status.replace(/_/g, " ")}
                          {st.error ? `. ${st.error}` : ""}
                        </p>
                      ) : null}
                    </div>
                  </Disclosure>
                  <Disclosure key={`source-${open}`} summary="Source and method" defaultOpen={open === "source"} value={<span>CalHABMAP, read {st.retrieved_at ? day(st.retrieved_at, { ref: now }) : "—"}</span>}>
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
                          <dd>
                            {vmeta?.matrix}
                            {vmeta?.fraction ? `, ${vmeta.fraction}` : ""}
                          </dd>
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
                  </Disclosure>
                  <Disclosure key={`model-${open}`} summary="The model beside it" defaultOpen={open === "model"} value={<span>{modelStrip ? `C-HARM, ${radius} km median` : "No C-HARM history here"}</span>}>
                    {!modelStrip ? (
                      <div className={s.detailText} data-testid="model-panel-none">
                        <p>No C-HARM nowcast history is published for this station, so the record has no model panel.</p>
                        {st.charm?.error ? <p className={`${s.small} ${s.url}`}>What the pipeline reported: {st.charm.error}</p> : null}
                      </div>
                    ) : (
                      <div className={s.detailText} data-testid="research-model">
                        <p>
                          {modelStrip.label}. {modelPlace(radius)} {MODEL_WORDS}
                        </p>
                        <p>
                          Neritic keeps each day&rsquo;s nowcast, from {modelStrip.historyStart ? day(modelStrip.historyStart, { year: true }) : "—"}; days with fewer than 60 % of the usual cells are marked and not joined. It is drawn on its own panel and never on the measured axis.
                        </p>
                        <p className={s.small}>{CHARM_ASSESSMENT}</p>
                        <p className={s.small}>{CHARM_SHORE}</p>
                      </div>
                    )}
                  </Disclosure>
                </section>

                {/* ------------------------------------------------ more tools */}
                <section className={v.group} aria-labelledby="rx-g-tools">
                  <h2 id="rx-g-tools" className={v.h2}>
                    More tools
                  </h2>
                  <Disclosure summary="Every variable on one time axis" value={<span>{count(strips.length, "strip")}, figure (SVG)</span>} open={stripsOpen} onOpenChange={setStripsOpen} data-testid="disc-strips">
                    {stripsOpen ? (
                      <div className={v.stripsBox}>
                        <RecordStrips ref={figureRef} strips={strips} win={committed} cursor={selected?.t ?? null} onCursor={() => undefined} focus={focus} onFocus={(id) => chooseVariable(id as ObsVariableId)} plotH={76} folded={folded} inkKey={st.station_id} now={nowMs} />
                        <ButtonGroup>
                          <Button variant="secondary" onClick={exportSvg} data-testid="export-svg">
                            The strips as a figure (SVG)
                          </Button>
                        </ButtonGroup>
                      </div>
                    ) : null}
                  </Disclosure>
                  <Disclosure key={`grid-${open}`} summary="Grid" defaultOpen={open === "grid"} value={<span>Cells, transects, areas</span>}>
                    <GridOpener manifest={manifest} ports={ports} official={official} baseUrl={baseUrl} station={st} now={now} />
                  </Disclosure>
                  <Disclosure summary="Landings" value={<span>Statewide, CDFW</span>}>
                    <div className={s.detailText}>
                      <p>Commercial landings and their value by species, statewide, from CDFW. They have no place on the coast, so they are not on the chart.</p>
                      <Button variant="quiet" href="/fisheries">
                        Open the landings history
                      </Button>
                    </div>
                  </Disclosure>
                </section>
              </>
            )}
          </DeskMain>
        </DeskBody>
        <DeskFoot>
          <p className={s.imprint}>Dataset generated {day(observations.generated_at, { ref: now })}. Water samples, not seafood tests. Not for navigation.</p>
        </DeskFoot>
      </Desk>
      <span hidden data-testid="research-window-ms" data-from={committed[0]} data-to={committed[1]} />
    </section>
  );
}

function GridOpener(props: { manifest: Manifest; ports: PortsCollection | null; official: OfficialDataset | null; baseUrl: string; station: { station_id: string; name: string; lat: number; lon: number }; now: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={s.detailText}>
      <p>The published C-HARM, satellite and HF-radar grids near {props.station.name}, cell by cell: read a cell, draw a transect or count an area, and export each as CSV.</p>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        Open the native grid
      </Button>
      {open ? <NativeGrid {...props} open={open} onOpenChange={setOpen} /> : null}
    </div>
  );
}

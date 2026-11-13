"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { animate, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { resolvePlace } from "@/lib/coast";
import type { ObsVariableId } from "@/lib/observations";
import { Button, ButtonGroup, Disclosure, Glyph, MapLayers, Micro, PageLayout, Pass, PlotSync, PlotTable, Stamp, Unavailable, brushEase, useMedia, usePhone, windowAround, type PlotSyncValue, type Provenance } from "@/nx/ds";
import { age, day, dayTime, latLon, range } from "@/nx/ds/format";
import { ChartScene, type ChartScene as Scene } from "@/nx/chart/scene";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import { openSheet } from "@/nx/shell/Margin";
import { distanceKm } from "@/lib/grid";
import { availabilityAll, chooseFocus, copyStart, emptyRange, exportCounts, inspectSample, latestWindow, modelCollapsed, nearestWithRecent, stationLatest, stepSample, visitTicks } from "./availability";
import { download, svgFigure } from "./dom";
import { citation, modelCsv, windowCsv } from "./exports";
import {
  buildStrips,
  chooseStation,
  clampWin,
  DAY,
  dayKey,
  DEFAULT_STATION,
  extentOf,
  figure,
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
import { plotPoints, type SamplePoint } from "./plotData";
import { RecordStrips, type RecordStripsHandle } from "./RecordStrips";
import { SampleCard, SampleRecordPanel } from "./SampleCard";
import { Overview } from "./Overview";
import { INSET, ModelPanel, ScrubPlot } from "./ScrubPlot";
import { StationMarks } from "./StationMarks";
import { VariableRail, type RailRow } from "./VariableRail";
import { CHARM_ASSESSMENT, CHARM_SHORE, MODEL_WORDS, modelPlace } from "./words";
import s from "./research.module.css";
import v from "./v13.module.css";
import x from "./rx.module.css";

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

/** The range control's words: the whole record is "All" where the row is short (the prototype's rule: under 1400 px). */
const PRESET_LABEL: Record<RangeId, string> = {
  "90d": "90 d",
  "1y": "1 y",
  "3y": "3 y",
  all: "Whole record",
};
const WINDOW_WORDS: Record<RangeId, string> = {
  "90d": "in these 90 days",
  "1y": "in this year",
  "3y": "in these 3 years",
  all: "in the whole record",
};

/** A query string as the router spells it, to compare the page's last write with the router's copy. */
const canon = (qs: string) => new URLSearchParams(qs).toString();

const SHORT: Record<ObsVariableId, string> = {
  pDA: "pDA",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "seriata group",
  pn_delicatissima: "delicatissima group",
  chl_extracted: "chlorophyll",
  temp: "temperature",
};
/** The short names at the head of a sentence: an abbreviation keeps its own case (pDA is never "PDA"). */
const SHORT_HEAD: Record<ObsVariableId, string> = {
  pDA: "pDA",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "Seriata group",
  pn_delicatissima: "Delicatissima group",
  chl_extracted: "Chlorophyll",
  temp: "Temperature",
};
/** The variable tabs' words (the prototype: Domoic acid, Pseudo-nitzschia, Chlorophyll, Temperature). */
const TAB: Record<ObsVariableId, string> = {
  pDA: "Domoic acid",
  dDA: "Dissolved DA",
  tDA: "Total DA",
  pn_seriata: "Pseudo-nitzschia",
  pn_delicatissima: "Delicatissima",
  chl_extracted: "Chlorophyll",
  temp: "Temperature",
};
const kmWords = (km: number) => (km < 1 ? `${Math.round(km * 1000)} m` : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`);
const checkName = (n: string) => ({ units: "Units", duplicates: "Duplicates", time: "Sample times" })[n] ?? n.charAt(0).toUpperCase() + n.slice(1).replace(/_/g, " ");
const count = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
/** How old a last sample is, in the prototype's words: days up to 400, then years to one decimal. */
const freshAge = (days: number) => (days <= 0 ? "today" : days === 1 ? "yesterday" : days <= 400 ? `${days} days ago` : `${(days / 365.25).toFixed(1)} years ago`);
/** Neighbouring stations within this distance are named on the chart; the others are quiet pins. */
const NAMED_KM = 80;

/** A row of radio buttons drawn as the prototype's word tabs or its segmented range control. Arrow keys move the choice. */
function Choice<T extends string>({
  label,
  items,
  value,
  onChange,
  look,
  testid,
  className,
}: {
  label: string;
  items: { id: T; label: string; aria?: string }[];
  value: T | null;
  onChange: (id: T) => void;
  look: "tabs" | "seg";
  testid: (id: T) => string;
  className?: string;
}) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const at = Math.max(
    0,
    items.findIndex((i) => i.id === value),
  );
  const key = (i: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : e.key === "Home" ? -i : e.key === "End" ? items.length - 1 - i : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + items.length) % items.length;
    onChange(items[n].id);
    refs.current[n]?.focus();
  };
  return (
    <div className={`${look === "tabs" ? x.tabs : x.seg} ${className ?? ""}`} role="radiogroup" aria-label={label}>
      {items.map((it, i) => (
        <button
          key={it.id}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={it.id === value}
          aria-label={it.aria}
          tabIndex={i === at ? 0 : -1}
          onClick={() => onChange(it.id)}
          onKeyDown={key(i)}
          data-testid={testid(it.id)}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

/** The height of an element's box, kept current. */
function useBoxHeight<T extends HTMLElement>(): [(el: T | null) => void, number] {
  const [h, setH] = useState(0);
  const [el, setEl] = useState<T | null>(null);
  useLayoutEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setH(Math.round(e.contentRect.height)));
    ro.observe(el);
    setH(Math.round(el.getBoundingClientRect().height));
    return () => ro.disconnect();
  }, [el]);
  return [setEl, h];
}

export function Research({ observations, ports, manifest, official, baseUrl, now }: ResearchProps) {
  const sp = useSearchParams();
  const nowMs = useMemo(() => Date.parse(now), [now]);
  const [hash, setHash] = useState("");
  useEffect(() => setHash(window.location.hash), []);
  // The view is the address, but the page does not wait for the router to say so. A choice is written to the
  // address AND taken as the page's state in the same event (`view`); the router's copy (`sp`) arrives a render
  // or more later, and on a slow machine a second choice was made before it did. The controls were then still
  // showing the old value: React put a select back to it, so choosing that value again raised no change at
  // all, and the first choice landed late over the second. With `view` every control shows the choice at once,
  // and the last choice is the state and the address.
  const [view, setView] = useState<string | null>(null);
  // every query string this page has written since the router last agreed with it, and the newest of them
  const written = useRef(new Set<string>());
  const newest = useRef<string | null>(null);
  const settle = useRef(0);
  const spKey = sp.toString();
  useEffect(() => {
    window.clearTimeout(settle.current);
    const mine = newest.current;
    if (mine == null) return;
    const release = () => {
      written.current.clear();
      newest.current = null;
      setView(null);
    };
    // The router repeats each of our writes to the address a little later, in order. An echo of an EARLIER
    // choice is not news: the newest choice stands (and is written again, see `write`). When the router's copy
    // is the newest choice and stays so, the address is the state again. Anything we never wrote is the reader
    // going somewhere else (back, a link): the address is the state at once.
    if (!written.current.has(canon(spKey))) release();
    else if (canon(spKey) === canon(mine)) settle.current = window.setTimeout(release, 800);
  }, [spKey]);
  useEffect(() => () => window.clearTimeout(settle.current), []);
  const write = useCallback((qs: string) => {
    window.clearTimeout(settle.current);
    written.current.add(canon(qs));
    newest.current = qs;
    setView(qs);
    // always written, even when the address already reads so: it may read so only because the router has just
    // repeated an earlier choice, and the router's last word must be this one
    window.history.replaceState(null, "", `${window.location.pathname}?${qs}${window.location.hash}`);
  }, []);
  const query = useMemo(() => parseQuery(view != null ? new URLSearchParams(view) : sp, hash), [view, sp, hash]);
  const phone = usePhone();
  // the prototype names the whole record "All" under 1400 px
  const roomy = useMedia("(min-width: 1400px)");
  const reduce = useReducedMotion();
  // The drawings (the brush's overview, the plots) are made in the browser, in boxes the stylesheet has already
  // sized. The words (station, last measured, counts, the empty range and its actions) are in the first HTML.
  const [drawn, setDrawn] = useState(false);
  useEffect(() => setDrawn(true), []);

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
  const focusStrip = strips.find((y) => y.v === focus) ?? null;
  const modelStrip = strips.find((y) => y.kind === "model") ?? null;
  const cmp = query.cmp && query.cmp !== focus && measured.includes(query.cmp) ? query.cmp : null;
  const cmpStrip = cmp ? (strips.find((y) => y.v === cmp) ?? null) : null;
  const extent = useMemo(() => extentOf(strips, nowMs), [strips, nowMs]);
  const focusPoints = useMemo(() => focusStrip?.points ?? [], [focusStrip]);

  // ------------------------------------------------------------ the window: the address holds it; a drag previews it
  const committed = useMemo<[number, number]>(
    () =>
      openingWindow({
        extent,
        from: query.from,
        to: query.to,
        range: query.range,
        day: query.day,
        focusTimes: focusPoints.filter((p) => p.cls !== "rejected").map((p) => p.t),
      }),
    [extent, query.from, query.to, query.range, query.day, focusPoints],
  );
  const [drag, setDragState] = useState<[number, number] | null>(null);
  const dragRef = useRef<[number, number] | null>(null);
  const setDrag = (w: [number, number] | null) => {
    dragRef.current = w;
    setDragState(w);
  };
  const dragging = useRef(false);
  // A preset or a jump eases the window to its new place (brushEase, 550 ms); a drag follows the hand. The data are
  // drawn in the window as it moves; nothing is interpolated between samples.
  const [eased, setEased] = useState<[number, number] | null>(null);
  const easedFrom = useRef<[number, number]>(committed);
  const lastStation = useRef(`${st.station_id}:${focus}`);
  useEffect(() => {
    const from = easedFrom.current;
    easedFrom.current = committed;
    const key = `${st.station_id}:${focus}`;
    const same = lastStation.current === key;
    lastStation.current = key;
    if (!same || reduce || dragging.current || (from[0] === committed[0] && from[1] === committed[1])) return setEased(null);
    const a = animate(0, 1, {
      ...brushEase,
      onUpdate: (k) => setEased([from[0] + (committed[0] - from[0]) * k, from[1] + (committed[1] - from[1]) * k]),
      onComplete: () => setEased(null),
    });
    return () => a.stop();
  }, [committed, st.station_id, focus, reduce]);
  const win = drag ?? eased ?? committed;

  /**
   * Writes the view to the address. What is not being changed is read from the ADDRESS as it is now, not from
   * this render: the address is ahead of React for a moment after each change, and a second choice made in
   * that moment (a variable, then a comparison) must not undo the first. `w: null` keeps the window.
   */
  const commit = useCallback(
    (
      w: [number, number] | null,
      patch: {
        station?: string;
        v?: ObsVariableId;
        day?: number | null;
        cmp?: ObsVariableId | null;
      } = {},
    ) => {
      // the newest choice of this page, else the address: never the address alone, which the router may have
      // just set back to an earlier choice
      const cur = parseQuery(new URLSearchParams(newest.current ?? window.location.search));
      const sameStation = cur.station == null || cur.station === st.station_id;
      const curV = sameStation && cur.v && measured.includes(cur.v) ? cur.v : focus;
      const curWin: [number, number] = cur.from != null && cur.to != null && sameStation ? [cur.from, cur.to] : committed;
      const vv = patch.v ?? curV;
      const cmpNext = patch.cmp === undefined ? (cur.station != null ? cur.cmp : cmp) : patch.cmp;
      write(
        queryFor({
          station: patch.station ?? st.station_id,
          v: vv,
          win: clampWin(w ?? curWin, extent),
          place: query.place,
          back: query.back,
          day: patch.day === undefined ? (cur.station != null ? cur.day : query.day) : patch.day,
          cmp: cmpNext === vv ? null : cmpNext,
        }),
      );
    },
    [extent, st.station_id, focus, measured, committed, query.place, query.back, query.day, cmp, write],
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
        const strip = strips.find((y) => y.v === a.v)!;
        return {
          a,
          unit: strip.unit,
          unitShort: strip.unitShort,
          ticks: visitTicks(st, a.v),
        };
      }),
    [availability, strips, st],
  );

  // ------------------------------------------------------------ the selected sample
  // The address's `day`, read on the selected variable. Without one, a station opens on its newest sample in the
  // window (the prototype: Monterey Wharf opens on its last pDA, Aug 31, 2022), until the reader clears it.
  const autoKey = `${st.station_id}:${focus}:${committed[0]}:${committed[1]}`;
  const [declined, setDeclined] = useState<string | null>(null);
  const selected = useMemo<RecPoint | null>(() => {
    if (query.day != null) return focusPoints.find((p) => Math.abs(p.t - query.day!) < 1.01 * DAY && inWin(p.t, committed)) ?? null;
    if (declined === autoKey) return null;
    for (let i = focusPoints.length - 1; i >= 0; i--) if (focusPoints[i].cls !== "rejected" && inWin(focusPoints[i].t, committed)) return focusPoints[i];
    return null;
  }, [query.day, focusPoints, committed, declined, autoKey]);
  const record = useMemo(() => (selected ? inspectSample(observations, st, focus, selected.i) : null), [selected, observations, st, focus]);
  const select = (p: RecPoint | null) => {
    if (!p) {
      setDeclined(autoKey);
      return commit(null, { day: null });
    }
    // a sample outside the window brings the window to it (same width)
    commit(inWin(p.t, committed) ? null : windowAround(extent, committed, p.t), { day: p.t });
  };
  const prev = selected ? stepSample(focusPoints, selected.t, -1) : null;
  const next = selected ? stepSample(focusPoints, selected.t, 1) : null;
  const position = selected ? focusPoints.findIndex((p) => p.i === selected.i) + 1 : null;

  // ------------------------------------------------------------ the record and the tools: a panel over the pass's main part
  const [panel, setPanel] = useState<"record" | "tools" | null>(() => (query.section ? "tools" : null));
  useEffect(() => {
    if (query.section) setPanel("tools");
  }, [query.section]);
  useEffect(() => {
    if (!record && panel === "record") setPanel(null);
  }, [record, panel]);

  // ------------------------------------------------------------ the plots' shared crosshair (the open station is always lit on the chart: StationMarks)
  const [hoverT, setHoverT] = useState<number | null>(null);
  const sync = useMemo<PlotSyncValue>(() => ({ hoverT, pinned: selected ? { t: selected.t } : null }), [hoverT, selected]);
  const onSync = (n: PlotSyncValue) => setHoverT(n.hoverT);

  const place2: [number, number] = useMemo(() => [st.lon, st.lat], [st.lon, st.lat]);
  const focusData = useMemo<SamplePoint[]>(() => (focusStrip ? plotPoints(focusStrip, place2) : []), [focusStrip, place2]);
  const cmpData = useMemo<SamplePoint[]>(() => (cmpStrip ? plotPoints(cmpStrip, place2) : []), [cmpStrip, place2]);
  const modelData = useMemo<SamplePoint[]>(() => (modelStrip ? plotPoints(modelStrip, place2) : []), [modelStrip, place2]);

  // ------------------------------------------------------------ the model: behind "Show the model"; never drawn alone beside nothing measured
  const [modelOn, setModelOn] = useState(false);
  const collapsed = modelStrip != null && modelCollapsed(focusPoints, committed);
  const [modelAnyway, setModelAnyway] = useState<string | null>(null);
  const modelKey = `${st.station_id}:${focus}`;
  const showModel = modelOn && modelStrip != null && (!collapsed || modelAnyway === modelKey);

  // ------------------------------------------------------------ choosing
  const chooseStationId = (id: string, w: [number, number] | null = null) => {
    const nextSt = stations.find((y) => y.station_id === id);
    const nextVars = nextSt ? chooseFocus(nextSt, focus).measured : [];
    commit(w, {
      station: id,
      v: nextVars.includes(focus) ? focus : undefined,
      day: null,
      cmp: null,
    });
  };
  const chooseVariable = (id: ObsVariableId) => commit(null, { v: id, day: null });
  const choosePreset = (r: RangeId) => commit(presetWindow(r, extent));
  const jumpLatest = () => {
    if (!latest) return;
    commit(latest.win, { day: latest.day });
  };
  const openNeighbour = () => {
    if (!neighbour) return;
    // open the neighbour on a year that holds its newest sample of this variable
    const w: [number, number] = [neighbour.last ? Date.parse(neighbour.last.time) - 335 * DAY : committed[0], neighbour.last ? Date.parse(neighbour.last.time) + 30 * DAY : committed[1]];
    write(
      queryFor({
        station: neighbour.station.station_id,
        v: focus,
        win: w,
        place: query.place,
        back: query.back,
      }),
    );
  };
  const showDetails = () => {
    openSheet("full");
    requestAnimationFrame(() =>
      requestAnimationFrame(() =>
        document.getElementById("rx-sample-detail")?.scrollIntoView({
          block: "start",
          behavior: reduce ? "auto" : "smooth",
        }),
      ),
    );
  };

  // ------------------------------------------------------------ choices are taken from the browser's own events
  // React does not always deliver an event: a `change` or a click that arrives while the router's transition
  // for the previous choice is being committed reaches the document and never reaches the component's handler
  // (measured at CPU 8x to 20x: the select showed the new variable, the page and the address kept the old one).
  // So the page's controls are also heard on the page's own element, where the event always arrives. Each
  // choice is idempotent: when React does deliver the same event afterwards, it writes the same view again.
  const pageRef = useRef<HTMLElement>(null);
  const choices = useRef({
    chooseVariable,
    chooseStationId,
    choosePreset,
    jumpLatest,
    commit,
  });
  useEffect(() => {
    choices.current = {
      chooseVariable,
      chooseStationId,
      choosePreset,
      jumpLatest,
      commit,
    };
  });
  useEffect(() => {
    const el = pageRef.current;
    if (!el) return;
    const id = (t: EventTarget | null, sel: string) => (t instanceof Element ? t.closest<HTMLElement>(sel) : null);
    const onChange = (e: Event) => {
      const sel = id(e.target, "select[data-testid]") as HTMLSelectElement | null;
      if (!sel) return;
      const c = choices.current;
      const which = sel.dataset.testid;
      if (which === "research-variable-select") c.chooseVariable(sel.value as ObsVariableId);
      else if (which === "research-station-select") c.chooseStationId(sel.value);
      else if (which === "research-compare") c.commit(null, { cmp: (sel.value || null) as ObsVariableId | null });
    };
    const onClick = (e: Event) => {
      const c = choices.current;
      const row = id(e.target, '[role="radio"][data-testid^="rail-"], [data-testid^="avail-"]');
      if (row) return c.chooseVariable(row.dataset.testid!.replace(/^(rail|avail)-/, "") as ObsVariableId);
      const r0 = id(e.target, '[data-testid^="range-"], [data-testid="empty-latest"], [data-testid="empty-all"]');
      const r = r0?.dataset.testid?.replace(/^(range|empty)-/, "");
      if (r === "latest") c.jumpLatest();
      else if (r && RANGES.some((y) => y.id === r)) c.choosePreset(r as RangeId);
    };
    el.addEventListener("change", onChange);
    el.addEventListener("click", onClick);
    return () => {
      el.removeEventListener("change", onChange);
      el.removeEventListener("click", onClick);
    };
  }, []);

  // ------------------------------------------------------------ the chart: the station framed with its neighbours (design issue 2 on a phone)
  const homeStation = port ? nearestStation(stations, port.lat, port.lon, port.key)?.station.station_id : DEFAULT_STATION;
  const sceneBase = port ? `research:${port.key}` : "research";
  const near = useMemo(() => stations.filter((y) => y.station_id !== st.station_id && distanceKm(st.lat, st.lon, y.lat, y.lon) <= NAMED_KM), [stations, st]);
  const scene: Scene = useMemo(() => {
    const id = st.station_id === homeStation ? sceneBase : `${sceneBase}:${st.station_id}`;
    const focusPort = port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null;
    // the station and its named neighbours, with room around them for their names: at least 60 km of coast each way
    const pts = [st, ...near];
    const dLat = 115 / 111.32;
    const dLon = 70 / (111.32 * Math.cos((st.lat * Math.PI) / 180));
    const lats = pts.map((p) => p.lat);
    const lons = pts.map((p) => p.lon);
    return {
      id,
      focus: focusPort,
      view: {
        bounds: [
          [Math.min(...lons, st.lon - dLon) - 0.12, Math.min(...lats, st.lat - dLat) - 0.06],
          [Math.max(...lons, st.lon + dLon) + 0.12, Math.max(...lats, st.lat + dLat) + 0.06],
        ],
        maxZoom: 11,
      },
      named: stations.map((y) => ({ name: y.name, lon: y.lon, lat: y.lat })),
    };
  }, [st, near, port, sceneBase, homeStation, stations]);
  const latestByStation = useMemo(() => new Map(stationLatest(stations, focus).map((y) => [y.station.station_id, y])), [stations, focus]);
  const stationSubline = (id: string) => {
    const l = latestByStation.get(id)?.last ?? null;
    if (!l) return null;
    const t = Date.parse(l.time);
    const strip = { kind: "measured" as const, v: focus };
    if (nowMs - t > 365 * DAY) return `${SHORT_HEAD[focus]} last measured ${new Date(t).getUTCFullYear()}`;
    return `${SHORT_HEAD[focus]} ${day(t)} · ${l.qualifier === "reported_zero" ? "reported 0" : `${figure(strip, l.value)} ${focusStrip?.unitShort ?? ""}`}`;
  };

  // ------------------------------------------------------------ exports
  const figureRef = useRef<RecordStripsHandle>(null);
  const [stripsOpen, setStripsOpen] = useState(false);
  const folded = useMemo(() => new Set(strips.filter((y) => !y.points.some((p) => inWin(p.t, committed))).map((y) => y.id)), [strips, committed]);
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
    if (el)
      download(
        `${fileBase}.svg`,
        svgFigure(el, {
          title: `${st.name} (${st.station_id}), ${range(committed[0], committed[1], { year: true })}`,
          strips,
          plotH: 76,
          folded,
          caption: cite,
        }),
        "image/svg+xml",
      );
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
  const vmeta = observations.variables.find((y) => y.id === focus);
  const lastVisit = st.sample_times.at(-1) ?? null;
  const lastOf = avail?.last ?? null;
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
  const cmpSelect = useId();
  const variableSelect = useId();

  const insteadWords = empty?.instead.length ? `${empty.instead.map((y) => SHORT[y]).join(", ")} only` : "none of the variables shown";
  const emptyReason = empty ? (
    <>
      {empty.kind === "never"
        ? `No ${LABEL[focus].toLowerCase()} has been published for ${st.name}.`
        : empty.last
          ? `Last measured ${day(empty.last.t, { year: true })} (${freshAge(Math.floor((nowMs - empty.last.t) / DAY))}): ${valueWords(focusStrip!, empty.last)}.`
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

  const tableRows = useMemo(() => logRows(focusPoints, committed, "date", "desc"), [focusPoints, committed]);
  const depthOf = (p: RecPoint) => st.depths_m[p.i];
  const lastDays = lastOf ? Math.floor((nowMs - lastOf.t) / DAY) : null;
  const zerosShown = focusStrip?.scale === "log" && (avail?.zeros ?? 0) > 0;
  const refWords = focusStrip?.threshold ? `C-HARM model reference, ${focusStrip.threshold.value.toLocaleString("en-US")} ${focusStrip.unitShort} (not a safety limit)` : null;
  const tabs = measured.map((id) => ({
    id,
    label: id === "pn_seriata" && measured.includes("pn_delicatissima") ? "Seriata" : TAB[id],
  }));

  // ------------------------------------------------------------ the plot area's room
  const [chartRef, chartH] = useBoxHeight<HTMLDivElement>();
  const modelRoom = showModel ? Math.max(70, Math.round(chartH * 0.3)) : 0;
  const cmpRoom = cmpStrip ? Math.max(80, Math.round((chartH - modelRoom) * 0.38)) : 0;
  const laneAxis = (focusStrip?.scale === "log" ? 24 : 0) + 22 + 26;
  // the phone's plot: the prototype's 300 px chart (232 px under 360 px wide), the overview and axis included
  const tiny = useMedia("(max-width: 359px)");
  const plotH = phone ? (tiny ? 120 : 180) : Math.max(48, chartH - modelRoom - cmpRoom - laneAxis);
  // The overview of the whole record stands under the plot where the pass leaves the chart 226 px or more (the
  // prototype's rule); on a short screen the range control alone moves the window, so the plot keeps its room.
  const [brushRef, brushH] = useBoxHeight<HTMLDivElement>();
  const overview = phone || chartH + brushH + 20 >= 226;

  const arrived =
    choice.arrival || askedMissing || query.askedVar || query.back ? (
      <p className={x.arrived} data-testid="research-arrived">
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
        {askedMissing ? (
          <span>
            {" "}
            {LABEL[askedMissing]} isn&rsquo;t measured at {st.name}; the tabs show what is.
          </span>
        ) : null}
        {query.askedVar ? <span> &ldquo;{query.askedVar}&rdquo; isn&rsquo;t a quantity a shore station measures; the tabs show what this one does.</span> : null}
        {query.back ? (
          <Link href={query.back} className={s.back} data-testid="research-back">
            Back
          </Link>
        ) : null}
      </p>
    ) : null;

  // ------------------------------------------------------------ the record and its tools (a panel over the pass on a wide screen; in the sheet on a phone)
  const tools = (
    <div className={x.toolsBody}>
      <div className={x.toolsHead}>
        <Micro>Record and tools · {st.name}</Micro>
      </div>
      {focusStrip ? (
        <section aria-label="Measured here" className={x.toolsRail}>
          <VariableRail list rows={railRows} drawTicks={drawn} selected={focus} onSelect={chooseVariable} extent={extent} win={win} now={nowMs} stationName={st.name} />
        </section>
      ) : null}
      <div className={x.picks}>
        <div className={v.cmpPick}>
          <label htmlFor={stationSelect}>Station</label>
          <span className={v.selectWrap}>
            <select id={stationSelect} className={v.select} value={st.station_id} onChange={(e) => chooseStationId(e.target.value)} data-testid="research-station-select">
              {stations.map((y) => (
                <option key={y.station_id} value={y.station_id}>
                  {y.name}
                </option>
              ))}
            </select>
          </span>
        </div>
        {measured.length > 1 ? (
          <div className={v.cmpPick}>
            <label htmlFor={cmpSelect}>Compare with</label>
            <span className={v.selectWrap}>
              <select
                id={cmpSelect}
                className={v.select}
                value={cmp ?? ""}
                onChange={(e) =>
                  commit(null, {
                    cmp: (e.target.value || null) as ObsVariableId | null,
                  })
                }
                data-testid="research-compare"
              >
                <option value="">Nothing</option>
                {measured
                  .filter((y) => y !== focus)
                  .map((y) => (
                    <option key={y} value={y}>
                      {LABEL[y]}
                    </option>
                  ))}
              </select>
            </span>
          </div>
        ) : null}
        {phone ? (
          <div className={v.cmpPick}>
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
        ) : null}
      </div>
      <Stamp provenance={provenance} now={now} />

      {focusStrip ? (
        <>
          <section className={v.group} aria-labelledby="rx-g-window">
            <h2 id="rx-g-window" className={v.h2}>
              This window
            </h2>
            <Disclosure key={`samples-${open}`} summary="Samples as a table" defaultOpen={open === "samples"} value={<span data-testid="row-samples">{count(tableRows.length, `${SHORT[focus]} sample`)}</span>} data-testid="disc-samples">
              <PlotTable<RecPoint>
                caption={`${LABEL[focus]} at ${st.name}, every sample from ${day(committed[0], { year: true })} to ${day(committed[1], { year: true })}, newest first, exactly as published`}
                columns={[
                  {
                    id: "date",
                    head: "Sampled (UTC)",
                    cell: (p) => `${p.time.slice(0, 10)} ${p.time.slice(11, 16)}`,
                  },
                  {
                    id: "value",
                    head: focusStrip.unitShort,
                    numeric: true,
                    cell: (p) => (p.cls === "rejected" ? "" : String(p.v)),
                  },
                  {
                    id: "state",
                    head: "State",
                    cell: (p) => (p.cls === "zero" ? "Reported 0" : p.cls === "rejected" ? "Rejected by QC" : p.flagged ? "Measured, flagged high" : "Measured"),
                  },
                  {
                    id: "depth",
                    head: "Depth, m",
                    numeric: true,
                    cell: (p) => depthOf(p) ?? "",
                  },
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
                    <dd>
                      {stats.median != null
                        ? valueWords(focusStrip, {
                            ...stats.max!,
                            v: stats.median,
                            flagged: false,
                          })
                        : "none"}
                    </dd>
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
                  {counts.rows
                    ? `: ${counts.by
                        .filter((y) => y.n)
                        .map((y) => `${SHORT[y.v]} ${y.n.toLocaleString("en-US")}`)
                        .join(", ")}`
                    : ""}
                  . One row per sample and variable, with source columns.
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
        </>
      ) : null}

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
              {stations.map((y) => {
                const l = latestByStation.get(y.station_id)?.last ?? null;
                return (
                  <li key={y.station_id}>
                    <button type="button" aria-current={y.station_id === st.station_id ? "true" : undefined} onClick={() => chooseStationId(y.station_id)} data-testid={`list-station-${y.station_id}`}>
                      <span className={s.stationListName}>{y.name}</span>
                      <span className={s.quiet} data-testid={`list-latest-${y.station_id}`}>
                        {l ? day(l.time, { year: true }) : `no ${SHORT[focus]}`}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
            <p className={s.small}>
              North to south, with each station&rsquo;s newest {SHORT[focus]} sample (not its newest visit
              {lastVisit ? `: ${st.name} was last visited ${day(lastVisit, { year: true })}` : ""}
              ). The chart shows the same stations where they sample.
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
                Neritic keeps each day&rsquo;s nowcast, from {modelStrip.historyStart ? day(modelStrip.historyStart, { year: true }) : "—"}; days with fewer than 60 % of the usual cells are marked and not joined. It is drawn on its own panel and
                never on the measured axis.
              </p>
              <p className={s.small}>{CHARM_ASSESSMENT}</p>
              <p className={s.small}>{CHARM_SHORE}</p>
            </div>
          )}
        </Disclosure>
      </section>

      <section className={v.group} aria-labelledby="rx-g-tools">
        <h2 id="rx-g-tools" className={v.h2}>
          More tools
        </h2>
        <Disclosure summary="Every variable on one time axis" value={<span>{count(strips.length, "strip")}, figure (SVG)</span>} open={stripsOpen} onOpenChange={setStripsOpen} data-testid="disc-strips">
          {stripsOpen ? (
            <div className={v.stripsBox}>
              <RecordStrips
                ref={figureRef}
                strips={strips}
                win={committed}
                cursor={selected?.t ?? null}
                onCursor={() => undefined}
                focus={focus}
                onFocus={(id) => chooseVariable(id as ObsVariableId)}
                plotH={76}
                folded={folded}
                inkKey={st.station_id}
                now={nowMs}
              />
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
      <p className={s.small} data-testid="research-copy-start">
        {copy.mayBeCut
          ? `Neritic's copy of this station starts ${day(`${copy.windowStart}T12:00:00Z`, { year: true })}; earlier samples, if any, are in `
          : `Neritic asks the source for samples from ${day(`${copy.windowStart}T12:00:00Z`, { year: true })}; the first here is ${copy.firstSample ? day(`${copy.firstSample}T12:00:00Z`, { year: true }) : "none"}. The whole record is in `}
        <a href={copy.sourceUrl} target="_blank" rel="noreferrer">
          the source record
        </a>
        . Each axis spans the station&rsquo;s whole record and does not rescale with the window. In the list of variables each tick is a visit: tall for a value, short at the foot for a reported 0, short at the head for one rejected by QC, a dot
        where the station was visited and the variable was not measured; the frame is the window.
      </p>
      <p className={s.imprint}>Dataset generated {day(observations.generated_at, { ref: now })}. Water samples, not seafood tests. Not for navigation.</p>
    </div>
  );

  const windowWords = preset ? WINDOW_WORDS[preset] : "in this window";
  const stub = focusStrip ? (
    <SampleCard
      record={record}
      point={selected}
      strip={focusStrip}
      n={position}
      total={focusPoints.length}
      outside={false}
      prev={prev}
      next={next}
      onStep={(d) => {
        const p = d < 0 ? prev : next;
        if (p) select(p);
      }}
      recordOpen={panel === "record"}
      onRecord={() => setPanel((p) => (p === "record" ? null : "record"))}
      onDetails={showDetails}
    />
  ) : (
    <div className={x.card} data-testid="research-inspector" data-empty="">
      <Micro glyph="none">No samples</Micro>
    </div>
  );

  return (
    <section ref={pageRef} className={x.page} data-testid="nx-research" data-station={st.station_id} data-var={focus} data-layout="stage">
      <PageLayout mode="stage" phoneMap={0.42} chartLabel={st.name} />
      <MapLayers page="research" defaults={{ toxin: true }} />
      <ChartScene scene={scene} />
      <StationMarks
        stations={stations.map((y) => ({
          id: y.station_id,
          name: y.name,
          lat: y.lat,
          lon: y.lon,
        }))}
        selected={st.station_id}
        named={new Set(near.map((y) => y.station_id))}
        subline={stationSubline}
        radiusKm={showModel ? radius : null}
        onPick={(id) => chooseStationId(id)}
      />

      <Pass label={`${st.name}: shore station record`} at={{ lon: st.lon, lat: st.lat }} stub={stub} stubWidth={340} className={x.pass} stubClassName={x.stub} data-testid="research-record">
        <div className={x.main} data-testid="research-chart" data-panel={panel ?? undefined}>
          {!phone ? (
            <button type="button" className={x.toolsLink} onClick={() => setPanel((p) => (p === "tools" ? null : "tools"))} aria-expanded={panel === "tools"} data-testid="research-tools-open">
              {panel === "tools" ? "Close the tools" : "Table, exports, sources"}
            </button>
          ) : null}
          <div className={x.head} data-many={tabs.length > 4 || undefined}>
            <Micro className={x.kicker} data-testid="research-variable">
              {LABEL[focus]} · water samples, not seafood tests
            </Micro>
            <div className={x.title}>
              <h1 data-testid="research-title">{st.name}</h1>
              <p className={x.fresh} data-testid="research-status">
                {focusStrip ? (
                  lastOf ? (
                    <>
                      <Glyph kind={lastDays != null && lastDays > 30 ? "old" : "measured"} />
                      {SHORT_HEAD[focus]} last measured {day(lastOf.t, { year: true })} ({freshAge(lastDays ?? 0)})
                    </>
                  ) : (
                    <>
                      <Glyph kind="none" />
                      {SHORT_HEAD[focus]} never measured here
                    </>
                  )
                ) : (
                  `${st.name} has published no samples Neritic can read.`
                )}
              </p>
            </div>
            {focusStrip ? <Choice label={`Variable measured at ${st.name}`} items={tabs} value={focus} onChange={chooseVariable} look="tabs" testid={(id) => `rail-${id}`} className={x.tabsArea} /> : null}
            <Choice
              label="Range"
              items={RANGES.map((r) => ({
                id: r.id,
                label: r.id === "all" && !roomy ? "All" : PRESET_LABEL[r.id],
                aria: r.id === "all" ? "Whole record" : undefined,
              }))}
              value={preset}
              onChange={choosePreset}
              look="seg"
              testid={(id) => `range-${id}`}
              className={x.segArea}
            />
            {arrived}
          </div>

          {!focusStrip ? (
            <div className={x.chart}>
              <Unavailable what={`${st.name} has published no samples`} why={st.error ?? "The station's last request returned nothing Neritic could read."} />
            </div>
          ) : (
            <>
              <Micro as="p" className={x.caption}>
                <span data-testid="research-window-count">
                  {focusStrip.unit}
                  {focusStrip.scale === "log" ? (phone ? " · log axis, whole record" : " · log axis fixed to the whole record") : ""} · {count(avail?.withValue ?? 0, "sample")} {windowWords}
                  {avail?.zeros ? ` · ${avail.zeros} reported 0` : ""}
                  {avail?.absent && !phone ? ` · ${count(avail.absent, "visit")} without it` : ""}
                </span>
                <span className={preset ? x.vh : undefined} data-testid="research-window">
                  {" "}
                  · {range(win[0], win[1], { year: true })}
                </span>
              </Micro>
              <div ref={chartRef} className={x.chart}>
                <PlotSync value={sync} onChange={onSync}>
                  <div className={x.measured} data-testid="research-main" data-empty={empty ? "" : undefined}>
                    {empty ? (
                      <div className={x.empty}>
                        <Micro glyph="none">Not collected in this range</Micro>
                        <h2 className={x.emptyH}>
                          {LABEL[focus]} at {st.name}
                        </h2>
                        <p className={x.emptyP} data-testid="research-empty">
                          {emptyReason}
                        </p>
                        <ButtonGroup data-testid="research-empty-actions" className={x.emptyActions}>
                          {latest ? (
                            <Button variant="primary" onClick={jumpLatest} data-testid="empty-latest">
                              Latest sample: {day(latest.day, { year: true })}
                            </Button>
                          ) : null}
                          {preset !== "all" ? (
                            <Button variant="secondary" onClick={() => choosePreset("all")} data-testid="empty-all">
                              Whole record
                            </Button>
                          ) : null}
                          {neighbour ? (
                            <Button variant="secondary" onClick={openNeighbour} data-testid="empty-neighbour">
                              Open {neighbour.station.name}
                            </Button>
                          ) : null}
                          <Button variant="quiet" href={st.source_url} external data-testid="empty-source">
                            The record at SCCOOS ERDDAP
                          </Button>
                        </ButtonGroup>
                      </div>
                    ) : !drawn || chartH === 0 ? (
                      <p className={x.drawing} data-testid="research-drawing">
                        Drawing {count(avail?.withValue ?? 0, "sample")} of {LABEL[focus].toLowerCase()}…
                      </p>
                    ) : (
                      <ScrubPlot
                        strip={focusStrip}
                        data={focusData}
                        win={win}
                        now={nowMs}
                        label={`${LABEL[focus]} at ${st.name}, ${range(win[0], win[1], { year: true })}`}
                        height={plotH}
                        pinnedT={selected?.t ?? null}
                        onSelect={(p) => select(p ? p.meta : null)}
                        drawKey={`${st.station_id}:${focus}`}
                        testid="research-plot"
                      />
                    )}
                  </div>

                  {cmpStrip ? (
                    <div className={x.second} data-testid="research-compare-plot">
                      <Micro as="p" className={x.secondTitle}>
                        Compared: {LABEL[cmp!]} · {cmpStrip.unit}
                      </Micro>
                      {drawn && chartH > 0 ? (
                        <ScrubPlot
                          strip={cmpStrip}
                          data={cmpData}
                          win={win}
                          now={nowMs}
                          label={`${LABEL[cmp!]} at ${st.name}, for comparison`}
                          height={Math.max(40, cmpRoom - 66)}
                          pinnedT={null}
                          drawKey={`${st.station_id}:${cmp}`}
                          testid="research-plot-cmp"
                        />
                      ) : null}
                    </div>
                  ) : null}

                  {modelStrip && modelOn ? (
                    <div className={x.second} data-testid="strip-model" data-collapsed={!showModel || undefined}>
                      {/* the mandatory words sit with the model, visible beside it whenever it is shown */}
                      <p className={x.modelNote}>
                        <b>
                          Model · {modelStrip.label}, median of model cells within {radius} km.
                        </b>{" "}
                        <span data-testid="research-model-words">
                          C-HARM: {MODEL_WORDS.charAt(0).toLowerCase()}
                          {MODEL_WORDS.slice(1)} No demonstrated skill at the shore; a high reading here is near-constant background, not an alarm.
                        </span>
                      </p>
                      {showModel ? (
                        drawn && chartH > 0 ? (
                          <ModelPanel strip={modelStrip} data={modelData} win={win} label={`Model: ${modelStrip.label}, near ${st.name}`} height={Math.max(36, modelRoom - 44)} testid="research-plot-model" />
                        ) : null
                      ) : (
                        <p className={x.collapsed} data-testid="research-model-collapsed">
                          <span>The model is not drawn: this window has no measured {SHORT[focus]} to read it beside, and a model probability is not a measurement.</span>
                          <Button variant="quiet" onClick={() => setModelAnyway(modelKey)} data-testid="model-show">
                            Show the model alone
                          </Button>
                        </p>
                      )}
                    </div>
                  ) : null}
                </PlotSync>
              </div>
              <div
                ref={brushRef}
                hidden={!overview}
                className={x.brush}
                onPointerDownCapture={() => {
                  dragging.current = true;
                }}
                data-testid="research-brush"
              >
                <Overview extent={extent} value={win} onChange={onBrush} points={drawn ? focusPoints : []} strip={focusStrip} selectedT={selected?.t ?? null} label={`${SHORT[focus]} and every plot`} inset={INSET} />
              </div>
              <div className={x.foot} data-testid="research-legend">
                <span className={x.k}>
                  <Glyph kind="measured" />
                  Measured
                </span>
                {zerosShown ? (
                  <span className={x.k}>
                    <Glyph kind="zero" />
                    Reported 0: at or below the lab&rsquo;s limit
                  </span>
                ) : null}
                {refWords ? (
                  <span className={x.k} data-testid="research-ref-key">
                    <i className={x.ln} aria-hidden="true" />
                    {refWords}
                  </span>
                ) : null}
                {modelStrip ? (
                  <button type="button" role="switch" aria-checked={modelOn} className={x.switch} onClick={() => setModelOn((m) => !m)} data-testid="research-model-switch">
                    <i aria-hidden="true" />
                    Show the model
                  </button>
                ) : null}
                <span className={x.note}>Water samples, not seafood tests.</span>
              </div>
            </>
          )}

          {record && panel === "record" ? (
            <section id="rx-record" className={x.panel} aria-label="Sample record" data-testid="research-sample-record">
              <SampleRecordPanel record={record} n={position} total={focusPoints.length} short={SHORT[focus]} onClose={() => setPanel(null)} />
            </section>
          ) : null}
          {phone || panel === "tools" ? (
            <section className={`${x.panel} ${x.toolsPanel}`} aria-label="Record and tools" data-testid="research-tools">
              {tools}
            </section>
          ) : null}
        </div>
      </Pass>
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

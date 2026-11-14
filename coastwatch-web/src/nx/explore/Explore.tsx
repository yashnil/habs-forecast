"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { animate, useReducedMotion } from "motion/react";
import { layerIn, layerOut } from "@/nx/ds/motion";
import { markForDisplay, westmostPerRow } from "@/nx/chart/layers/marks";
import { shoreLon } from "@/nx/chart/shore";
import { setMapLayer, useMapLayer } from "@/nx/chart/mapLayers";
import { registerCover } from "@/nx/chart/cover";
import { ChartRegulation, RAMP_ZONES, rampZoneAt } from "@/nx/chart/regulation";
import { readPrefs, writePrefs } from "./prefs";
import { useOnline, useSavedMark } from "@/nx/offline/useOffline";
import { asValueGrid, loadChlEstimateIndex, loadOceanCodes, loadWcofsCurrents, loadWcofsIndex } from "@/lib/ocean/client";
import { wcofsLabel, type ChlEstimateIndex, type OceanGrid, type WcofsIndex } from "@/lib/ocean/contract";
import { chlEstimateStateWords, chlEstimateUsable, extendAxis, maskedCodes, maskedVectors, pickServedStep, playableSteps, stepsAxis, stepWhen, wcofsTracks, wcofsUsable } from "./ocean";
import { ChlEstimateLegend, EstimateRow, SstModelLegend, WcofsNote } from "./OceanParts";
import { token } from "@/nx/chart/layers/colors";
import type { FieldGrid } from "@/nx/chart/layers/grid";
import { isolineLabels, traceIsolines } from "@/nx/chart/layers/contours";
import { clearIsotherms, recolourIsotherms, showIsotherms } from "@/nx/chart/layers/isotherms";

/** Isotherms shorter than this are dropped (tiny closed rings and specks of a 4 km model grid), km. */
const ISO_MIN_KM = 12;
import type { Map as MapLibreMap } from "maplibre-gl";
import { age, Button, day, range, fixed, latLon, plainError, sig, useMedia, usePhone, Readout, Unavailable, PageLayout, Ticket, TicketMain, TicketStub, Dock, MapLayers, type Provenance } from "@/nx/ds";
import { getFrame, onFrame } from "@/nx/chart/runtime";
import { capability, type GraphicsTier } from "@/lib/capability";
import type { CurrentField } from "@/lib/currents";
import { openMargin } from "@/nx/shell/Margin";
import { StaticField } from "./StaticField";
import { ChartScene } from "@/nx/chart/scene";
import { ChartMarks } from "@/nx/chart/Labels";
import { keepChartPointInView, whenChartReady } from "@/nx/chart/control";
import { addChartLayer } from "@/nx/chart/layers/registry";
import { modeColours, onModeChange } from "@/nx/chart/layers/colors";
import { createFieldLayer, type FieldController, type FieldStyle } from "@/nx/chart/layers/FieldLayer";
import { createFlowLayer, type FlowController } from "@/nx/chart/layers/FlowLayer";
import { persistence } from "@/nx/chart/layers/grid";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import { loadCodes, loadedChunks, boxAround } from "./codes";
import { COAST_ZOOM, coversPort, EXPLORE_ZOOM, exploreBounds, STATE_BOUNDS, withinFor } from "./camera";
import { backLink, readFrom, readView, type ExploreView } from "./links";
import { cellKmOf, chlViewOf, frameForView, chipWords, layerMetas, notPublished, resolutionNote, speedStats, speedWords } from "./layers";
import { StationPoints, type StationPick } from "./StationPoints";
import { StationReadout } from "./StationReadout";
import { gridBounds } from "@/nx/chart/layers/grid";
import { ExploreLegend } from "./ExploreLegend";
import { Locator, ReadingOnChart } from "./ChartOverlay";
import type { CurrentFrame, ExploreFrame, ExploreModel, ExploreProduct, GridSpec, ProductId } from "./model";
import { ADJACENT_WORDS, isLandAdjacent, loadAdjacency } from "./adjacency";
import { markAdjacent } from "@/nx/chart/layers/adjacent";
import { outsideObservedArea, cellGeometry, cellSizeWords, coastHref, compassWord, gapPosition, isothermLabels, KNOTS, NEARBY_KM, nearestCurrent, type CellGeometry, type GapPosition, type IsoLabel, pixelDate, readPoint, researchHref, RESEARCH_VAR, stationFor, transect, type TransectPoint } from "./read";
import { nextOf, playFrames, playHours, qualityHours } from "./timeline";
import { TimeHead, TimeRail } from "./TimeRail";
import { Transect } from "./Transect";
import { aboutWords, CHL_DOMAIN, coverageNote, currentsByDefault, currentsStill, currentsToggleWords, isOld, ISO_F, LAYER_LABEL, layerKindWord, leadSentence, noValueWords, charmStalledWords, OUTLOOK_NOT_AVAILABLE, oldMark as oldWord, shortLine, oldMark, readingStamp, readoutTitle, unitOf, valueWords, type CurrentsState } from "./words";
import s from "./explore.module.css";

/** What a layer's nearest pier measurement is called, in a sentence. */
const MEASURE_WORDS: Record<ProductId, string> = { chl: "chlorophyll", outlook: "chlorophyll", sst: "water temperature", charm: "domoic acid" };

/** A measured cell that is land or beyond the coastal product's water (review j10: never "cloud"). */
const OUTSIDE_AREA_WORDS = "Outside this layer's area: land, or water the coastal chlorophyll product does not cover. No value is not a low value.";

/** The flight between a port and the whole coast, ms (v15.1 review: a deliberate 1.2–1.8 s, not a 0.4 s zoom). */
const STATE_FLIGHT_MS = 1500;

type Win = Window & { __nxExplore?: Record<string, unknown> };

/** How each product is drawn (src/nx/chart/layers/FieldLayer.ts). */
function styleFor(p: ExploreProduct, f: ExploreFrame): FieldStyle {
  if (p.id === "chl") return { kind: "ramp", domain: CHL_DOMAIN, scale: "log" };
  // a past outlook steps back like a past C-HARM run (T3)
  if (p.id === "outlook") return { kind: "outlook", domain: CHL_DOMAIN, scale: "log", opacity: p.pastRun ? 0.5 : 0.9 };
  // a past run is past in every frame: drawn lighter, as its legend is (T3, T23)
  if (p.id === "charm") return { kind: "classes", past: f.past || !!p.pastRun };
  return { kind: "isolines", ...ISO_F };
}

const viewBox = (m: MapLibreMap): [number, number, number, number] => {
  const b = m.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
};

function decodeVector(g: GridSpec, codes: Uint16Array): Float32Array {
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i++) out[i] = codes[i] === g.nodata ? NaN : codes[i] * g.scale_factor + g.add_offset;
  return out;
}

async function loadCurrent(base: string, f: CurrentFrame): Promise<CurrentField> {
  const [u, v] = await Promise.all([loadCodes(base, f.u), loadCodes(base, f.v)]);
  return { g: f.u as unknown as CurrentField["g"], u: decodeVector(f.u, u.codes), v: decodeVector(f.v, v.codes) };
}

type Reading =
  | { status: "loading" }
  | { status: "outside" }
  | { status: "error" }
  | {
      status: "ready";
      /** the tap this reading answers (a new tap starts over; a new frame at the same tap updates in place) */
      at: { lat: number; lon: number };
      /** the layer and the frame the value was read from */
      product: ProductId;
      frame: string;
      value: number | null;
      typical: number | null;
      /** the cell that was read: its centre, size and outline (what the value stands for) */
      cell: CellGeometry;
      /** where a cell without a value sits in its layer's grid */
      gap: GapPosition | null;
      /** a measured cell with no value that is the pixel beside land the pipeline leaves out (not cloud) */
      besideLand: boolean;
      /** a measured cell without a value that is land, or past the coastal product's water (not cloud) */
      outsideArea: boolean;
      pixel: string | null;
      transect: TransectPoint[];
      /** the next frame's value is being read: the last one stays, dimmed (no "Reading…" flash while playing) */
      updating: boolean;
    };

/** URL state: ?layer=chl&frame=olci:latest&cur=0&hour=<id>&at=36.80,-121.95 (with the shell's place). */
function readUrl(model: ExploreModel, sp: URLSearchParams) {
  // older Explore links name the model "forecast" and a sensor for chlorophyll
  const alias: Record<string, ProductId> = { forecast: "charm", olci300: "chl", viirs750: "chl", multi: "chl" };
  const asked = sp.get("layer") ?? "";
  const p = model.products.find((x) => x.id === (alias[asked.split(":")[0]] ?? asked)) ?? model.products[0] ?? null;
  const f = p?.frames.find((x) => x.id === sp.get("frame")) ?? p?.frames.find((x) => x.id === p.defaultFrame) ?? null;
  const hour = model.currents?.frames.find((x) => x.id === sp.get("hour"))?.id ?? model.currents?.defaultFrame ?? null;
  const m = /^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$/.exec(sp.get("at") ?? "");
  const cur = sp.get("cur");
  const hourFrame = model.currents?.frames.find((x) => x.id === hour) ?? null;
  // an explicit ?cur=1 / ?cur=0 wins; otherwise the default for this layer and this hour's freshness
  const currents = !!model.currents && (cur === "1" ? true : cur === "0" ? false : currentsByDefault(p, hourFrame));
  return { product: p?.id ?? null, frame: f?.id ?? null, currents, hour, at: m ? { lat: Number(m[1]), lon: Number(m[2]) } : null, view: readView(sp), from: readFrom(sp), stations: sp.get("st") === "1", chart: asked === "chart", rail: sp.get("rail") === "1" };
}

export function Explore({ model }: { model: ExploreModel }) {
  const sp = useSearchParams();
  const { port, key, fromUrl } = useCurrentPort();
  useEffect(() => {
    if (fromUrl && key) savePort(key);
  }, [fromUrl, key]);
  const init = useMemo(() => readUrl(model, new URLSearchParams(sp.toString())), [model]); // eslint-disable-line react-hooks/exhaustive-deps
  const [productId, setProductId] = useState<ProductId | null>(init.product);
  // round 2 (A): the plain bathymetric chart, no data layer (the switch's first option, kept on this device)
  const [chartOnly, setChartOnly] = useState(init.chart);
  const chartOnlyRef = useRef(chartOnly);
  chartOnlyRef.current = chartOnly;
  const [opacity, setOpacity] = useState(1);
  // round 2 (B, D): the live model and estimate layers (lib/ocean), read from the app's own routes after mount
  const [ocean, setOcean] = useState<{ wcofs: WcofsIndex | null; chl: ChlEstimateIndex | null; loaded: boolean; offline?: boolean }>({ wcofs: null, chl: null, loaded: false });
  // offline or a saved copy: the live model routes are not in the field pack, so they are never asked for (and a
  // cached model run is never shown as current); the layers say they need a connection
  const online = useOnline();
  const savedCopy = useSavedMark();
  const [oceanField, setOceanField] = useState<null | "sst" | "chlest">(null);
  const oceanFieldRef = useRef(oceanField);
  oceanFieldRef.current = oceanField;
  const [curSource, setCurSource] = useState<"measured" | "predicted">("measured");
  const [stepId, setStepId] = useState<string | null>(null);
  const [failedSteps, setFailedSteps] = useState<Set<string>>(() => new Set());
  useEffect(() => {
    let live = true;
    if (navigator.onLine === false || (window as unknown as { __NERITIC_SAVED__?: unknown }).__NERITIC_SAVED__ || !online || savedCopy) {
      setOcean({ wcofs: null, chl: null, loaded: true, offline: true });
      setOceanField((f) => (f ? null : f));
      setCurSource("measured");
      return;
    }
    Promise.allSettled([loadWcofsIndex(), loadChlEstimateIndex()]).then(([w, c]) => {
      if (!live) return;
      const wcofs = w.status === "fulfilled" ? w.value : null;
      setOcean({ wcofs, chl: c.status === "fulfilled" ? c.value : null, loaded: true });
      // open on the served step nearest to now (live every listed step is served; a fixture serves a few), never on
      // one that cannot be read; the ones tried and not served are marked
      if (wcofs && wcofs.status !== "unavailable" && wcofs.steps.length)
        void pickServedStep(wcofs.steps, model.now, (st) =>
          loadOceanCodes(st.grids.mask).then(
            () => true,
            () => {
              if (live) setFailedSteps((f) => new Set(f).add(st.id));
              return false;
            },
          ),
        ).then((st) => live && st && setStepId((x) => x ?? st.id));
    });
    return () => {
      live = false;
    };
    // once, on arrival (the clock is the page's own)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [online, savedCopy]);
  const wtracks = useMemo(() => wcofsTracks(ocean.wcofs), [ocean.wcofs]);
  const stepRef = useRef(stepId);
  stepRef.current = stepId;
  const failedRef = useRef(failedSteps);
  failedRef.current = failedSteps;
  const wstep = ocean.wcofs?.steps.find((x) => x.id === stepId) ?? null;
  const predicted = curSource === "predicted" && !!wstep;
  const depthOn = useMapLayer("depth");
  const regOn = useMapLayer("regulation");
  const setChartElement = (id: "depth" | "regulation", on: boolean) => {
    setMapLayer(id, on);
    writePrefs({ [id]: on });
  };
  const changeOpacity = (v: number) => {
    setOpacity(v);
    writePrefs({ opacity: v });
  };
  const [frameId, setFrameId] = useState<string | null>(init.frame);
  const [currentsOn, setCurrentsOn] = useState(init.currents);
  const [hourId, setHourId] = useState<string | null>(init.hour);
  const [at, setAt] = useState<{ lat: number; lon: number } | null>(init.at);
  // v15.1: the whole coast (Oregon border to Mexico) or the port's water; where "Back to …" goes
  const [view, setView] = useState<ExploreView>(init.view);
  const from = init.from;
  const [stationsOn, setStationsOn] = useState(init.stations);
  // the timeline: a slim strip statewide until opened, open at a port (null: the view's default)
  const [railOpen, setRailOpen] = useState<boolean | null>(null);
  // round 2 (E): the time control is a slim strip until the reader opens it (kept on the device; ?rail=1 opens it)
  const railShown = railOpen ?? init.rail;
  const [pick, setPick] = useState<StationPick | null>(null);
  // the camera's zoom and latitude, for the legend's resolution line (rounded, so a pan does not re-render)
  const [scale, setScale] = useState<{ zoom: number; lat: number } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<TransectPoint | null>(null);
  const [map, setMap] = useState<MapLibreMap | null>(null);
  const [tier, setTier] = useState<GraphicsTier | null>(null);
  const reduce = useReducedMotion() ?? false;
  const phone = usePhone();
  const narrow = useMedia("(max-width: 340px)");
  // round 3: the statewide fit measures what floats over the chart live. The shell's bar is registered as a top cover
  // while the whole coast is on show, so the camera's padding AND the chart's frame both make room for it (a padding
  // alone could not move the view north past the frame's edge once the open timeline took the foot); the time dock,
  // the panel and the phone sheet are covers already, and every change of their size refits (ResizeObserver)
  useEffect(() => {
    if (view !== "coast") return;
    const el = document.querySelector<HTMLElement>("[data-testid=nx-bar]");
    if (!el) return;
    return registerCover("explore-statewide-bar", el, "top");
  }, [view]);
  const product = model.products.find((p) => p.id === productId) ?? null;
  const frame = product?.frames.find((f) => f.id === frameId) ?? null;
  const current = model.currents?.frames.find((f) => f.id === hourId) ?? null;
  // T25: no streaks on a forecast layer; over a model or outlook the currents wait, switched on
  const overModel = product?.kind === "model" || product?.kind === "outlook";
  // and the static tier draws none (no GPU), so its legend does not promise any
  const streaks = currentsOn && !overModel && tier !== "static";
  const fieldRef = useRef<FieldController | null>(null);
  const flowRef = useRef<FlowController | null>(null);
  const [field, setField] = useState<CurrentField | null>(null);
  // "ready" with `cells` 0 is an hour the radar published with no measurement in it
  const [fieldState, setFieldState] = useState<{ status: "idle" | "loading" | "failed" } | { status: "ready"; cells: number }>({ status: "idle" });
  const [isoLabels, setIsoLabels] = useState<IsoLabel[]>([]);
  const readoutRef = useRef<HTMLDivElement>(null);
  const switchRef = useRef<HTMLDivElement>(null);

  // ------------------------------------------------------------- URL state (the Explore selection survives a reload and a link)
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const set = (k: string, v: string | null) => (v === null ? q.delete(k) : q.set(k, v));
    set("layer", chartOnly ? "chart" : productId && productId !== model.products[0]?.id ? productId : null);
    set("frame", frame && product && frame.id !== product.defaultFrame ? frame.id : null);
    const byDefault = currentsByDefault(product, model.currents?.frames.find((x) => x.id === model.currents?.defaultFrame) ?? null);
    set("cur", model.currents && currentsOn !== byDefault ? (currentsOn ? "1" : "0") : null);
    set("hour", hourId && hourId !== model.currents?.defaultFrame ? hourId : null);
    set("at", at ? `${at.lat.toFixed(4)},${at.lon.toFixed(4)}` : null);
    set("view", view === "coast" ? "coast" : null);
    set("st", stationsOn ? "1" : null);
    const next = `${window.location.pathname}${q.toString() ? `?${q}` : ""}`;
    if (next !== `${window.location.pathname}${window.location.search}`) window.history.replaceState(window.history.state, "", next);
  }, [productId, frame, product, currentsOn, hourId, at, model, view, stationsOn, chartOnly]);

  // the reader's own choices from an earlier visit, where the address does not name them (prefs.ts). The address as
  // it ARRIVED decides (the URL sync above drops `layer=chl`, the default, before this runs): a link always wins
  useEffect(() => {
    const q = new URLSearchParams(sp.toString());
    const pr = readPrefs();
    if (!q.has("layer")) {
      if (pr.base === "chart") setChartOnly(true);
      else if (model.products.some((p) => p.id === pr.base) && pr.base !== productId) {
        const p = model.products.find((x) => x.id === pr.base)!;
        setProductId(p.id);
        setFrameId(p.defaultFrame);
      }
    }
    setOpacity(pr.opacity);
    if (!q.has("rail") && pr.rail !== null) setRailOpen(pr.rail);
    if (!q.has("st") && pr.stations) setStationsOn(true);
    if (!q.has("cur") && pr.currents !== null && model.currents) setCurrentsOn(pr.currents);
    setMapLayer("depth", pr.depth);
    setMapLayer("regulation", pr.regulation);
    // once, on arrival
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------------------------------------------------------------- the chart
  useEffect(() => {
    const t = capability().tier;
    setTier(t);
    if (t === "static") return;
    let live = true;
    whenChartReady().then((m) => live && setMap(m));
    return () => {
      live = false;
    };
  }, []);

  // the field layer: once, then a new frame swaps its data (a cut)
  useEffect(() => {
    if (!map) return;
    const layer = createFieldLayer("nx-field", modeColours());
    fieldRef.current = layer;
    const remove = addChartLayer(map, layer, "field");
    const offMode = onModeChange(() => layer.setColours(modeColours()));
    return () => {
      offMode();
      remove();
      fieldRef.current = null;
    };
  }, [map]);

  // each track's westernmost pixel per row, from its fully loaded latest view: west of it is outside the layer's
  // area (a plain tint), not cloud (the hatch) (chart/layers/marks.ts)
  const westRef = useRef(new Map<string, Int32Array>());
  // the statewide frame's progress while it loads (the field is drawn once, whole, then fades in)
  const [fieldLoad, setFieldLoad] = useState<{ loaded: number; total: number } | null>(null);
  useEffect(() => {
    const layer = fieldRef.current;
    if (!map || !layer || !product || !frame?.grid || chartOnly || oceanField) {
      layer?.setData(null, { kind: "ramp", domain: CHL_DOMAIN, scale: "log" });
      setIsoLabels([]);
      setFieldLoad(null);
      return;
    }
    let live = true;
    const g = frame.grid;
    const st = styleFor(product, frame);
    const aux = frame.typicalGrid ?? null;
    const coast = view === "coast";
    const trackKey = `${product.id}:${frame.track}`;
    // the band the pipeline left out beside land is drawn as its own mark, not as cloud (adjacency.ts); the
    // reference is read only for a frame whose QC says the rule was applied, and without it nothing changes
    const maskP = frame.besideLandLeftOut ? loadAdjacency() : Promise.resolve(null);
    const marked = async (codes: Uint16Array, complete: boolean) => {
      const mask = await maskP;
      const d = mask ? markAdjacent(g, codes, mask) : { codes, adjacentCode: null };
      if (st.kind !== "ramp") return { ...d, outCode: null, loadingCode: null };
      if (complete && frame.label === "Latest" && !westRef.current.has(trackKey)) westRef.current.set(trackKey, westmostPerRow(g, codes, d.adjacentCode != null ? [d.adjacentCode] : []));
      const m = markForDisplay(g, d.codes, { westmost: westRef.current.get(trackKey) ?? null, loaded: complete ? null : loadedChunks(model.baseUrl, g) });
      return { codes: m.codes, adjacentCode: d.adjacentCode, outCode: m.outCode, loadingCode: m.loadingCode };
    };
    const total = g.chunks ? g.chunks.present.length : 1;
    const already = (loadedChunks(model.baseUrl, g)?.size ?? 0) >= total;
    // statewide: nothing half-drawn while it arrives (no chunk-by-chunk, no stale port picture under the flight)
    if (coast && !already) {
      layer.setData(null, st);
      layer.setFade(0);
      setFieldLoad({ loaded: loadedChunks(model.baseUrl, g)?.size ?? 0, total });
    } else {
      layer.setFade(1);
      setFieldLoad(null);
    }
    let reveal: { stop: () => void } | null = null;
    const onPart = (c: { codes: Uint16Array }) => {
      if (!live) return;
      if (coast) setFieldLoad({ loaded: loadedChunks(model.baseUrl, g)?.size ?? 0, total });
      else void marked(c.codes, false).then((d) => live && layer.setData({ grid: g, ...d }, st));
    };
    const load = () =>
      Promise.all([loadCodes(model.baseUrl, g, coast ? null : viewBox(map), onPart), aux ? loadCodes(model.baseUrl, aux).catch(() => null) : Promise.resolve(null)])
        .then(async ([c, a]) => {
          const d = await marked(c.codes, c.complete);
          if (!live) return;
          layer.setData({ grid: g, ...d, aux: a && aux ? { grid: aux, codes: a.codes } : null }, st);
          if (coast && !already) {
            // the whole coast arrives at once: a fade in (motion.dev), a brief one under reduced motion
            setFieldLoad(null);
            reveal?.stop();
            reveal = animate(0, 1, { ...(reduce ? layerOut : layerIn), onUpdate: (v) => layer.setFade(v) });
          }
          setIsoLabels(st.kind === "isolines" ? isothermLabels(g, c.codes, ISO_F.index) : []);
          (window as Win).__nxExplore = { ...(window as Win).__nxExplore, field: frame.id, fieldDraws: layer.draws, adjacentCode: d.adjacentCode, fieldComplete: c.complete };
        })
        .catch(() => live && setFieldLoad(null));
    load();
    // a large chunked grid at a port: the chunks for wherever the reader pans (statewide holds them all)
    const onMove = () => load();
    if (g.chunks && !coast) map.on("moveend", onMove);
    return () => {
      live = false;
      reveal?.stop();
      layer.setFade(1);
      map.off("moveend", onMove);
    };
  }, [map, product, frame, model.baseUrl, view, reduce, chartOnly, oceanField]);

  // the live model temperature or the gap-filled estimate in the same field layer, near shore left out (ocean.ts)
  const oceanCodesRef = useRef<{ grid: OceanGrid; codes: Uint16Array } | null>(null);
  const [oceanState, setOceanState] = useState<"idle" | "loading" | "ready" | "failed">("idle");
  useEffect(() => {
    const layer = fieldRef.current;
    if (!map || !layer || !oceanField) {
      oceanCodesRef.current = null;
      setOceanState("idle");
      return;
    }
    let live = true;
    const pick = oceanField === "sst" ? (wstep ? { g: wstep.grids.sst, m: wstep.grids.mask } : null) : ocean.chl?.grids ? { g: ocean.chl.grids.chl, m: ocean.chl.grids.mask } : null;
    const st: FieldStyle = oceanField === "sst" ? { kind: "isolines", ...ISO_F } : { kind: "outlook", domain: CHL_DOMAIN, scale: "log", opacity: 0.9 };
    layer.setData(null, st);
    setIsoLabels([]);
    if (!pick) {
      setOceanState("failed");
      return;
    }
    setOceanState("loading");
    layer.setFade(1);
    Promise.all([loadOceanCodes(pick.g), loadOceanCodes(pick.m).catch(() => null)])
      .then(([c, m]) => {
        if (!live) return;
        const codes = maskedCodes(pick.g, c, m);
        const grid = asValueGrid(pick.g) as unknown as FieldGrid;
        if (oceanField === "sst") {
          // isotherms traced on the CPU: no segment along the mask or the grid's edge, no tiny rings; their own warm
          // colour, labelled in °F (contours.ts, isotherms.ts)
          const lines = traceIsolines(grid, codes, { step: ISO_F.step, index: ISO_F.index, toValue: (k) => (k * grid.scale_factor + grid.add_offset) * ISO_F.scale + ISO_F.offset, minKm: ISO_MIN_KM });
          showIsotherms(map, lines);
          setIsoLabels(isolineLabels(lines));
          (window as Win).__nxExplore = { ...(window as Win).__nxExplore, isotherms: lines.length };
        } else layer.setData({ grid, codes }, st);
        oceanCodesRef.current = { grid: pick.g, codes };
        setOceanState("ready");
        (window as Win).__nxExplore = { ...(window as Win).__nxExplore, ocean: oceanField, oceanStep: oceanField === "sst" ? wstep?.id ?? null : ocean.chl?.date ?? null, fieldDraws: layer.draws };
      })
      .catch(() => {
        if (!live) return;
        setOceanState("failed");
        clearIsotherms(map);
        if (oceanField === "sst" && wstep) setFailedSteps((f) => new Set(f).add(wstep.id));
      });
    const offMode = onModeChange(() => recolourIsotherms(map));
    return () => {
      live = false;
      offMode();
      clearIsotherms(map);
    };
  }, [map, oceanField, wstep, ocean.chl]);
  // the reader's opacity for the field: the chart's contours show beneath it
  useEffect(() => {
    fieldRef.current?.setOpacity(opacity);
  }, [opacity, map]);

  // currents: the hour's field, read only while streaks are asked for (never on a still picture, never with the
  // switch off); a failure is a state the switch and the legend say, never streaks promised and not drawn
  useEffect(() => {
    if ((!current && !predicted) || !streaks) {
      setField(null);
      setFieldState({ status: "idle" });
      return;
    }
    let live = true;
    setFieldState({ status: "loading" });
    // measured (HF radar) or predicted (WCOFS, near shore left out): one field for the streak layer
    const fieldP = predicted && wstep ? Promise.all([loadWcofsCurrents(wstep), loadOceanCodes(wstep.grids.mask).catch(() => null)]).then(([f, m]) => ({ ...f, ...maskedVectors(f.u, f.v, m) })) : loadCurrent(model.baseUrl, current!);
    fieldP
      .then((f) => {
        if (!live) return;
        let cells = 0;
        for (let i = 0; i < f.u.length; i++) if (Number.isFinite(f.u[i]) && Number.isFinite(f.v[i])) cells++;
        setField(f);
        setFieldState({ status: "ready", cells });
      })
      .catch(() => {
        if (!live) return;
        setFieldState({ status: "failed" });
        if (predicted && wstep) {
          // a step that could not be read draws no streaks at all: an empty field, never the last one left in place
          const n = wstep.grids.u.width * wstep.grids.u.height;
          setField({ g: asValueGrid(wstep.grids.u) as unknown as CurrentField["g"], u: new Float32Array(n).fill(NaN), v: new Float32Array(n).fill(NaN) });
          setFailedSteps((f) => new Set(f).add(wstep.id));
        } else setField(null);
      });
    return () => {
      live = false;
    };
  }, [current, model.baseUrl, streaks, predicted, wstep]);

  // how steadily each cell was measured: a few hours around the one on show, read one after another once the
  // streaks are drawn (never every published hour at once)
  const [quality, setQuality] = useState<Float32Array | null>(null);
  const fieldReady = fieldState.status === "ready";
  useEffect(() => {
    if (!streaks || !model.currents || quality || !fieldReady || predicted) return;
    let live = true;
    (async () => {
      const masks: Uint8Array[] = [];
      for (const h of qualityHours(model.currents!.frames, hourId)) {
        const c = await loadCodes(model.baseUrl, h.u);
        if (!live) return;
        masks.push(Uint8Array.from(c.codes, (x) => (x === h.u.nodata ? 0 : 1)));
      }
      if (live && masks.length) setQuality(persistence(masks, masks[0].length));
    })().catch(() => undefined);
    return () => {
      live = false;
    };
    // the hours are chosen around the hour on show when the streaks first draw; a later hour keeps the mask
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streaks, model, quality, fieldReady]);

  // an hour past the radar's current limit is drawn still and labelled "(old)", never animated as if live (T17)
  const currentOld = !predicted && !!current && isOld(current.fresh);
  // and the 24 h mean is not one moment of the water: drawn still too (lead decision 7)
  const drawnStill = !predicted && !!current && currentsStill(current);
  const flowState = useMemo(() => ({ still: reduce || drawnStill, paused: false }), [reduce, drawnStill]);
  // the streak layer is added once while currents are on; a new hour swaps its field (a cut)
  useEffect(() => {
    if (!map || !streaks || !tier) return;
    flowRef.current ??= createFlowLayer("nx-flow", tier === "reduced" ? "reduced" : "full", modeColours(), flowState);
    const layer = flowRef.current;
    const remove = addChartLayer(map, layer, "flow");
    const offMode = onModeChange(() => layer.setColours(modeColours()));
    (window as Win).__nxExplore = { ...(window as Win).__nxExplore, flow: layer.debug };
    return () => {
      offMode();
      remove();
    };
    // flowState is applied by its own effect
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, streaks, tier]);
  useEffect(() => {
    const layer = flowRef.current;
    if (!layer || !field || (!current && !predicted) || !streaks) return;
    layer.setField(predicted && wstep ? `wcofs:${wstep.id}` : current!.id, field, !predicted && quality && quality.length === field.u.length ? quality : null);
  }, [field, current, quality, streaks, map, tier, predicted, wstep]);

  // predicted currents are drawn in their own colour (the model's ultramarine), never the measured streak ink
  useEffect(() => {
    flowRef.current?.setColours(predicted ? { ...modeColours(), streak: token("--ultra") } : modeColours());
  }, [predicted, streaks, map, tier, field]);

  // isotherms and isobaths are both lines: under the temperature the depth lines step back
  useEffect(() => {
    if (!map || (product?.id !== "sst" && oceanField !== "sst")) return;
    const dim: Array<[string, number]> = [
      ["isobaths-minor", 0.14],
      ["isobaths-index", 0.32],
    ];
    const was = dim.map(([id]) => (map.getLayer(id) ? map.getPaintProperty(id, "line-opacity") : undefined));
    for (const [id, o] of dim) if (map.getLayer(id)) map.setPaintProperty(id, "line-opacity", o);
    return () => {
      dim.forEach(([id], i) => {
        if (map.style && map.getLayer(id) && was[i] !== undefined) map.setPaintProperty(id, "line-opacity", was[i] as number);
      });
    };
  }, [map, product?.id, oceanField]);
  useEffect(() => {
    flowRef.current?.setState(flowState);
  }, [flowState]);

  // tap the water
  useEffect(() => {
    if (!map) return;
    const onClick = (e: { lngLat: { lat: number; lng: number } }) => {
      setPick(null);
      // the plain chart has no data layer to read
      if (chartOnlyRef.current) return;
      setAt({ lat: e.lngLat.lat, lon: e.lngLat.lng });
      setPlaying(false);
    };
    map.on("click", onClick);
    return () => {
      map.off("click", onClick);
    };
  }, [map]);

  // the zoom the chart is drawn at, for the legend's resolution line (updated as the camera settles, rounded)
  useEffect(() => {
    let last = "";
    return onFrame((f) => {
      const z = Math.round(f.camera.zoom * 4) / 4, lat = Math.round(f.camera.center[1]);
      const k = `${z}|${lat}`;
      if (k !== last) {
        last = k;
        setScale({ zoom: z, lat });
      }
    });
  }, []);
  // the hour's measured speeds, from the very u/v the streaks are drawn from
  const speeds = useMemo(() => (field ? speedStats(field.u, field.v) : null), [field]);

  // ------------------------------------------------------------- the reading at the tapped point
  const [reading, setReading] = useState<Reading | null>(null);
  useEffect(() => {
    if (!at || !frame?.grid || oceanField) return setReading(null);
    let live = true;
    // the same tap on a new frame (playback, a layer change): the last value stays, dimmed, until the next arrives
    setReading((prev) => (prev?.status === "ready" && prev.at === at && prev.product === frame.product ? { ...prev, updating: true } : { status: "loading" }));
    const g = frame.grid;
    const box = boxAround(at.lon, at.lat, 0.6, 0.05);
    (async () => {
      const c = await loadCodes(model.baseUrl, g, box);
      const pt = readPoint(g, c.codes, at.lat, at.lon);
      if (!pt) return live && setReading({ status: "outside" });
      let pixel: string | null = null;
      if (frame.ageGrid && frame.referenceDate && pt.value !== null) {
        const a = await loadCodes(model.baseUrl, frame.ageGrid, box);
        const ap = readPoint(frame.ageGrid, a.codes, at.lat, at.lon);
        if (ap?.value != null) pixel = pixelDate(frame.referenceDate, ap.value);
      }
      let typical: number | null = null;
      if (frame.typicalGrid && pt.value !== null) {
        const t = await loadCodes(model.baseUrl, frame.typicalGrid);
        typical = readPoint(frame.typicalGrid, t.codes, at.lat, at.lon)?.value ?? null;
      }
      // no value in a frame that leaves out the pixel beside land: is this that pixel, or cloud?
      let besideLand = false;
      if (pt.value === null && frame.besideLandLeftOut) {
        const mask = await loadAdjacency();
        const cell = cellGeometry(g, pt.row, pt.col);
        besideLand = !!mask && isLandAdjacent(mask, cell.lat, cell.lon);
      }
      // no value on land, or past the coastal product's water: outside the layer's area, never "cloud" (review j10)
      let outsideArea = false;
      if (pt.value === null && frame.kind === "measured" && !besideLand) {
        const cell = cellGeometry(g, pt.row, pt.col);
        const latest = model.products.find((p) => p.id === frame.product)?.frames.find((f) => f.track === frame.track && f.label === "Latest" && f.grid) ?? null;
        let compositeEmpty = true;
        if (latest?.grid && latest.id !== frame.id) {
          const lc = await loadCodes(model.baseUrl, latest.grid, box);
          compositeEmpty = (readPoint(latest.grid, lc.codes, cell.lat, cell.lon)?.value ?? null) === null;
        }
        outsideArea = outsideObservedArea({ lat: cell.lat, lon: cell.lon, row: pt.row, col: pt.col, compositeEmpty, westmost: westRef.current.get(`${frame.product}:${frame.track}`) ?? null, shoreLon });
      }
      if (!live) return;
      setReading({
        status: "ready",
        at,
        product: frame.product,
        frame: frame.id,
        value: pt.value,
        typical,
        cell: cellGeometry(g, pt.row, pt.col),
        gap: pt.value === null && frame.kind !== "measured" ? gapPosition(g, c.codes, pt.row, pt.col) : null,
        besideLand,
        outsideArea,
        pixel,
        transect: transect(g, c.codes, at.lat, at.lon),
        updating: false,
      });
    })().catch(() => live && setReading({ status: "error" }));
    return () => {
      live = false;
    };
  }, [at, frame, model, oceanField]);

  // the nearest measured current to the tapped point, from the hour already on the chart (no second read)
  const nearCurrent = useMemo(() => (at && field && streaks ? nearestCurrent(field.g as unknown as GridSpec, field.u, field.v, at.lat, at.lon) : null), [at, field, streaks]);

  // a reading arrives: on a phone the sheet rises to half (the tapped point stays in view above it), then the
  // readout scrolls into view wherever the margin is
  useEffect(() => {
    if (reading?.status !== "ready") return;
    // phone: the sheet drops to a peek (the reading leads it, see explore.module.css) and the chart pans just
    // enough to keep the read cell clear of the sheet, so the next tap can land on the water beside it
    // On the narrowest screens the peek (30 % of a 640 px screen) ends above the reading's figure, so the sheet
    // takes its resting height there instead: the figure is in view and the chart still keeps about half the screen
    if (phone) openMargin(narrow ? "rest" : "peek");
    const cell = reading.cell;
    const t = window.setTimeout(
      () => {
        // phone: the sheet returns to its top, where the layer switch, the time and then the reading are
        if (phone) switchRef.current?.scrollIntoView({ block: "start", behavior: "auto" });
        else readoutRef.current?.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
        keepChartPointInView(cell.lon, cell.lat);
      },
      phone ? 380 : 0,
    );
    return () => window.clearTimeout(t);
    // one move per reading: a new cell, not every frame of a playback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reading?.status, reading?.status === "ready" ? reading.cell.row : null, reading?.status === "ready" ? reading.cell.col : null, reduce, phone, narrow]);

  // ------------------------------------------------------------- choosing and playing frames (cuts, never tweens: T26)
  const chooseFrame = useCallback((p: ExploreProduct, f: ExploreFrame) => {
    setChartOnly(false);
    setOceanField(null);
    setProductId(p.id);
    setFrameId(f.id);
    setPlaying(false);
  }, []);
  const chooseSstModel = () => {
    setChartOnly(false);
    setOceanField("sst");
    setAt(null);
    setPlaying(false);
    if (!stepId && ocean.wcofs?.default_step) setStepId(ocean.wcofs.default_step);
  };
  const chooseStep = (id: string) => {
    setStepId(id);
    setPlaying(false);
    if (oceanField !== "sst" && !predicted) setOceanField("sst");
  };
  const chooseChart = () => {
    setOceanField(null);
    setChartOnly(true);
    setAt(null);
    setPlaying(false);
    writePrefs({ base: "chart" });
  };
  const chooseProduct = (id: ProductId) => {
    const p = model.products.find((x) => x.id === id);
    if (!p) return;
    setChartOnly(false);
    setOceanField(null);
    writePrefs({ base: id });
    setProductId(id);
    setFrameId(p.defaultFrame);
    setPlaying(false);
    setHover(null);
  };
  const chooseHour = useCallback((f: CurrentFrame) => {
    setCurrentsOn(true);
    setHourId(f.id);
    setPlaying(false);
  }, []);

  const hours = useMemo(() => (model.currents ? playHours(model.currents.frames) : []), [model.currents]);
  const dayFrames = useMemo(() => (product ? playFrames(product) : []), [product]);
  const playsHours = streaks && !predicted && hours.length > 1 && !currentOld;
  // the WCOFS forecast steps play when the model is on show: only steps that are served (an unserved one is skipped)
  const wcofsMode = (oceanField === "sst" || predicted) && playableSteps(ocean.wcofs?.steps ?? [], failedSteps).length > 1;
  const canPlay = wcofsMode || playsHours || (!chartOnly && !oceanField && dayFrames.length > 1);
  // a model on show with fewer than two served steps left: playback ends (it never steps a layer that is not shown)
  useEffect(() => {
    if (playing && (oceanField === "sst" || predicted) && !wcofsMode) setPlaying(false);
  }, [playing, oceanField, predicted, wcofsMode]);
  useEffect(() => {
    if (!playing || !wcofsMode) return;
    let live = true;
    let t = 0;
    const advance = async () => {
      const list = playableSteps(ocean.wcofs?.steps ?? [], failedRef.current);
      const i = list.findIndex((x) => x.id === stepRef.current);
      for (let k = 1; k <= list.length; k++) {
        const next = list[(i + k + list.length) % list.length];
        // the next step is read before it is shown: playback never stops on one that cannot be read
        const ok = await loadOceanCodes(oceanField === "sst" ? next.grids.sst : next.grids.u).then(
          () => true,
          () => false,
        );
        if (!live) return;
        if (ok) {
          setStepId(next.id);
          break;
        }
        failedRef.current = new Set(failedRef.current).add(next.id);
        setFailedSteps(failedRef.current);
      }
      if (live) t = window.setTimeout(advance, 1400);
    };
    t = window.setTimeout(advance, 400);
    return () => {
      live = false;
      window.clearTimeout(t);
    };
  }, [playing, wcofsMode, ocean.wcofs, oceanField]);
  useEffect(() => {
    if (!playing || wcofsMode) return;
    const id = window.setInterval(
      () => {
        if (playsHours) setHourId((h) => nextOf(hours, h)?.id ?? h);
        else setFrameId((f) => nextOf(dayFrames, f)?.id ?? f);
      },
      playsHours ? 900 : 1400,
    );
    return () => window.clearInterval(id);
  }, [playing, playsHours, hours, dayFrames, wcofsMode]);

  // the e2e hook: what is drawn, read from the page
  useEffect(() => {
    (window as Win).__nxExplore = { ...(window as Win).__nxExplore, product: productId, frame: frameId, hour: streaks ? hourId : null, at, cell: reading?.status === "ready" ? reading.cell : null };
  }, [productId, frameId, hourId, streaks, at, reading]);

  const extent = model.extent;
  // was the statewide view on show this visit (the flight back to the port is the same deliberate one)
  const wasCoast = useRef(view === "coast");
  if (view === "coast") wasCoast.current = true;
  const scene = useMemo(
    () =>
      view === "coast"
        ? {
            // the whole coast, Oregon border to Mexico, at every screen shape; the port stays marked, so the way back is in view
            id: "explore:state",
            view: { bounds: STATE_BOUNDS },
            focus: port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null,
            zoomRange: COAST_ZOOM,
            refitLabel: "Whole coast",
            // a deliberate flight out to the state (the whole frame is fetched meanwhile and fades in once, whole)
            flightMs: STATE_FLIGHT_MS,
          }
        : {
      id: `explore:${port?.key ?? "coast"}`,
      // the view stops at the data's own edge rather than showing a sliver of chart past it
      // the data's water fills the free area (the box's longer side cropped) rather than leaving inland hills beside it
      // a port the data does not reach is framed as itself, never slid to another bay's data (camera.ts coversPort)
      view: { bounds: exploreBounds(port ? { lon: port.lon, lat: port.lat } : null, extent), within: withinFor(extent, port ? { lon: port.lon, lat: port.lat } : null), fill: !!extent && coversPort(extent, port ? { lon: port.lon, lat: port.lat } : null) },
      focus: port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null,
      zoomRange: EXPLORE_ZOOM,
      // back down from the whole coast to the port: the same deliberate flight
      ...(wasCoast.current ? { flightMs: STATE_FLIGHT_MS } : {}),
      // the chart's own buttons, with a way back to everything the layers hold
      controls: extent ? { fit: { label: "Fit the data", bounds: [[extent[0], extent[1]], [extent[2], extent[3]]] as [[number, number], [number, number]] } } : undefined,
          },
    [port, extent, view],
  );
  const back = backLink(from, key);
  const metas = useMemo(() => layerMetas(model), [model]);
  const unpublished = useMemo(() => notPublished(model), [model]);
  // the chlorophyll chip names the view on show (round 2 review): observed day, recent composite or the estimate
  const chlChipWords =
    oceanField === "chlest" && ocean.chl?.date
      ? `Estimate · ${day(ocean.chl.date)}${ocean.chl.status === "stale" ? " (stale)" : ""}`
      : frame && product?.id === "chl"
        ? chlViewOf(frame) === "composite"
          ? `Composite · ${frame.start !== frame.end ? range(frame.start, frame.end) : day(frame.end)}${oldMark(frame.fresh)}`
          : `Observation · ${day(frame.end)}${oldMark(frame.fresh)}`
        : (() => {
            const m = metas.find((x) => x.id === "chl");
            return m ? chipWords(m) : "";
          })();
  const goView = (v: ExploreView) => {
    setView(v);
    setPlaying(false);
  };
  const toggleStations = () => {
    setStationsOn((on) => {
      writePrefs({ stations: !on });
      return !on;
    });
    setPick(null);
  };
  const choosePick = (p: StationPick) => {
    setPick(p);
    setAt(null);
    setPlaying(false);
  };
  // the surface layer is finer than the screen at statewide scale: said in the legend, with the real numbers
  const cellKm = cellKmOf(frame?.grid);
  const resNote = scale && cellKm ? resolutionNote(scale.zoom, scale.lat, cellKm) : null;

  if (!product || !frame) {
    return (
      <section className={s.explore} data-testid="nx-explore">
        <ChartScene scene={scene} />
        <p className={s.kicker}>Explore</p>
        <Unavailable what="No layers are published right now" why="The latest data run has no observations or models to show." next="Check the record in Research, or come back after the next run." />
      </section>
    );
  }

  const r = reading;
  const pt = at;
  const nowIso = model.now;
  const unit = unitOf(product.id);
  const fmt = (v: number) => valueWords(product.id, v);
  const currentsState: CurrentsState = !currentsOn ? "off" : overModel ? "over-model" : tier === "static" ? "static" : fieldState.status === "failed" ? "failed" : fieldState.status === "ready" ? (fieldState.cells ? "on" : "empty") : "loading";

  let readout = null;
  // a tap on a model or estimate layer: the value of its own cell, worded as what it is (never a measurement)
  if (oceanField && pt) {
    const oc = oceanCodesRef.current;
    const pr = oc ? readPoint(asValueGrid(oc.grid) as unknown as FieldGrid, oc.codes, pt.lat, pt.lon) : null;
    const sst = oceanField === "sst";
    const title = sst ? "Sea temperature (model) here" : "Chlorophyll estimate here";
    const onClose = () => setAt(null);
    if (!oc) readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state={oceanState === "failed" ? "error" : "loading"} message={oceanState === "failed" ? "This layer could not be read." : "Reading…"} onClose={onClose} />;
    else if (!pr || pr.value === null)
      readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="outside" message={`No ${sst ? "model" : "estimate"} value here: land, within ${(sst ? ocean.wcofs?.near_shore_km : ocean.chl?.near_shore_km) ?? 0} km of shore, or outside the grid. No value is not a low value.`} onClose={onClose} />;
    else {
      const v = pr.value;
      const prov: Provenance = sst
        ? { kind: "prediction", title: "Sea temperature (model)", value: `${fixed(v * 1.8 + 32, 1)} °F`, source: "NOAA WCOFS (developmental)", date: wstep?.valid_time ?? ocean.wcofs?.valid?.start ?? model.now, dateRole: "valid", place: latLon(pt.lat, pt.lon), resolution: "one 4 km model cell", stale: ocean.wcofs?.status === "expired" }
        : { kind: "estimated", title: "Chlorophyll estimate", value: `${sig(v, 2)} mg/m³`, source: "NOAA DINEOF (gap-filled)", date: ocean.chl?.date ?? model.now, dateRole: "valid", place: latLon(pt.lat, pt.lon), resolution: "one 2.3 km cell", stale: ocean.chl?.status === "stale" };
      readout = (
        <Readout
          title={title}
          lat={pt.lat}
          lon={pt.lon}
          state="value"
          value={sst ? fixed(v * 1.8 + 32, 1) : sig(v, 2)}
          unit={sst ? "°F" : "mg/m³"}
          provenance={prov}
          now={model.now}
          note={sst ? `${fixed(v, 1)} °C. A model ${wstep?.kind === "nowcast" ? "nowcast" : "forecast"}, not a measurement.` : `Reconstructed, not observed${ocean.chl?.status === "stale" ? `; stale (${ocean.chl?.date ? day(ocean.chl.date) : ""})` : ""}. Algae, not toxin.`}
          onClose={onClose}
        />
      );
    }
  }
  if (pt && r && !oceanField) {
    // the reading is one cell of the layer's grid: its title says so, with the cell's size once it is known
    const title = readoutTitle(product.id, r.status === "ready" ? r.cell : null);
    const onClose = () => {
      setAt(null);
      setHover(null);
    };
    if (r.status === "loading") readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="loading" message={`Reading ${LAYER_LABEL[product.id].toLowerCase()} at this point…`} onClose={onClose} />;
    else if (r.status === "error") readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="error" message="This point couldn't be read. Check the connection and tap again." onClose={onClose} />;
    else if (r.status === "outside")
      readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="outside" message={frame.kind === "measured" ? `Outside this layer's area: ${frame.sensor} has no cells here. No value is not a low value.` : noValueWords(product, "outside-grid")} onClose={onClose} />;
    else {
      // while the next frame is being read, the value on show still belongs to the frame it was read from
      const rf = product.frames.find((f) => f.id === r.frame) ?? frame;
      // the record's window ends on the reading's own day; the stamp says what that day is (words.ts readingStamp)
      const date = r.pixel ?? rf.end;
      const stamp = readingStamp(rf, r.pixel);
      const noSignal = product.id === "outlook" && r.value !== null && r.typical !== null && Math.abs(Math.log10(r.value) - Math.log10(r.typical)) < 0.002;
      const prov: Provenance = {
        kind: rf.kind === "outlook" ? "outlook" : rf.kind,
        title: LAYER_LABEL[product.id],
        value: r.value !== null ? `${fmt(r.value)} ${unit}` : undefined,
        source: stamp.source,
        sourceUrl: product.sourceUrl ?? undefined,
        date: stamp.date,
        dateRole: stamp.dateRole,
        // the cell that was read, not the tap
        place: latLon(r.cell.lat, r.cell.lon),
        resolution: `one ${cellSizeWords(r.cell)} cell`,
        stale: rf.fresh === "stale" || rf.fresh === "historical",
      };
      const cur = nearCurrent;
      // the nearest current says which it is: a predicted (model) current is never worded as the measured radar
      const curLine =
        streaks && predicted && wstep && fieldState.status === "ready"
          ? cur
            ? `Predicted current (NOAA WCOFS model, developmental) · valid ${stepWhen(wstep)} · ${sig(cur.speed * KNOTS, 2)} kt toward the ${compassWord(cur.dirTo)}${cur.km < 1 ? "" : `, nearest model cell ${sig(cur.km, 2)} km away`} · a model forecast, not a measurement.`
            : `No predicted current within 6 km (the model leaves out the near-shore band).`
          : streaks && current && fieldState.status === "ready"
          ? cur
            ? `Nearest measured current ${cur.km < 1 ? "here" : `${sig(cur.km, 2)} km away`}: ${sig(cur.speed * KNOTS, 2)} kt toward the ${compassWord(cur.dirTo)} (HF radar, ${current.mean ? current.label : current.stamp}${oldMark(current.fresh)}).`
            : `No measured current within 6 km in ${current.mean ? "the 24 h mean" : "this hour"}.`
          : null;
      const note =
        product.id === "chl"
          ? `Algae, not toxin.${r.pixel ? ` This pixel was seen ${day(r.pixel)}, ${age(r.pixel, nowIso)}.` : ""}`
          : product.id === "sst"
            ? `${r.value !== null ? `${fixed(r.value, 1)} °C. ` : ""}${rf.kind === "analysis" ? "An analysis blended from satellites and buoys, not a measurement at this point." : ""}`
            : product.id === "charm"
              ? `Probability of conditions in the water, not toxin in seafood.${rf.past ? ` Past run: valid ${day(rf.end)}, before today.` : ""}`
              : noSignal
                ? `No signal: here the outlook equals the usual for this time of year (${aboutWords(r.typical!)} mg/m³); there was no recent view to start from.`
                : "Typical error about a factor of 2. Read as higher or lower than usual, not as a value. Algae, not toxin.";
      // a "nearby" record is within 25 km and measures the variable; a farther one is named with its distance
      const choice = stationFor(model.stations, r.cell.lat, r.cell.lon, RESEARCH_VAR[product.id]);
      // what was sampled: one cell of the layer's own grid
      // (on a phone the stamp below already names the grid, and the line's height is the value's room at the peek)
      const subtitle = phone ? undefined : `One cell of the ${rf.sensor} grid`;
      const action = (
        <span className={s.actions}>
          {choice.link ? (
            <Link className={s.recordLink} href={researchHref({ station: choice.link.id, to: date, product: product.id, place: key })} data-testid="nx-record-link">
              See the record nearby: {choice.link.name}, {sig(choice.link.km, 2)} km
            </Link>
          ) : choice.far ? (
            <span data-testid="nx-record-none">
              No station within {NEARBY_KM} km measures this.{" "}
              <Link className={s.recordLink} href={researchHref({ station: choice.far.id, to: date, product: product.id, place: key })} data-testid="nx-record-far">
                The nearest that does: {choice.far.name}, {sig(choice.far.km, 2)} km away
              </Link>
            </span>
          ) : choice.unknown ? (
            // the station list carries toxin and Pseudo-nitzschia only, so for this measurement no pier is KNOWN to
            // measure it (some may): said from the reader's side, with the nearest pier's record to look in
            <span data-testid="nx-record-unknown">
              No pier nearby is known to measure {MEASURE_WORDS[product.id]}.{" "}
              <Link className={s.recordLink} href={researchHref({ station: choice.unknown.id, to: date, product: product.id, place: key })} data-testid="nx-record-look">
                Nearest pier&rsquo;s record: {choice.unknown.name}, {sig(choice.unknown.km, 2)} km{choice.unknown.km > NEARBY_KM ? " away" : ""}
              </Link>
            </span>
          ) : (
            <span data-testid="nx-record-none">No monitoring station measures this.</span>
          )}
          <Link className={s.recordLink} href={coastHref({ place: key, lat: r.cell.lat, lon: r.cell.lon, date })} data-testid="nx-coast-link">
            Use this place in My Coast
          </Link>
        </span>
      );
      const xs = r.transect;
      const tr =
        xs.length > 2 ? (
          <Transect
            points={xs}
            unit={unit}
            scale={product.id === "chl" || product.id === "outlook" ? "log" : "linear"}
            label={`Along ${latLon(r.cell.lat, r.cell.lon).split(",")[0]}, west to east`}
            observed={rf.kind === "measured"}
            format={fmt}
            onHover={setHover}
            hoverKm={hover?.km ?? null}
          />
        ) : null;
      readout =
        r.value === null ? (
          <Readout
            title={title}
            lat={r.cell.lat}
            lon={r.cell.lon}
            state={rf.kind === "measured" && !r.besideLand && !r.outsideArea ? "noview" : "outside"}
            subtitle={subtitle}
            actions={action}
            message={r.besideLand ? ADJACENT_WORDS : r.outsideArea ? OUTSIDE_AREA_WORDS : rf.kind === "measured" ? `No view here (${rf.label === "Latest" ? `latest view, ${rf.stamp}` : rf.stamp}): cloud, fog or no pass. No value is not a low value.` : noValueWords(product, r.gap ?? "inside")}
            onClose={onClose}
          >
            {tr}
          </Readout>
        ) : (
          <Readout
            title={title}
            lat={r.cell.lat}
            lon={r.cell.lon}
            state="value"
            value={noSignal ? "No signal" : product.id === "outlook" ? aboutWords(r.value) : fmt(r.value)}
            unit={noSignal ? undefined : unit}
            provenance={prov}
            now={nowIso}
            note={
              <>
                {note}
                {curLine ? <span className={s.curLine}>{curLine}</span> : null}
              </>
            }
            subtitle={subtitle}
            actions={action}
            onClose={onClose}
          >
            {tr}
          </Readout>
        );
    }
  }

  const cell = r?.status === "ready" ? r.cell : null;
  // the line on the chart spans the water the transect has values for: never across land or past the grid's water
  const valued = r?.status === "ready" ? r.transect.filter((p) => p.value !== null) : [];
  const tStart = valued.length ? valued[0].lon : null;
  const tEnd = valued.length ? valued[valued.length - 1].lon : null;
  // the reading's ticket stands on the side of its point with the more open chart (it follows the point; the camera stays)
  const f = getFrame();
  const readAt = cell ?? pt;
  const readSide: "east" | "west" = readAt && f && f.project(readAt.lon, readAt.lat).x > f.viewport.width * 0.55 ? "west" : "east";
  const dockH = !railShown ? 96 : streaks ? 284 : 204;
  // the live model and estimate rows of the time rail (round 2, D/E): WCOFS nowcast and the estimate on Analysis /
  // Estimates, WCOFS forecast on Forecasts; a source that cannot be read says so on its row
  const wcofsOk = wcofsUsable(ocean.wcofs);
  const chooseEstimate = () => {
    setChartOnly(false);
    if (model.products.some((x) => x.id === "chl")) setProductId("chl");
    setOceanField("chlest");
    setAt(null);
  };
  // the live model steps and the estimate's day at their true times on the rail's shared axis (round 2, lead's review)
  const oceanRail = {
    nowcast: wcofsOk ? wtracks.nowcast : [],
    forecast: wcofsOk ? wtracks.forecast : [],
    selected: oceanField === "sst" || predicted ? stepId : null,
    failed: failedSteps,
    onStep: chooseStep,
    estimate: chlEstimateUsable(ocean.chl) && ocean.chl?.date ? { date: ocean.chl.date, words: chlEstimateStateWords(ocean.chl, ocean.loaded), on: oceanField === "chlest", onChoose: chooseEstimate } : null,
  };
  // the rail's axis holds everything shown at its true time (WCOFS forecasts run days past today)
  const railModel: ExploreModel = { ...model, axis: extendAxis(model.axis, wcofsOk ? (ocean.wcofs?.steps ?? []).map((x) => x.valid_time) : [], oceanRail.estimate ? [oceanRail.estimate.date] : []) };
  const oceanExtra = {
    est: !chlEstimateUsable(ocean.chl) ? <EstimateRow ix={ocean.chl} loaded={ocean.loaded} on={false} onChoose={chooseEstimate} offline={ocean.offline} /> : null,
    // a recorded test cycle says so on its own row (live there is no reason to show)
    fc: wcofsOk ? (ocean.wcofs?.reason ? <p className={s.oceanNote} data-testid="nx-wcofs-reason">{ocean.wcofs.reason}</p> : null) : <WcofsNote ix={ocean.wcofs} loaded={ocean.loaded} offline={ocean.offline} />,
  };
  const pickAt = pick ? ((pick.kind === "shore" ? model.stations.find((x) => x.id === pick.id) : model.buoys.find((x) => x.id === pick.id)) ?? null) : null;
  const stationSlot =
    pick && pickAt ? (
      <div className={s.readoutSlot} data-open="1" data-testid="nx-station-slot">
        {phone ? (
          <StationReadout pick={pick} stations={model.stations} buoys={model.buoys} now={nowIso} place={key} onClose={() => setPick(null)} />
        ) : (
          <Ticket at={{ lon: pickAt.lon, lat: pickAt.lat }} side={f && f.project(pickAt.lon, pickAt.lat).x > f.viewport.width - 400 ? "west" : "east"} lead={false} anchorY={48} width={340} label="Station" className={s.readTicket}>
            <TicketMain>
              <StationReadout pick={pick} stations={model.stations} buoys={model.buoys} now={nowIso} place={key} onClose={() => setPick(null)} />
            </TicketMain>
          </Ticket>
        )}
      </div>
    ) : null;
  const readSlot = stationSlot ?? (
    <div ref={readoutRef} className={s.readoutSlot} data-testid="nx-readout" data-open={readout ? "1" : "0"} data-updating={r?.status === "ready" && r.updating ? "1" : "0"} aria-busy={r?.status === "ready" && r.updating}>
      {readout && readAt ? (
        phone ? (
          readout
        ) : (
          <Ticket at={{ lon: readAt.lon, lat: readAt.lat }} side={readSide} lead={false} anchorY={48} width={372} label="Reading at this point" className={s.readTicket}>
            <TicketMain>{readout}</TicketMain>
          </Ticket>
        )
      ) : null}
    </div>
  );
  return (
    <section className={s.explore} data-testid="nx-explore" data-nx-explore-field={chartOnly ? "0" : "1"} data-base={chartOnly ? "chart" : product.id} data-product={product.id} data-frame={frame.id} style={{ "--explore-dock": `${dockH}px` } as React.CSSProperties}>
      {/* statewide on a phone, the sheet rests lower so most of the coast shows; it still rises to read */}
      <PageLayout mode="stage" phoneMap={view === "coast" ? 0.64 : 0.5} />
      {/* the base chart's own elements, switched in Explore's panel (one set of layer controls) */}
      <MapLayers page="explore" defaults={{ depth: true, regulation: false }} />
      <ChartRegulation spec={port && RAMP_ZONES.length ? { zones: { highlight: rampZoneAt(port.lat) } } : null} />
      <ChartScene scene={scene} />
      {/* the mark sits on the cell that was read (its centre, its outline), not on the tap */}
      {pt ? <ReadingOnChart lat={cell?.lat ?? pt.lat} lon={cell?.lon ?? pt.lon} lon0={tStart} lon1={tEnd} hoverLon={hover?.lon ?? null} outline={cell?.outline ?? null} /> : null}
      {/* the whole coast is its own locator: the inset only at a port */}
      {phone && view !== "coast" ? <Locator /> : null}
      {stationsOn ? <StationPoints stations={model.stations} buoys={model.buoys} selected={pick} onSelect={choosePick} /> : null}
      {(product.id === "sst" || oceanField === "sst") && isoLabels.length ? <ChartMarks marks={isoLabels.map((l) => ({ id: l.id, lon: l.lon, lat: l.lat, priority: 12, anchors: ["center" as const], content: <span className={oceanField === "sst" ? `${s.isoLabel} ${s.isoLabelWarm}` : s.isoLabel}>{l.text}</span>, label: `${l.text} F isotherm` }))} /> : null}

      {/* the layer group: what this view is, the data layers as chips, the legend (a ticket standing west, with no place) */}
      <Ticket at={null} side="west" edge="gutter" width={348} labelledBy="nx-explore-title" className={s.panel} data-testid="nx-explore-panel">
        <TicketMain>
          <div className={s.head}>
            {/* the layer on show is part of the heading's name ("Explore California Waters: C-HARM"); the switch shows it */}
            <h1 id="nx-explore-title" className={s.title} data-testid="nx-explore-heading" aria-label={`Explore California Waters: ${LAYER_LABEL[product.id]}`}>
              Explore California Waters
            </h1>
            <div className={s.nav} data-testid="nx-explore-nav">
              {view === "coast" ? (
                port ? (
                  <Button variant="quiet" onClick={() => goView("port")} data-testid="nx-back-port">
                    Back to {port.name}
                  </Button>
                ) : null
              ) : (
                <Button variant="quiet" onClick={() => goView("coast")} data-testid="nx-whole-coast">
                  Whole coast
                </Button>
              )}
              <Button variant="quiet" href={back.href} data-testid="nx-back-from">
                {back.label}
              </Button>
            </div>
          </div>
          {/* one short line in view; the full sentence, the method and the caveats are in "About this layer" */}
          <p className={s.lead} data-testid="nx-explore-lead">
            {chartOnly ? "The chart: depth, the coast and whale zones · no data layer" : oceanField === "sst" ? "Model sea temperature · WCOFS, developmental · not a measurement" : oceanField === "chlest" ? "Gap-filled chlorophyll estimate · not observed · not toxin" : shortLine(product)}
          </p>
          {fieldLoad ? (
            <p className={s.fieldLoading} role="status" data-testid="nx-field-loading">
              <span>
                Loading the whole coast · {fieldLoad.loaded} of {fieldLoad.total}
              </span>
              <span className={s.fieldBar} aria-hidden="true">
                <span style={{ transform: `scaleX(${fieldLoad.total ? fieldLoad.loaded / fieldLoad.total : 0})` }} />
              </span>
            </p>
          ) : null}
          {/* the layer has no cells at the reader's port: said in words, with where it does have them */}
          {!chartOnly && port && frame.grid && !coversPort(gridBounds(frame.grid), { lon: port.lon, lat: port.lat }) ? (
            <p className={s.coverage} data-testid="nx-coverage-note">
              {coverageNote(LAYER_LABEL[product.id], port.name, gridBounds(frame.grid))}
            </p>
          ) : null}
          <div ref={switchRef} className={s.switch} role="radiogroup" aria-label="Data layer" data-testid="nx-data-layers">
            {/* round 2 (A): the plain bathymetric chart first, one tap from any layer */}
            {/* design-lint-allow raw-button: one radio of the data-layer group (see below) */}
            <button type="button" role="radio" aria-checked={chartOnly} className={s.switchItem} onClick={chooseChart} data-testid="nx-layer-chart">
              <span>Chart</span>
              <small>no data layer</small>
            </button>
            {model.products.map((p) => {
              const meta = metas.find((m) => m.id === p.id);
              return (
                // design-lint-allow raw-button: the data layers are one radio group of chips whose items carry a second line (the layer's kind) and per-layer test hooks, with a switch (currents) in the same row; ds Chip is a toggle, not a radio
                <button key={p.id} type="button" role="radio" aria-checked={!chartOnly && oceanField !== "sst" && p.id === product.id} className={s.switchItem} onClick={() => chooseProduct(p.id)} data-testid={`nx-layer-${p.id}`} data-class={meta?.productClass}>
                  <span>{LAYER_LABEL[p.id]}</span>
                  {/* observation or model, then its own dates ("Model · run of Nov 3, valid Nov 2–5 (past run)") */}
                  <small>{p.id === "chl" ? chlChipWords : meta ? chipWords(meta) : p.pastRun && p.kind !== "measured" && p.kind !== "analysis" ? "past run" : layerKindWord(p)}</small>
                </button>
              );
            })}
            {/* round 2 (D): NOAA WCOFS model temperature, read live; a model, never offered when it cannot be read */}
            {ocean.loaded ? (
              // design-lint-allow raw-button: one radio of the data-layer group (see above)
              <button type="button" role="radio" aria-checked={oceanField === "sst"} disabled={!wcofsUsable(ocean.wcofs)} data-missing={wcofsUsable(ocean.wcofs) ? undefined : "1"} className={s.switchItem} onClick={chooseSstModel} data-testid="nx-layer-sst-model" data-class="model">
                <span>Sea temp</span>
                <small>{wcofsUsable(ocean.wcofs) ? `Model · WCOFS${ocean.wcofs?.status === "expired" ? " (expired)" : ", developmental"}` : ocean.offline ? "needs a connection" : "model unavailable"}</small>
              </button>
            ) : null}
            {/* a layer this data run does not hold stays in the group, disabled, with the reason (never dropped silently) */}
            {/* (a layer not published yet is named once, in the line under the switch, never as a chip that looks like a layer) */}
            {model.missing.filter((m) => m.why !== "Not published yet").map((m) => (
              <button key={m.id} type="button" role="radio" aria-checked={false} disabled className={s.switchItem} data-missing="1" data-testid={`nx-layer-${m.id}`}>
                <span>{m.label}</span>
                <small>{m.why.toLowerCase()}</small>
              </button>
            ))}
          </div>
          {/* chlorophyll's three views (round 2, B): what was seen that day, each pixel's latest clear day, a reconstruction */}
          {!chartOnly && product.id === "chl" && oceanField !== "sst" ? (
            <div className={s.views} role="radiogroup" aria-label="Chlorophyll view" data-testid="nx-chl-views">
              {(["observed", "composite", "estimate"] as const).map((v) => {
                const target = v === "estimate" ? (chlEstimateUsable(ocean.chl) ? "estimate" : null) : frameForView(product, v, frame.track);
                const on = v === "estimate" ? oceanField === "chlest" : !oceanField && chlViewOf(frame) === v;
                const label = v === "observed" ? "Observed" : v === "composite" ? "Composite" : "Estimate";
                const sub = v === "observed" ? "one day, gaps kept" : v === "composite" ? "recent, pixel ages" : chlEstimateUsable(ocean.chl) ? `gap-filled · ${ocean.chl?.status === "stale" ? "stale" : "estimate"}` : ocean.offline ? "needs a connection" : "not available";
                return (
                  // design-lint-allow raw-button: one radio of a three-way view switch with a second line per item (as the data-layer chips)
                  <button key={v} type="button" role="radio" aria-label={v === "observed" ? "Observed: one day, gaps kept" : v === "composite" ? "Recent composite: each pixel's latest clear day, with its age" : chlEstimateUsable(ocean.chl) ? `Gap-filled estimate: reconstructed, not observed${ocean.chl?.status === "stale" ? ", stale" : ""}` : "Gap-filled estimate: not available"} aria-checked={on} disabled={!target} className={s.viewItem} data-missing={!target ? "1" : undefined} onClick={() => {
                      if (!target) return;
                      if (v === "estimate") {
                        setOceanField("chlest");
                        setAt(null);
                        setPlaying(false);
                        return;
                      }
                      setOceanField(null);
                      setFrameId(target);
                    }} data-testid={`nx-chl-${v}`}>
                    <span>{label}</span>
                    <small>{sub}</small>
                  </button>
                );
              })}
            </div>
          ) : null}
          {/* the chart's own elements and the overlays, switched on and off over any base (one place, not two systems) */}
          <div className={`${s.switch} ${s.chartLayers}`} role="group" aria-label="On the chart" data-testid="nx-chart-layers">
            {/* design-lint-allow raw-button: a switch in the same chip row as the layers */}
            <button type="button" role="switch" aria-checked={depthOn} className={s.switchItem} data-on={depthOn ? "1" : "0"} onClick={() => setChartElement("depth", !depthOn)} data-testid="nx-depth-toggle">
              <span>Depth</span>
            </button>
            {/* design-lint-allow raw-button: a switch in the same chip row as the layers */}
            <button type="button" role="switch" aria-checked={regOn} className={s.switchItem} data-on={regOn ? "1" : "0"} onClick={() => setChartElement("regulation", !regOn)} data-testid="nx-regulation-toggle">
              <span>Whale zones</span>
            </button>
            {model.currents ? (
              <button type="button" role="switch" aria-checked={currentsOn} className={s.switchItem} data-on={currentsOn ? "1" : "0"} onClick={() => setCurrentsOn((v) => { writePrefs({ currents: !v }); return !v; })} data-testid="nx-currents-toggle">
                <span>Currents</span>
                <small data-state={currentsState}>{predicted && currentsState === "on" ? "on, predicted (model)" : currentsToggleWords(currentsState, current)}</small>
              </button>
            ) : null}
            {model.stations.length || model.buoys.length ? (
              // design-lint-allow raw-button: an overlay switch in the same chip row as the layers (see above)
              <button type="button" role="switch" aria-checked={stationsOn} className={s.switchItem} data-on={stationsOn ? "1" : "0"} onClick={toggleStations} data-testid="nx-stations-toggle">
                <span>Stations</span>
              </button>
            ) : null}
          </div>
          {currentsOn && model.currents ? (
            <div className={s.views} role="radiogroup" aria-label="Currents" data-testid="nx-currents-source">
              {/* design-lint-allow raw-button: one radio of the currents' two sources */}
              <button type="button" role="radio" aria-checked={curSource === "measured"} className={s.viewItem} onClick={() => setCurSource("measured")} data-testid="nx-currents-measured" aria-label="Measured currents: HF radar">
                <span>Measured</span>
              </button>
              {/* design-lint-allow raw-button: one radio of the currents' two sources */}
              <button type="button" role="radio" aria-checked={curSource === "predicted"} disabled={!wcofsUsable(ocean.wcofs)} data-missing={wcofsUsable(ocean.wcofs) ? undefined : "1"} className={s.viewItem} onClick={() => setCurSource("predicted")} data-testid="nx-currents-predicted" aria-label={ocean.offline ? "Predicted currents: needs a connection, live model data is read when online" : "Predicted currents: NOAA WCOFS model, developmental"}>
                <span>Predicted (model)</span>
              </button>
            </div>
          ) : null}
          {!chartOnly ? (
            <label className={s.opacity} data-testid="nx-opacity">
              <span>Layer opacity</span>
              <input type="range" min={20} max={100} step={5} value={Math.round(opacity * 100)} onChange={(e) => changeOpacity(Number(e.target.value) / 100)} aria-valuetext={`${Math.round(opacity * 100)} percent`} />
              <b>{Math.round(opacity * 100)} %</b>
            </label>
          ) : null}
          {/* phone: what is on show and Play sit with the layers at the top of the sheet (review F-01) */}
          {phone ? (
            <div className={s.timeBar}>
              <TimeHead stamp={oceanField === "sst" && wstep ? `Model sea temperature: valid ${stepWhen(wstep)}` : oceanField === "chlest" && ocean.chl?.date ? `Estimate: ${day(ocean.chl.date)}${ocean.chl.status === "stale" ? " (stale)" : ""}` : predicted && wstep && !(chartOnly || !!oceanField) ? `Predicted currents: valid ${stepWhen(wstep)}` : null} model={model} product={product} frameId={chartOnly ? "" : frame.id} currentsOn={streaks && !predicted} currentId={hourId} playing={playing} canPlay={canPlay} onPlay={() => setPlaying((p) => !p)} />
            </div>
          ) : null}
          {phone ? readSlot : null}
        </TicketMain>
        <TicketStub>
          <div data-testid="nx-legend-sticky">
            {chartOnly ? (
              <div className={s.legendBlock} data-testid="nx-legend" data-product="chart">
                <p className={s.chartTitle}>Chart</p>
                <ul className={s.keys} aria-label="Marks">
                  <li>
                    <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true">
                      <path d="M0,11 C10,6 22,13 40,7" className={s.swatchDepth} />
                    </svg>
                    <span>Depth contours, fathoms</span>
                  </li>
                  <li>
                    <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true">
                      <path d="M0,8 H40" className={s.swatchReg} />
                    </svg>
                    <span>Whale-zone limits</span>
                  </li>
                </ul>
                <p className={s.caption}>Depth: NOAA ETOPO 2022 and the Coastal Relief Model. Not for navigation.</p>
              </div>
            ) : oceanField === "sst" && ocean.wcofs ? (
              <SstModelLegend ix={ocean.wcofs} step={wstep} failed={oceanState === "failed"} />
            ) : oceanField === "chlest" && ocean.chl ? (
              <ChlEstimateLegend ix={ocean.chl} />
            ) : (
            <ExploreLegend
              product={product}
              frame={frame}
              currents={currentsState === "on" && !predicted ? current : null}
              currentsLine={
                current ? (
                  <>
                    Measured currents · {current.mean ? current.label : current.stamp}
                    {oldWord(current.fresh)}
                    {speeds ? (
                      <span data-testid="nx-currents-speed">
                        {" "}
                        · median {speedWords(speeds.median)}, top {sig(speeds.max * KNOTS, 2)} kn
                      </span>
                    ) : null}
                    {/* a streak is at most one 2 km cell long (T27): statewide that is a speck, so the reader is told where they show */}
                    {scale && scale.zoom < 7 ? <b className={s.zoomHint}> · zoom in toward the coast to see the streaks</b> : null}
                  </>
                ) : undefined
              }
              about={
                <>
                  <p data-testid="nx-lead-full">{leadSentence(product, frame)}</p>
                  {currentsState === "on" && speeds ? (
                    <p>
                      Speed this {current?.mean ? "24 h mean" : "hour"}: median {speedWords(speeds.median)}, fastest tenth above {speedWords(speeds.p90)}, top {speedWords(speeds.max)}; {speeds.cells.toLocaleString("en-US")} cells measured. Streaks move faster where the water does.
                    </p>
                  ) : null}
                  {resNote ? <p data-testid="nx-resolution-note">{resNote}</p> : null}
                  {charmStalledWords(product) ? <p data-testid="nx-charm-stalled">{charmStalledWords(product)}</p> : null}
                  {product.id === "chl" && !model.products.some((x) => x.id === "outlook") ? <p data-testid="nx-outlook-na">{OUTLOOK_NOT_AVAILABLE}</p> : null}
                  {unpublished.length ? <p data-testid="nx-not-published">Not published yet: {unpublished.join(" and ")}.</p> : null}
                </>
              }
            />
            )}
            {/* predicted currents: their own key, colour and words (a model, never the measured streaks) */}
            {predicted && wstep && streaks ? (
              <p className={s.currentsKey} data-testid="nx-currents-predicted-key">
                <svg viewBox="0 0 40 16" className={s.swatch} aria-hidden="true">
                  <path d="M3,12 C14,11 24,7 37,4" className={s.streakPredicted} />
                </svg>
                <span>
                  Predicted currents · {ocean.wcofs?.cycle ? wcofsLabel(ocean.wcofs.cycle.id, wstep) : stepWhen(wstep)}
                  {ocean.wcofs?.reason ? ` · ${ocean.wcofs.reason}` : ""}
                  {fieldState.status === "failed" ? " · this step could not be read; nothing is drawn" : ""}
                  {speeds ? ` · median ${speedWords(speeds.median)}` : ""}. Not drawn within {ocean.wcofs?.near_shore_km ?? 0} km of shore.
                </span>
              </p>
            ) : null}
            {stationsOn ? (
              <p className={s.stationKey} data-testid="nx-stations-key">
                <span className={s.stationKeyShore} aria-hidden="true" /> Piers
                <span className={s.stationKeyBuoy} aria-hidden="true" /> Buoys · tap one for its dated values
              </p>
            ) : null}
          </div>
          {tier === "static" && frame?.grid && product.id !== "sst" ? (
            <StaticField base={model.baseUrl} grid={frame.grid} kind={product.id === "charm" ? "classes" : product.id === "outlook" ? "outlook" : "ramp"} past={product.id === "charm" ? frame.past || !!product.pastRun : undefined} domain={CHL_DOMAIN} />
          ) : null}
          {tier === "static" ? (
            product.id === "sst" ? (
              <Unavailable what="Isotherms need a device that draws the chart live" why="This device shows the chart as a still picture (no WebGL, data saver, or a saved copy); the other layers are baked into it." next="Station temperatures are in Research, as tables and series." />
            ) : (
              <p className={s.prompt} data-testid="nx-static-note">
                A still picture of this frame, baked on this device (no WebGL, data saver, or a saved copy). Tapping the water needs the live chart.
              </p>
            )
          ) : null}
          {!readout && tier !== "static" && !chartOnly ? <p className={s.prompt}>Tap the water to read one cell.</p> : null}
        </TicketStub>
      </Ticket>

      {/* the reading: a ticket anchored to the cell that was read, with its links to Research and My Coast (on a phone, a
          block that leads the sheet with the layers and Play: see the panel) */}
      {!phone ? readSlot : null}

      {/* the time rail: a dock along the chart's foot, full width; on a phone it reads in the sheet */}
      <Dock label="Time" align="full" className={s.timeDock} data-testid="nx-time-dock" data-open={railShown ? "1" : "0"} data-view={view}>
        {/* statewide, the time is a slim strip (what is on show, Play) until the reader opens the timeline */}
        {!phone ? (
          <div className={s.railToggle}>
            <Button variant="quiet" aria-expanded={railShown} onClick={() => {
                setRailOpen(!railShown);
                writePrefs({ rail: !railShown });
              }} data-testid="nx-rail-toggle">
              {railShown ? "Hide timeline" : "Timeline"}
            </Button>
          </div>
        ) : null}
        {!railShown && !phone ? (
          <TimeHead stamp={oceanField === "sst" && wstep ? `Model sea temperature: valid ${stepWhen(wstep)}` : oceanField === "chlest" && ocean.chl?.date ? `Estimate: ${day(ocean.chl.date)}${ocean.chl.status === "stale" ? " (stale)" : ""}` : predicted && wstep && !(chartOnly || !!oceanField) ? `Predicted currents: valid ${stepWhen(wstep)}` : null} model={model} product={product} frameId={chartOnly ? "" : frame.id} currentsOn={streaks && !predicted} currentId={hourId} playing={playing} canPlay={canPlay} onPlay={() => setPlaying((p) => !p)} />
        ) : railShown || phone ? (
        <TimeRail
          head={!phone}
          product={product}
          frameId={chartOnly || oceanField ? "" : frame.id}
          stamp={oceanField === "sst" && wstep ? `Model sea temperature: valid ${stepWhen(wstep)}` : oceanField === "chlest" && ocean.chl?.date ? `Estimate: ${day(ocean.chl.date)}${ocean.chl.status === "stale" ? " (stale)" : ""}` : predicted && wstep && !(chartOnly || !!oceanField) ? `Predicted currents: valid ${stepWhen(wstep)}` : null}
          extra={oceanExtra}
          ocean={oceanRail}
          model={railModel}
          zoomTo={oceanField === "sst" || predicted ? stepsAxis(ocean.wcofs?.steps ?? []) : null}
          axisNote={oceanField === "sst" || predicted ? "Axis zoomed to the WCOFS steps" : null}
          onFrame={chooseFrame}
          onProduct={chooseProduct}
          currentsOn={streaks && !predicted}
          currentId={hourId}
          onCurrent={chooseHour}
          playing={playing}
          canPlay={canPlay}
          onPlay={() => setPlaying((p) => !p)}
        />
        ) : null}
      </Dock>
    </section>
  );
}

export function ExploreUnavailable({ why }: { why: string }) {
  return (
    <section className={s.explore} data-testid="nx-explore">
      <p className={s.kicker}>Explore</p>
      <Unavailable what="The water can't be shown right now" why={`The published data could not be read (${plainError(why)}).`} next="Try again in a few minutes; the data is republished several times a day." />
    </section>
  );
}

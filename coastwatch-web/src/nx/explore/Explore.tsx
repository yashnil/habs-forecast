"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "motion/react";
import type { Map as MapLibreMap } from "maplibre-gl";
import { age, ChartDock, MarginSticky, day, fixed, latLon, plainError, sig, usePhone, Readout, Unavailable, type Provenance } from "@/nx/ds";
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
import { loadCodes, boxAround } from "./codes";
import { coversPort, EXPLORE_ZOOM, exploreBounds, withinFor } from "./camera";
import { gridBounds } from "@/nx/chart/layers/grid";
import { ExploreLegend } from "./ExploreLegend";
import { Locator, ReadingOnChart } from "./ChartOverlay";
import type { CurrentFrame, ExploreFrame, ExploreModel, ExploreProduct, GridSpec, ProductId } from "./model";
import { cellGeometry, cellSizeWords, coastHref, compassWord, gapPosition, isothermLabels, KNOTS, NEARBY_KM, nearestCurrent, type CellGeometry, type GapPosition, type IsoLabel, pixelDate, readPoint, researchHref, RESEARCH_VAR, stationFor, transect, type TransectPoint } from "./read";
import { nextOf, playFrames, playHours, qualityHours } from "./timeline";
import { TimeRail } from "./TimeRail";
import { Transect } from "./Transect";
import { aboutWords, CHL_DOMAIN, coverageNote, currentsByDefault, currentsStill, currentsToggleWords, isOld, ISO_F, LAYER_LABEL, layerKindWord, leadSentence, noValueWords, oldMark, readingStamp, readoutTitle, unitOf, valueWords, type CurrentsState } from "./words";
import s from "./explore.module.css";

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
  return { product: p?.id ?? null, frame: f?.id ?? null, currents, hour, at: m ? { lat: Number(m[1]), lon: Number(m[2]) } : null };
}

export function Explore({ model }: { model: ExploreModel }) {
  const sp = useSearchParams();
  const { port, key, fromUrl } = useCurrentPort();
  useEffect(() => {
    if (fromUrl && key) savePort(key);
  }, [fromUrl, key]);
  const init = useMemo(() => readUrl(model, new URLSearchParams(sp.toString())), [model]); // eslint-disable-line react-hooks/exhaustive-deps
  const [productId, setProductId] = useState<ProductId | null>(init.product);
  const [frameId, setFrameId] = useState<string | null>(init.frame);
  const [currentsOn, setCurrentsOn] = useState(init.currents);
  const [hourId, setHourId] = useState<string | null>(init.hour);
  const [at, setAt] = useState<{ lat: number; lon: number } | null>(init.at);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<TransectPoint | null>(null);
  const [map, setMap] = useState<MapLibreMap | null>(null);
  const [tier, setTier] = useState<GraphicsTier | null>(null);
  const reduce = useReducedMotion() ?? false;
  const phone = usePhone();
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

  // ------------------------------------------------------------- URL state (the Explore selection survives a reload and a link)
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const set = (k: string, v: string | null) => (v === null ? q.delete(k) : q.set(k, v));
    set("layer", productId && productId !== model.products[0]?.id ? productId : null);
    set("frame", frame && product && frame.id !== product.defaultFrame ? frame.id : null);
    const byDefault = currentsByDefault(product, model.currents?.frames.find((x) => x.id === model.currents?.defaultFrame) ?? null);
    set("cur", model.currents && currentsOn !== byDefault ? (currentsOn ? "1" : "0") : null);
    set("hour", hourId && hourId !== model.currents?.defaultFrame ? hourId : null);
    set("at", at ? `${at.lat.toFixed(4)},${at.lon.toFixed(4)}` : null);
    const next = `${window.location.pathname}${q.toString() ? `?${q}` : ""}`;
    if (next !== `${window.location.pathname}${window.location.search}`) window.history.replaceState(window.history.state, "", next);
  }, [productId, frame, product, currentsOn, hourId, at, model]);

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

  useEffect(() => {
    const layer = fieldRef.current;
    if (!map || !layer || !product || !frame?.grid) {
      layer?.setData(null, { kind: "ramp", domain: CHL_DOMAIN, scale: "log" });
      setIsoLabels([]);
      return;
    }
    let live = true;
    const g = frame.grid;
    const st = styleFor(product, frame);
    const aux = frame.typicalGrid ?? null;
    const load = () =>
      Promise.all([loadCodes(model.baseUrl, g, viewBox(map), (c) => live && layer.setData({ grid: g, codes: c.codes }, st)), aux ? loadCodes(model.baseUrl, aux).catch(() => null) : Promise.resolve(null)])
        .then(([c, a]) => {
          if (!live) return;
          layer.setData({ grid: g, codes: c.codes, aux: a && aux ? { grid: aux, codes: a.codes } : null }, st);
          setIsoLabels(st.kind === "isolines" ? isothermLabels(g, c.codes, ISO_F.index) : []);
          (window as Win).__nxExplore = { ...(window as Win).__nxExplore, field: frame.id, fieldDraws: layer.draws };
        })
        .catch(() => undefined);
    load();
    // a large chunked grid: the chunks for wherever the reader pans
    const onMove = () => load();
    if (g.chunks) map.on("moveend", onMove);
    return () => {
      live = false;
      map.off("moveend", onMove);
    };
  }, [map, product, frame, model.baseUrl]);

  // currents: the hour's field, read only while streaks are asked for (never on a still picture, never with the
  // switch off); a failure is a state the switch and the legend say, never streaks promised and not drawn
  useEffect(() => {
    if (!current || !streaks) {
      setField(null);
      setFieldState({ status: "idle" });
      return;
    }
    let live = true;
    setFieldState({ status: "loading" });
    loadCurrent(model.baseUrl, current)
      .then((f) => {
        if (!live) return;
        let cells = 0;
        for (let i = 0; i < f.u.length; i++) if (Number.isFinite(f.u[i]) && Number.isFinite(f.v[i])) cells++;
        setField(f);
        setFieldState({ status: "ready", cells });
      })
      .catch(() => {
        if (!live) return;
        setField(null);
        setFieldState({ status: "failed" });
      });
    return () => {
      live = false;
    };
  }, [current, model.baseUrl, streaks]);

  // how steadily each cell was measured: a few hours around the one on show, read one after another once the
  // streaks are drawn (never every published hour at once)
  const [quality, setQuality] = useState<Float32Array | null>(null);
  const fieldReady = fieldState.status === "ready";
  useEffect(() => {
    if (!streaks || !model.currents || quality || !fieldReady) return;
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
  const currentOld = !!current && isOld(current.fresh);
  // and the 24 h mean is not one moment of the water: drawn still too (lead decision 7)
  const drawnStill = !!current && currentsStill(current);
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
    if (!layer || !field || !current || !streaks) return;
    layer.setField(current.id, field, quality && quality.length === field.u.length ? quality : null);
  }, [field, current, quality, streaks, map, tier]);

  // isotherms and isobaths are both lines: under the temperature the depth lines step back
  useEffect(() => {
    if (!map || product?.id !== "sst") return;
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
  }, [map, product?.id]);
  useEffect(() => {
    flowRef.current?.setState(flowState);
  }, [flowState]);

  // tap the water
  useEffect(() => {
    if (!map) return;
    const onClick = (e: { lngLat: { lat: number; lng: number } }) => {
      setAt({ lat: e.lngLat.lat, lon: e.lngLat.lng });
      setPlaying(false);
    };
    map.on("click", onClick);
    return () => {
      map.off("click", onClick);
    };
  }, [map]);

  // ------------------------------------------------------------- the reading at the tapped point
  const [reading, setReading] = useState<Reading | null>(null);
  useEffect(() => {
    if (!at || !frame?.grid) return setReading(null);
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
        pixel,
        transect: transect(g, c.codes, at.lat, at.lon),
        updating: false,
      });
    })().catch(() => live && setReading({ status: "error" }));
    return () => {
      live = false;
    };
  }, [at, frame, model]);

  // the nearest measured current to the tapped point, from the hour already on the chart (no second read)
  const nearCurrent = useMemo(() => (at && field && streaks ? nearestCurrent(field.g as unknown as GridSpec, field.u, field.v, at.lat, at.lon) : null), [at, field, streaks]);

  // a reading arrives: on a phone the sheet rises to half (the tapped point stays in view above it), then the
  // readout scrolls into view wherever the margin is
  useEffect(() => {
    if (reading?.status !== "ready") return;
    // phone: the sheet drops to a peek (the reading leads it, see explore.module.css) and the chart pans just
    // enough to keep the read cell clear of the sheet, so the next tap can land on the water beside it
    if (phone) openMargin("peek");
    const cell = reading.cell;
    const t = window.setTimeout(
      () => {
        readoutRef.current?.scrollIntoView({ block: phone ? "start" : "nearest", behavior: reduce || phone ? "auto" : "smooth" });
        keepChartPointInView(cell.lon, cell.lat);
      },
      phone ? 380 : 0,
    );
    return () => window.clearTimeout(t);
    // one move per reading: a new cell, not every frame of a playback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reading?.status, reading?.status === "ready" ? reading.cell.row : null, reading?.status === "ready" ? reading.cell.col : null, reduce, phone]);

  // ------------------------------------------------------------- choosing and playing frames (cuts, never tweens: T26)
  const chooseFrame = useCallback((p: ExploreProduct, f: ExploreFrame) => {
    setProductId(p.id);
    setFrameId(f.id);
    setPlaying(false);
  }, []);
  const chooseProduct = (id: ProductId) => {
    const p = model.products.find((x) => x.id === id);
    if (!p) return;
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
  const playsHours = streaks && hours.length > 1 && !currentOld;
  const canPlay = playsHours || dayFrames.length > 1;
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(
      () => {
        if (playsHours) setHourId((h) => nextOf(hours, h)?.id ?? h);
        else setFrameId((f) => nextOf(dayFrames, f)?.id ?? f);
      },
      playsHours ? 900 : 1400,
    );
    return () => window.clearInterval(id);
  }, [playing, playsHours, hours, dayFrames]);

  // the e2e hook: what is drawn, read from the page
  useEffect(() => {
    (window as Win).__nxExplore = { ...(window as Win).__nxExplore, product: productId, frame: frameId, hour: streaks ? hourId : null, at, cell: reading?.status === "ready" ? reading.cell : null };
  }, [productId, frameId, hourId, streaks, at, reading]);

  const extent = model.extent;
  const scene = useMemo(
    () => ({
      id: `explore:${port?.key ?? "coast"}`,
      // the view stops at the data's own edge rather than showing a sliver of chart past it
      // the data's water fills the free area (the box's longer side cropped) rather than leaving inland hills beside it
      // a port the data does not reach is framed as itself, never slid to another bay's data (camera.ts coversPort)
      view: { bounds: exploreBounds(port ? { lon: port.lon, lat: port.lat } : null, extent), within: withinFor(extent, port ? { lon: port.lon, lat: port.lat } : null), fill: !!extent && coversPort(extent, port ? { lon: port.lon, lat: port.lat } : null) },
      focus: port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null,
      zoomRange: EXPLORE_ZOOM,
      // the chart's own buttons, with a way back to everything the layers hold
      controls: extent ? { fit: { label: "Fit the data", bounds: [[extent[0], extent[1]], [extent[2], extent[3]]] as [[number, number], [number, number]] } } : undefined,
    }),
    [port, extent],
  );

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
  if (pt && r) {
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
      const curLine =
        streaks && current && fieldState.status === "ready"
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
      const subtitle = `One cell of the ${rf.sensor} grid`;
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
            // the station list carries toxin and Pseudo-nitzschia only: for this measurement no pier is claimed
            <span data-testid="nx-record-unknown">
              Neritic&rsquo;s station list does not say which piers measure this.{" "}
              <Link className={s.recordLink} href={researchHref({ station: choice.unknown.id, to: date, product: product.id, place: key })} data-testid="nx-record-look">
                Look in the nearest pier&rsquo;s record: {choice.unknown.name}, {sig(choice.unknown.km, 2)} km{choice.unknown.km > NEARBY_KM ? " away" : ""}
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
            state={rf.kind === "measured" ? "noview" : "outside"}
            subtitle={subtitle}
            actions={action}
            message={rf.kind === "measured" ? `No view here (${rf.label === "Latest" ? `latest view, ${rf.stamp}` : rf.stamp}): cloud, fog or no pass. No value is not a low value.` : noValueWords(product, r.gap ?? "inside")}
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
  return (
    <section className={s.explore} data-testid="nx-explore" data-product={product.id} data-frame={frame.id}>
      <ChartScene scene={scene} />
      {/* the mark sits on the cell that was read (its centre, its outline), not on the tap */}
      {pt ? <ReadingOnChart lat={cell?.lat ?? pt.lat} lon={cell?.lon ?? pt.lon} lon0={tStart} lon1={tEnd} hoverLon={hover?.lon ?? null} outline={cell?.outline ?? null} /> : null}
      {phone ? <Locator /> : null}
      {product.id === "sst" && isoLabels.length ? <ChartMarks marks={isoLabels.map((l) => ({ id: l.id, lon: l.lon, lat: l.lat, priority: 12, anchors: ["center" as const], content: <span className={s.isoLabel}>{l.text}</span>, label: `${l.text} F isotherm` }))} /> : null}
      <h1 className={s.srOnly}>Explore: {LAYER_LABEL[product.id]}</h1>
      <p className={s.lead} data-testid="nx-explore-lead">
        {leadSentence(product, frame)}
      </p>
      {/* the layer has no cells at the reader's port: said in words, with where it does have them */}
      {port && frame.grid && !coversPort(gridBounds(frame.grid), { lon: port.lon, lat: port.lat }) ? (
        <p className={s.coverage} data-testid="nx-coverage-note">
          {coverageNote(LAYER_LABEL[product.id], port.name, gridBounds(frame.grid))}
        </p>
      ) : null}
      {/* the legend stays at the head of the margin while a reading is open, so the ramp is always in sight */}
      <MarginSticky edge="top" data-testid="nx-legend-sticky">
        <ExploreLegend product={product} frame={frame} currents={currentsState === "on" ? current : null} />
      </MarginSticky>
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
      <div className={s.switch} role="radiogroup" aria-label="Layer" data-testid="nx-layers">
        {model.products.map((p) => (
          // design-lint-allow raw-button: the layer switch is one radio group whose items carry a second line (the layer's kind) and per-layer test hooks, with a switch (currents) in the same row; ds Segmented and OptionList are single-line
          <button key={p.id} type="button" role="radio" aria-checked={p.id === product.id} className={s.switchItem} onClick={() => chooseProduct(p.id)} data-testid={`nx-layer-${p.id}`}>
            <span>{LAYER_LABEL[p.id]}</span>
            {/* the tab says only "past run"; its date is in the legend and on the rail (critic N8: no 3-line tab) */}
            <small>{p.pastRun && p.kind !== "measured" && p.kind !== "analysis" ? "past run" : layerKindWord(p)}</small>
          </button>
        ))}
        {/* a layer this data run does not hold stays in the switch, disabled, with the reason (never dropped silently) */}
        {model.missing.map((m) => (
          <button key={m.id} type="button" role="radio" aria-checked={false} disabled className={s.switchItem} data-missing="1" data-testid={`nx-layer-${m.id}`}>
            <span>{m.label}</span>
            <small>{m.why.toLowerCase()}</small>
          </button>
        ))}
        {model.currents ? (
          <button type="button" role="switch" aria-checked={currentsOn} className={s.switchItem} data-on={currentsOn ? "1" : "0"} onClick={() => setCurrentsOn((v) => !v)} data-testid="nx-currents-toggle">
            <span>Currents</span>
            <small data-state={currentsState}>{currentsToggleWords(currentsState, current)}</small>
          </button>
        ) : null}
      </div>
      {/* wide: the rail runs along the foot of the chart it steps, full width; on a phone it reads in the sheet */}
      <ChartDock label="Time" height={streaks ? 284 : 204} align="frame" variant="rail">
      <TimeRail
        model={model}
        product={product}
        frameId={frame.id}
        onFrame={chooseFrame}
        onProduct={chooseProduct}
        currentsOn={streaks}
        currentId={hourId}
        onCurrent={chooseHour}
        playing={playing}
        canPlay={canPlay}
        onPlay={() => setPlaying((p) => !p)}
      />
      </ChartDock>
      <div ref={readoutRef} className={s.readoutSlot} data-testid="nx-readout" data-open={readout ? "1" : "0"} data-updating={r?.status === "ready" && r.updating ? "1" : "0"} aria-busy={r?.status === "ready" && r.updating}>
        {readout ?? <p className={s.prompt}>Tap the water for its value, its date and a transect along its latitude.</p>}
      </div>
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

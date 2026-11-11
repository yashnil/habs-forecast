"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "motion/react";
import type { Map as MapLibreMap } from "maplibre-gl";
import { age, day, fixed, latLon, pct, sig, usePhone, Readout, Unavailable, type Provenance } from "@/nx/ds";
import { capability, type GraphicsTier } from "@/lib/capability";
import type { CurrentField } from "@/lib/currents";
import { ChartScene } from "@/nx/chart/scene";
import { whenChartReady } from "@/nx/chart/control";
import { addChartLayer } from "@/nx/chart/layers/registry";
import { modeColours, onModeChange } from "@/nx/chart/layers/colors";
import { createFieldLayer, type FieldController, type FieldStyle } from "@/nx/chart/layers/FieldLayer";
import { createFlowLayer, type FlowController } from "@/nx/chart/layers/FlowLayer";
import { persistence } from "@/nx/chart/layers/grid";
import { savePort, useCurrentPort } from "@/nx/shell/port";
import { loadCodes, boxAround } from "./codes";
import { EXPLORE_ZOOM, exploreBounds } from "./camera";
import { ExploreLegend } from "./ExploreLegend";
import { Locator, ReadingOnChart } from "./ChartOverlay";
import type { CurrentFrame, ExploreFrame, ExploreModel, ExploreProduct, GridSpec, ProductId } from "./model";
import { compassWord, KNOTS, nearestCurrent, nearestStation, pixelDate, readPoint, researchHref, transect, type NearestCurrent, type TransectPoint } from "./read";
import { nextOf, playFrames, playHours } from "./timeline";
import { TimeRail } from "./TimeRail";
import { Transect } from "./Transect";
import { aboutWords, CHL_DOMAIN, ISO_F, LAYER_LABEL } from "./words";
import s from "./explore.module.css";

type Win = Window & { __nxExplore?: Record<string, unknown> };

/** How each product is drawn (src/nx/chart/layers/FieldLayer.ts). */
function styleFor(p: ExploreProduct, f: ExploreFrame): FieldStyle {
  if (p.id === "chl") return { kind: "ramp", domain: CHL_DOMAIN, scale: "log" };
  if (p.id === "outlook") return { kind: "outlook", domain: CHL_DOMAIN, scale: "log", opacity: 0.9 };
  if (p.id === "charm") return { kind: "classes", past: f.past };
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
  | { status: "ready"; value: number | null; typical: number | null; cellLat: number; cellLon: number; pixel: string | null; transect: TransectPoint[]; current: NearestCurrent; station: ReturnType<typeof nearestStation> };

/** URL state: ?layer=chl&frame=olci:latest&cur=0&hour=<id>&at=36.80,-121.95 (with the shell's place). */
function readUrl(model: ExploreModel, sp: URLSearchParams) {
  const p = model.products.find((x) => x.id === sp.get("layer")) ?? model.products[0] ?? null;
  const f = p?.frames.find((x) => x.id === sp.get("frame")) ?? p?.frames.find((x) => x.id === p.defaultFrame) ?? null;
  const hour = model.currents?.frames.find((x) => x.id === sp.get("hour"))?.id ?? model.currents?.defaultFrame ?? null;
  const m = /^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$/.exec(sp.get("at") ?? "");
  return { product: p?.id ?? null, frame: f?.id ?? null, currents: sp.get("cur") !== "0" && !!model.currents, hour, at: m ? { lat: Number(m[1]), lon: Number(m[2]) } : null };
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
  const streaks = currentsOn && !overModel;
  const fieldRef = useRef<FieldController | null>(null);
  const flowRef = useRef<FlowController | null>(null);
  const [field, setField] = useState<CurrentField | null>(null);
  const readoutRef = useRef<HTMLDivElement>(null);

  // ------------------------------------------------------------- URL state (the Explore selection survives a reload and a link)
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const set = (k: string, v: string | null) => (v === null ? q.delete(k) : q.set(k, v));
    set("layer", productId && productId !== model.products[0]?.id ? productId : null);
    set("frame", frame && product && frame.id !== product.defaultFrame ? frame.id : null);
    set("cur", model.currents && !currentsOn ? "0" : null);
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

  // currents: the hour's field (and how steadily each cell was measured across the published hours)
  useEffect(() => {
    if (!current) return setField(null);
    let live = true;
    loadCurrent(model.baseUrl, current)
      .then((f) => live && setField(f))
      .catch(() => live && setField(null));
    return () => {
      live = false;
    };
  }, [current, model.baseUrl]);

  const [quality, setQuality] = useState<Float32Array | null>(null);
  useEffect(() => {
    if (!currentsOn || !model.currents || quality) return;
    let live = true;
    const hours = model.currents.frames.filter((f) => !f.mean);
    Promise.all(hours.map((h) => loadCodes(model.baseUrl, h.u).then((c) => Uint8Array.from(c.codes, (x) => (x === h.u.nodata ? 0 : 1)))))
      .then((masks) => live && setQuality(persistence(masks, masks[0]?.length ?? 0)))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [currentsOn, model, quality]);

  const flowState = useMemo(() => ({ still: reduce, paused: false }), [reduce]);
  // the streak layer is added once while currents are on; a new hour swaps its field (a cut)
  useEffect(() => {
    if (!map || !streaks || !tier || tier === "static") return;
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
    setReading({ status: "loading" });
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
      const cur = field && streaks ? nearestCurrent(field.g as unknown as GridSpec, field.u, field.v, at.lat, at.lon) : null;
      if (!live) return;
      setReading({ status: "ready", value: pt.value, typical, cellLat: pt.lat, cellLon: pt.lon, pixel, transect: transect(g, c.codes, at.lat, at.lon), current: cur, station: nearestStation(model.stations, at.lat, at.lon) });
    })().catch(() => live && setReading({ status: "error" }));
    return () => {
      live = false;
    };
  }, [at, frame, model, field, streaks]);

  useEffect(() => {
    if (reading?.status === "ready") readoutRef.current?.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
  }, [reading?.status, reduce]);

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
  const playsHours = streaks && hours.length > 1;
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
    (window as Win).__nxExplore = { ...(window as Win).__nxExplore, product: productId, frame: frameId, hour: streaks ? hourId : null };
  }, [productId, frameId, hourId, streaks]);

  const extent = model.extent;
  const scene = useMemo(
    () => ({
      id: `explore:${port?.key ?? "coast"}`,
      view: { bounds: exploreBounds(port ? { lon: port.lon, lat: port.lat } : null, extent) },
      focus: port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null,
      zoomRange: EXPLORE_ZOOM,
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
  const valueWords = (v: number) => (product.id === "chl" ? sig(v, 2) : product.id === "outlook" ? aboutWords(v).replace("about ", "") : product.id === "charm" ? pct(v * 100).replace(" %", "") : fixed(v * 1.8 + 32, 1));
  const unit = product.id === "charm" ? "%" : product.id === "sst" ? "°F" : "mg/m³";

  let readout = null;
  if (pt && r) {
    const title = product.id === "chl" ? "Chlorophyll here" : product.id === "sst" ? "Sea temperature here" : product.id === "charm" ? "C-HARM here" : "Outlook here";
    const onClose = () => {
      setAt(null);
      setHover(null);
    };
    if (r.status === "loading") readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="loading" message={`Reading ${LAYER_LABEL[product.id].toLowerCase()} at this point…`} onClose={onClose} />;
    else if (r.status === "error") readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="error" message="This point couldn't be read. Check the connection and tap again." onClose={onClose} />;
    else if (r.status === "outside") readout = <Readout title={title} lat={pt.lat} lon={pt.lon} state="noview" message={`Outside this layer's area: ${frame.sensor} has no cells here. No value is not a low value.`} onClose={onClose} />;
    else {
      const date = r.pixel ?? frame.end;
      const noSignal = product.id === "outlook" && r.value !== null && r.typical !== null && Math.abs(Math.log10(r.value) - Math.log10(r.typical)) < 0.002;
      const prov: Provenance = {
        kind: frame.kind === "outlook" ? "outlook" : frame.kind,
        title: LAYER_LABEL[product.id],
        value: r.value !== null ? `${valueWords(r.value)} ${unit}` : undefined,
        source: frame.sensor,
        sourceUrl: product.sourceUrl ?? undefined,
        date,
        dateRole: frame.kind === "measured" ? "measured" : frame.kind === "model" ? "valid" : undefined,
        place: latLon(r.cellLat, r.cellLon),
        stale: frame.fresh === "stale" || frame.fresh === "historical",
      };
      const cur = r.current;
      const curLine =
        streaks && current
          ? cur
            ? `Nearest measured current ${cur.km < 1 ? "here" : `${sig(cur.km, 2)} km away`}: ${sig(cur.speed * KNOTS, 2)} kt toward the ${compassWord(cur.dirTo)} (HF radar, ${current.mean ? current.label : current.stamp}).`
            : `No measured current within 6 km in ${current.mean ? "the 24 h mean" : "this hour"}.`
          : null;
      const note =
        product.id === "chl"
          ? `Algae, not toxin.${r.pixel ? ` This pixel was seen ${day(r.pixel)}, ${age(r.pixel, nowIso)}.` : ""}`
          : product.id === "sst"
            ? `${r.value !== null ? `${fixed(r.value, 1)} °C. ` : ""}${frame.kind === "analysis" ? "An analysis blended from satellites and buoys, not a measurement at this point." : ""}`
            : product.id === "charm"
              ? `Probability of conditions in the water, not toxin in seafood.${frame.past ? ` Past run: valid ${day(frame.end)}, before today.` : ""}`
              : noSignal
                ? `No signal: here the outlook equals the usual for this time of year (${aboutWords(r.typical!)} mg/m³); there was no recent view to start from.`
                : "Typical error about a factor of 2. Read as higher or lower than usual, not as a value. Algae, not toxin.";
      const st = r.station;
      const action = st ? (
        <Link className={s.recordLink} href={researchHref({ station: st.id, to: date, product: product.id, place: key })} data-testid="nx-record-link">
          See the record nearby: {st.name}, {sig(st.km, 2)} km
        </Link>
      ) : null;
      const xs = r.transect;
      const tr =
        xs.length > 2 ? (
          <Transect
            points={xs}
            unit={unit}
            scale={product.id === "chl" || product.id === "outlook" ? "log" : "linear"}
            label={`Along ${latLon(pt.lat, pt.lon).split(",")[0]}, west to east`}
            observed={frame.kind === "measured"}
            format={(v) => valueWords(v)}
            onHover={setHover}
            hoverKm={hover?.km ?? null}
          />
        ) : null;
      readout =
        r.value === null ? (
          <Readout
            title={title}
            lat={pt.lat}
            lon={pt.lon}
            state="noview"
            message={frame.kind === "measured" ? `No view here (${frame.label === "Latest" ? `latest view, ${frame.stamp}` : frame.stamp}): cloud, fog or no pass. No value is not a low value.` : `Outside the model's area here (it leaves out the nearshore band and the bays). No value is not a low value.`}
            onClose={onClose}
          >
            {tr}
          </Readout>
        ) : (
          <Readout
            title={title}
            lat={pt.lat}
            lon={pt.lon}
            state="value"
            value={noSignal ? "No signal" : product.id === "outlook" ? aboutWords(r.value) : valueWords(r.value)}
            unit={noSignal ? undefined : unit}
            provenance={prov}
            now={nowIso}
            note={
              <>
                {note}
                {curLine ? <span className={s.curLine}>{curLine}</span> : null}
              </>
            }
            action={action}
            onClose={onClose}
          >
            {tr}
          </Readout>
        );
    }
  }

  const tStart = r?.status === "ready" && r.transect.length ? r.transect[0].lon : null;
  const tEnd = r?.status === "ready" && r.transect.length ? r.transect[r.transect.length - 1].lon : null;
  return (
    <section className={s.explore} data-testid="nx-explore" data-product={product.id} data-frame={frame.id}>
      <ChartScene scene={scene} />
      {pt ? <ReadingOnChart lat={pt.lat} lon={pt.lon} lon0={tStart} lon1={tEnd} hoverLon={hover?.lon ?? null} /> : null}
      {phone ? <Locator /> : null}
      <h1 className={s.srOnly}>Explore: {LAYER_LABEL[product.id]}</h1>
      <ExploreLegend product={product} frame={frame} currents={streaks ? current : null} />
      {tier === "static" ? (
        <Unavailable what="The layers can't be drawn on this device" why="The chart is showing as a still picture (no WebGL, data saver, offline or a saved copy)." next="Station records are in Research, as tables and series." />
      ) : null}
      <div className={s.switch} role="radiogroup" aria-label="Layer" data-testid="nx-layers">
        {model.products.map((p) => (
          <button key={p.id} type="button" role="radio" aria-checked={p.id === product.id} className={s.switchItem} onClick={() => chooseProduct(p.id)} data-testid={`nx-layer-${p.id}`}>
            <span>{LAYER_LABEL[p.id]}</span>
            <small>{p.kind === "measured" ? "measured" : p.kind === "analysis" ? "analysis" : p.pastRun ? "past run" : p.kind}</small>
          </button>
        ))}
        {model.currents ? (
          <button type="button" role="switch" aria-checked={currentsOn} className={s.switchItem} data-on={currentsOn ? "1" : "0"} onClick={() => setCurrentsOn((v) => !v)} data-testid="nx-currents-toggle">
            <span>Currents</span>
            <small>{!currentsOn ? "off" : overModel ? "not over a model" : "on, measured"}</small>
          </button>
        ) : null}
      </div>
      <TimeRail
        model={model}
        product={product}
        frameId={frame.id}
        onFrame={chooseFrame}
        currentsOn={streaks}
        currentId={hourId}
        onCurrent={chooseHour}
        playing={playing}
        canPlay={canPlay}
        onPlay={() => setPlaying((p) => !p)}
      />
      <div ref={readoutRef} className={s.readoutSlot} data-testid="nx-readout">
        {readout ?? <p className={s.prompt}>Tap the water for its value, its date and a transect along its latitude.</p>}
      </div>
    </section>
  );
}

export function ExploreUnavailable({ why }: { why: string }) {
  return (
    <section className={s.explore} data-testid="nx-explore">
      <p className={s.kicker}>Explore</p>
      <Unavailable what="The water can't be shown right now" why={`The published data could not be read (${why}).`} next="Try again in a few minutes; the data is republished several times a day." />
    </section>
  );
}

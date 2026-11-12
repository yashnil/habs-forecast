"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Map as MapLibreMap } from "maplibre-gl";
import { capability, noteContextLoss, type GraphicsTier } from "@/lib/capability";
import { EASE_CHART, pan } from "@/nx/ds/motion";
import { alignIn, coverCamera, fillZoom, fitAtZoom, fitBounds, keepInFrame, NO_INSETS, scaleBar, screenRect, slideWithin, unproject, type Bounds, type Camera, type Insets, type Viewport, latY, lngX, worldSize, xLng, yLat } from "./mercator";
import { placeAll, type Anchor, type Rect } from "./placement";
import { allMarks, onMarksChange, publishFrame, staticProjector, type ChartFrame } from "./runtime";
import { getScene, subscribeScene, type ChartScene } from "./scene";
import { setChartMap } from "./control";
import { measureCovers, onCoversChange } from "./cover";
import { chartStyle, CHART_FRAME, modePaint, posterImages, posterLandEast, readPalette } from "./style";
import { Neatline } from "./Neatline";
import { ChartLabels, CanyonName, type ChartPort } from "./Labels";
import { RegulationLayer } from "./RegulationLayer";

type Win = Window & { __cwMap?: MapLibreMap; __nxChart?: { inits: number; map: MapLibreMap | null; tier: GraphicsTier; camera: Camera | null; scene: string | null } };

const bezier = (t: number) => {
  // cubic-bezier(EASE_CHART) solved for y at x = t
  const [x1, y1, x2, y2] = EASE_CHART;
  const bx = (u: number) => 3 * x1 * u * (1 - u) ** 2 + 3 * x2 * u * u * (1 - u) + u ** 3;
  const by = (u: number) => 3 * y1 * u * (1 - u) ** 2 + 3 * y2 * u * u * (1 - u) + u ** 3;
  let lo = 0, hi = 1;
  for (let i = 0; i < 22; i++) {
    const m = (lo + hi) / 2;
    if (bx(m) < t) lo = m;
    else hi = m;
  }
  return by((lo + hi) / 2);
};

/** The land polygon's whole box, a little inside it so its outline never shows: the map's maxBounds. */
const FRAME_BOUNDS: Bounds = (() => {
  const [w, s, e, n] = CHART_FRAME.bounds.map((v, i) => v + (i < 2 ? 0.05 : -0.05));
  return [
    [w, s],
    [e, n],
  ];
})();

/**
 * The chart's extent with room for what covers the sheet: the map's bounds hold the whole canvas, but the part under
 * the phone sheet or a panel is never seen, so the frame reaches past the drawn extent by the cover's depth (at the
 * least zoom the view allows, so the visible part never runs off the drawing). Without it a phone could never bring
 * southern California above its sheet.
 */
function frameFor(cover: Insets, minZoom: number): Bounds {
  const ws = worldSize(minZoom);
  const [[w, s], [e, n]] = FRAME_BOUNDS;
  return [
    [xLng(lngX(w) - cover.left / ws), yLat(latY(s) + cover.bottom / ws)],
    [xLng(lngX(e) + cover.right / ws), yLat(latY(n) - cover.top / ws)],
  ];
}
const minZoomOf = (scene: ChartScene) => scene.zoomRange?.[0] ?? CHART_ZOOM[0];

/** The free area inside the sheet: room for the neatline's breath, plus whatever covers the sheet (the phone sheet, a panel). */
function sceneInsets(vp: Viewport, cover: Insets): Insets {
  const pad = vp.width < 560 ? 20 : 44;
  return { top: cover.top + pad, right: cover.right + pad, bottom: cover.bottom + pad, left: cover.left + pad };
}

export function cameraFor(scene: ChartScene, vp: Viewport, cover: Insets): Camera {
  if ("cover" in scene.view) {
    // the box covers the free part of the sheet (the phone's resting margin excluded), as the still's CSS does
    const free = { width: vp.width, height: Math.max(1, vp.height - cover.bottom) };
    const c = coverCamera(scene.view.cover, free, scene.view.position);
    return { center: unproject(c, free, free.width / 2, free.height / 2 + cover.bottom / 2), zoom: c.zoom };
  }
  const ins = sceneInsets(vp, cover);
  let cam = fitBounds(scene.view.bounds, vp, ins, scene.view.maxZoom ?? 12.5);
  if (scene.view.fill) cam = fitAtZoom(scene.view.bounds, vp, ins, Math.min(scene.view.maxZoom ?? 12.5, fillZoom(scene.view.bounds, vp, ins)));
  // a box too large for this sheet (the whole coast on a phone) is shown from the zoom the chart's extent allows,
  // centred on the box (then aligned), rather than left for the map's clamp to crop at random
  const frame = frameFor(cover, minZoomOf(scene));
  const inFrame = keepInFrame(cam, vp, frame);
  if (inFrame.zoom > cam.zoom) cam = fitAtZoom(scene.view.bounds, vp, ins, inFrame.zoom);
  if (scene.view.align) cam = alignIn(cam, scene.view.bounds, vp, ins, scene.view.align);
  if (scene.view.within) cam = slideWithin(cam, vp, scene.view.bounds, scene.view.within, cover);
  return keepInFrame(cam, vp, frame);
}

/** Paper between the chart's own neatline and a panel set beside it (Research's record), so the two read as two sheets. */
const PANE_GAP = 24;

/**
 * How much of the sheet is covered, edge by edge, and what covers it elsewhere (src/nx/chart/cover.ts). A panel on
 * a side also takes a strip of paper beside it: the chart's pane, its neatline and everything written on it stop
 * there, so no graduation runs on over the panel (README §2.1: never a map behind a panel).
 */
const covered = (frame: HTMLElement) => {
  const c = measureCovers(frame.getBoundingClientRect());
  if (c.insets.right) c.insets.right += PANE_GAP;
  if (c.insets.left) c.insets.left += PANE_GAP;
  return c;
};
const sameInsets = (a: Insets, b: Insets) => a.top === b.top && a.right === b.right && a.bottom === b.bottom && a.left === b.left;

/** The chart's own zoom limits; a scene may narrow them (ChartScene.zoomRange). */
const CHART_ZOOM = [4, 13.5] as const;

const chartModeOf = () => document.documentElement.getAttribute("data-chart-mode") || "day";

/**
 * The persistent chart (README §9 Sheet): one MapLibre instance for every page of the route group, mounted
 * once by its layout and never rebuilt. Pages declare scenes; the host pans to each with the `pan` preset
 * (a jump under reduced motion). Everything written on the chart is placed here in one collision pass. The
 * static tier (no WebGL2, data saver, offline, a saved copy, repeated context loss) never loads MapLibre: it
 * shows the baked chart image of the same view under the same names and notes.
 */
export function ChartHost({ ports, opening }: { ports: ChartPort[]; opening?: React.ReactNode }) {
  const frameRef = useRef<HTMLDivElement>(null);
  const mapEl = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const [tier, setTier] = useState<GraphicsTier | null>(null);
  const [reduced, setReduced] = useState(false);
  const [vp, setVp] = useState<Viewport | null>(null);
  const [cam, setCam] = useState<Camera | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [lost, setLost] = useState(false);
  const [mode, setMode] = useState("day");
  const [neat, setNeat] = useState({ band: 4, label: 20 });
  const coverRef = useRef<Insets>(NO_INSETS);
  const coverObstacles = useRef<Rect[]>([]);
  const sceneId = useRef<string | null>(null);
  const prevAnchors = useRef(new Map<string, Anchor>());
  const raf = useRef(0);
  /** when the pan now travelling lands (performance.now() ms); a refinement of the same view glides on until then */
  const panUntil = useRef(0);
  const frameRefLatest = useRef<ChartFrame | null>(null);

  // ---------------------------------------------------------------- placement of everything on the chart
  const placeNow = useCallback(() => {
    raf.current = 0;
    const f = frameRefLatest.current;
    const frameEl = frameRef.current;
    if (!f || !frameEl) return;
    const { width: W, height: H } = f.viewport;
    const ins = f.insets;
    // nothing is written under a panel, the resting phone sheet or the chart's own scale and sources
    const area: Rect = { x0: 6 + ins.left, y0: 6 + ins.top, x1: W - 6 - ins.right, y1: H - 6 - ins.bottom };
    const obstacles: Rect[] = [...coverObstacles.current];
    const scale = frameEl.querySelector<HTMLElement>(".n12-scale");
    if (scale) obstacles.push({ x0: ins.left, y0: H - ins.bottom - scale.offsetHeight - 16, x1: ins.left + scale.offsetWidth + 24, y1: H });
    const attrib = frameEl.querySelector<HTMLElement>(".n12-attrib");
    if (attrib && attrib.offsetWidth) obstacles.push({ x0: W - ins.right - attrib.offsetWidth - 20, y0: H - ins.bottom - attrib.offsetHeight - 16, x1: W - ins.right, y1: H - ins.bottom });
    const canyon = frameEl.querySelector<SVGGraphicsElement>(".n12-canyon-name text");
    if (canyon && canyon.getAttribute("visibility") !== "hidden") {
      const b = canyon.getBBox();
      if (b.width) obstacles.push({ x0: b.x, y0: b.y, x1: b.x + b.width, y1: b.y + b.height });
    }
    // a regulation line's or band's name holds its place (RegulationLayer.tsx draws a line only with its name)
    frameEl.querySelectorAll<SVGGraphicsElement>(".n12-reg text[data-reg-hold]").forEach((t) => {
      if (t.getAttribute("visibility") === "hidden") return;
      const b = t.getBBox();
      if (b.width) obstacles.push({ x0: b.x - 2, y0: b.y - 2, x1: b.x + b.width + 2, y1: b.y + b.height + 2 });
    });
    const z = f.camera.zoom;
    const entries = allMarks().filter((e) => (e.spec.minZoom ?? -1) <= z && z <= (e.spec.maxZoom ?? 99));
    const items = entries.map((e) => {
      const p = f.project(e.spec.lon, e.spec.lat);
      return { ...e.spec, x: p.x, y: p.y, width: e.w, height: e.h };
    });
    const placed = placeAll(items, area, obstacles, prevAnchors.current);
    // the rest of regulation's lettering (zone names, the corner note, a notice's latitudes) yields to readings,
    // stations and ports (priority 20 and up), which matter more where they stand, and holds its place against the
    // names of water below them (a label that yielded keeps `data-reg-yield` so the layer's per-frame drawing does
    // not flash it back)
    const reg = [...frameEl.querySelectorAll<SVGGraphicsElement>(".n12-reg text:not([data-reg-hold])")].filter((t) => t.getAttribute("visibility") !== "hidden" || t.hasAttribute("data-reg-yield")).map((t) => ({ t, b: t.getBBox() }));
    if (reg.length) {
      const hits = (i: number, b: DOMRect) => b.x < placed[i].left + items[i].width && b.x + b.width > placed[i].left && b.y < placed[i].top + items[i].height && b.y + b.height > placed[i].top;
      const live = reg.filter(({ t, b }) => {
        const yields = placed.some((pl, i) => pl.visible && items[i].priority >= 20 && hits(i, b));
        if (yields) t.setAttribute("data-reg-yield", "");
        else t.removeAttribute("data-reg-yield");
        t.setAttribute("visibility", yields ? "hidden" : "visible");
        return !yields;
      });
      placed.forEach((pl, i) => {
        if (pl.visible && items[i].priority < 20 && live.some(({ b }) => hits(i, b))) pl.visible = false;
      });
    }
    const shown = new Set<string>();
    const leaders: string[] = [];
    placed.forEach((pl, i) => {
      const e = entries[i];
      const it = items[i];
      if (pl.visible) {
        shown.add(pl.id);
        prevAnchors.current.set(pl.id, pl.anchor);
        e.el.style.transform = `translate(${Math.round(pl.left)}px, ${Math.round(pl.top)}px)`;
        e.el.style.visibility = "visible";
        e.el.dataset.anchor = pl.anchor;
        e.el.style.setProperty("--nx-px", `${Math.round(it.x - pl.left)}px`);
        e.el.style.setProperty("--nx-py", `${Math.round(it.y - pl.top)}px`);
        if (e.spec.leader) {
          // from the point to the nearest corner of the box's near edge
          const bx = Math.min(Math.max(it.x, pl.left), pl.left + it.width);
          const by = it.y < pl.top ? pl.top : it.y > pl.top + it.height ? pl.top + it.height : pl.top + it.height / 2;
          const ex = it.x < pl.left ? pl.left : it.x > pl.left + it.width ? pl.left + it.width : bx;
          leaders.push(`<line x1="${it.x.toFixed(1)}" y1="${it.y.toFixed(1)}" x2="${ex.toFixed(1)}" y2="${by.toFixed(1)}"/><circle cx="${it.x.toFixed(1)}" cy="${it.y.toFixed(1)}" r="4"/>`);
        }
      } else e.el.style.visibility = "hidden";
      const pt = e.el.previousElementSibling as HTMLElement | null;
      if (pt?.dataset.nxPoint === e.spec.id) {
        pt.style.transform = `translate(${Math.round(it.x)}px, ${Math.round(it.y)}px)`;
        pt.style.visibility = pl.visible ? "visible" : "hidden";
      }
    });
    const svg = frameEl.querySelector(".n12-leaders");
    if (svg) svg.innerHTML = leaders.join("");
    for (const e of allMarks()) {
      if (shown.has(e.spec.id)) continue;
      e.el.style.visibility = "hidden";
      const pt = e.el.previousElementSibling as HTMLElement | null;
      if (pt?.dataset.nxPoint === e.spec.id) pt.style.visibility = "hidden";
    }
  }, []);

  const schedule = useCallback(() => {
    if (!raf.current) raf.current = requestAnimationFrame(placeNow);
  }, [placeNow]);

  const publish = useCallback(
    (camera: Camera, viewport: Viewport, live: boolean) => {
      const m = mapRef.current;
      const project = live && m ? (lon: number, lat: number) => m.project([lon, lat]) : staticProjector(camera, viewport);
      const f: ChartFrame = { camera, viewport, insets: coverRef.current, project, live };
      const el = frameRef.current;
      if (el) {
        for (const k of ["top", "right", "bottom", "left"] as const) el.style.setProperty(`--nx-cover-${k}`, `${coverRef.current[k]}px`);
        el.dataset.free = viewport.width - coverRef.current.left - coverRef.current.right < 600 ? "narrow" : "wide";
      }
      frameRefLatest.current = f;
      publishFrame(f);
      setCam(camera);
      schedule();
      const w = window as Win;
      if (w.__nxChart) w.__nxChart.camera = camera;
    },
    [schedule],
  );

  // ---------------------------------------------------------------- capability, mode, size
  useEffect(() => {
    const c = capability();
    setTier(c.tier);
    setReduced(c.reducedMotion);
    setMode(chartModeOf());
    const w = window as Win;
    w.__nxChart ??= { inits: 0, map: null, tier: c.tier, camera: null, scene: null };
    w.__nxChart.tier = c.tier;
    const mo = new MutationObserver(() => setMode(chartModeOf()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-chart-mode"] });
    const mq = matchMedia("(prefers-reduced-motion: reduce)");
    const onMq = () => setReduced(mq.matches);
    mq.addEventListener("change", onMq);
    return () => {
      mo.disconnect();
      mq.removeEventListener("change", onMq);
    };
  }, []);

  useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const c = covered(el);
      coverRef.current = c.insets;
      coverObstacles.current = c.obstacles;
      const cs = getComputedStyle(el);
      setNeat((o) => {
        const band = parseFloat(cs.getPropertyValue("--neat-band")) || 4;
        const label = parseFloat(cs.getPropertyValue("--neat-label")) || 20;
        return o.band === band && o.label === label ? o : { band, label };
      });
      setVp((v) => (v && Math.abs(v.width - r.width) < 0.5 && Math.abs(v.height - r.height) < 0.5 ? v : { width: Math.round(r.width), height: Math.round(r.height) }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ---------------------------------------------------------------- the map (once)
  useEffect(() => {
    if (tier === null || tier === "static" || !mapEl.current || mapRef.current) return;
    let cancelled = false;
    let destroy = () => {};
    (async () => {
      const ml = await import("maplibre-gl");
      if (cancelled || !mapEl.current) return;
      // MapLibre 6's worker is an ES module the bundler does not emit (scripts/copy-maplibre-worker.mjs)
      ml.setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");
      const palette = readPalette();
      const op = document.documentElement.getAttribute("data-nx-opening");
      const hidden = op === "on" || op === "run";
      const scene = getScene();
      const box = mapEl.current.getBoundingClientRect();
      const viewport = { width: box.width, height: box.height };
      const startCover = covered(frameRef.current!).insets;
      const start = scene ? cameraFor(scene, viewport, startCover) : { center: [-120.5, 37] as [number, number], zoom: 5 };
      const map = new ml.Map({
        container: mapEl.current,
        style: chartStyle(palette, location.origin, { hidden }),
        center: start.center,
        zoom: start.zoom,
        minZoom: CHART_ZOOM[0],
        maxZoom: CHART_ZOOM[1],
        maxBounds: scene ? frameFor(startCover, Math.min(minZoomOf(scene), start.zoom)) : FRAME_BOUNDS,
        renderWorldCopies: false,
        attributionControl: false,
        dragRotate: false,
        pitchWithRotate: false,
        touchPitch: false,
        trackResize: false,
        fadeDuration: 0,
        pixelRatio: Math.min(window.devicePixelRatio || 1, tier === "full" ? 2 : 1.5),
        canvasContextAttributes: { antialias: true },
      });
      map.touchZoomRotate.disableRotation();
      map.keyboard.disableRotation();
      const win = window as Win;
      win.__nxChart!.inits += 1;
      win.__nxChart!.map = map;
      win.__cwMap = map;
      mapRef.current = map;
      setChartMap(map, false);
      const onMove = () => {
        const c = map.getCenter();
        const cvs = map.getCanvas();
        publish({ center: [c.lng, c.lat], zoom: map.getZoom() }, { width: cvs.clientWidth, height: cvs.clientHeight }, true);
      };
      map.on("move", onMove);
      map.once("idle", () => {
        if (cancelled) return;
        setMapReady(true);
        setChartMap(map, true);
        onMove();
      });
      const canvas = map.getCanvas();
      canvas.setAttribute("aria-label", "Chart of the California coast");
      const onLost = (ev: Event) => {
        ev.preventDefault();
        noteContextLoss();
        setLost(true);
        if (capability().tier === "static") setTier("static");
      };
      const onRestored = () => {
        setLost(false);
        const sc = getScene();
        const cvs = map.getCanvas();
        if (sc) {
          const c = cameraFor(sc, { width: cvs.clientWidth, height: cvs.clientHeight }, coverRef.current);
          map.jumpTo({ center: c.center, zoom: c.zoom });
        }
      };
      canvas.addEventListener("webglcontextlost", onLost);
      canvas.addEventListener("webglcontextrestored", onRestored);
      destroy = () => {
        canvas.removeEventListener("webglcontextlost", onLost);
        canvas.removeEventListener("webglcontextrestored", onRestored);
        map.remove();
        mapRef.current = null;
        setChartMap(null);
      };
    })();
    return () => {
      cancelled = true;
      destroy();
    };
  }, [tier, publish]);

  // static tier: drop a map that was running (a second context loss)
  useEffect(() => {
    if (tier === "static" && mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
      setChartMap(null);
      setMapReady(false);
    }
  }, [tier]);

  // ---------------------------------------------------------------- mode
  useEffect(() => {
    const m = mapRef.current;
    if (!m || !mapReady) return;
    const p = readPalette();
    for (const [layer, prop, value] of modePaint(p)) if (m.getLayer(layer)) m.setPaintProperty(layer, prop as Parameters<MapLibreMap["setPaintProperty"]>[1], value as Parameters<MapLibreMap["setPaintProperty"]>[2]);
  }, [mode, mapReady]);

  // ---------------------------------------------------------------- scenes
  const applyScene = useCallback(
    (animate: boolean) => {
      const scene = getScene();
      const el = frameRef.current;
      if (!scene || !el || !vp) return;
      const c = covered(el);
      coverRef.current = c.insets;
      coverObstacles.current = c.obstacles;
      const target = cameraFor(scene, vp, coverRef.current);
      const m = mapRef.current;
      const w = window as Win;
      if (w.__nxChart) w.__nxChart.scene = scene.id;
      document.documentElement.dataset.nxScene = scene.id;
      if (m) {
        // the view's own zoom limits (Explore keeps the coast in sight), else the chart's
        const [zLo, zHi] = scene.zoomRange ?? [CHART_ZOOM[0], CHART_ZOOM[1]];
        const limit = (also: number[]) => {
          m.setMinZoom(Math.min(zLo, target.zoom, ...also));
          m.setMaxZoom(Math.max(zHi, target.zoom, ...also));
        };
        m.setMaxBounds(frameFor(coverRef.current, Math.min(zLo, target.zoom)));
        // loosened to the present zoom while the camera travels (a clamp would jump it), tightened when it lands
        limit([m.getZoom()]);
        m.once("moveend", () => {
          if (getScene()?.id === scene.id) limit([]);
        });
        const interactive = scene.interactive !== false;
        for (const h of [m.dragPan, m.scrollZoom, m.boxZoom, m.doubleClickZoom, m.touchZoomRotate, m.keyboard]) {
          if (interactive) h.enable();
          else h.disable();
        }
        const opening = document.documentElement.getAttribute("data-nx-opening");
        // the same view refined while its pan is still travelling (Today's soundings arrive a moment after the
        // page): the pan carries on to the refined camera instead of jumping there
        const left = panUntil.current - performance.now();
        const glide = !animate && left > 60;
        const still = (!animate && !glide) || reduced || opening === "on" || opening === "run";
        if (still) m.jumpTo({ center: target.center, zoom: target.zoom });
        else {
          const duration = glide ? left : tier === "reduced" ? 300 : (pan.duration as number) * 1000;
          if (!glide) panUntil.current = performance.now() + duration;
          m.easeTo({ center: target.center, zoom: target.zoom, duration, easing: glide ? (t: number) => 1 - (1 - t) ** 2 : bezier, essential: false });
        }
      } else publish(target, vp, false);
    },
    [vp, reduced, tier, publish],
  );

  useEffect(() => {
    const onScene = () => {
      const s = getScene();
      const fresh = s?.id !== sceneId.current;
      sceneId.current = s?.id ?? null;
      applyScene(fresh);
    };
    onScene();
    return subscribeScene(onScene);
  }, [applyScene]);

  // a new size: resize the canvas (only on a real change; the phone's address bar does not reach here) and re-fit
  useEffect(() => {
    const m = mapRef.current;
    if (m && vp) m.resize();
    applyScene(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vp?.width, vp?.height]);

  useEffect(() => onMarksChange(schedule), [schedule]);

  // a panel or the phone sheet arriving, leaving or changing size: the camera refits into the free part of the
  // sheet (gliding, since it is the same view) and the names are placed again
  useEffect(
    () =>
      onCoversChange(() => {
        const el = frameRef.current;
        if (!el) return;
        const c = covered(el);
        coverObstacles.current = c.obstacles;
        if (sameInsets(c.insets, coverRef.current)) return schedule();
        applyScene(true);
      }),
    [applyScene, schedule],
  );

  const showPoster = tier === "static" || !mapReady || lost;
  // the bar is sized to the free part of the sheet, so beside a panel it is never cut by it
  const freeW = vp ? vp.width - coverRef.current.left - coverRef.current.right : 0;
  const scale = cam && vp ? scaleBar(cam.center[1], cam.zoom, Math.min(240, freeW * 0.3)) : null;

  return (
    <div className="n12-sheet" data-testid="nx-sheet" data-tier={tier ?? "pending"} data-map-ready={mapReady ? "1" : "0"}>
      <div className="n12-frame" ref={frameRef}>
        {showPoster && cam && vp ? (
          <div className="n12-poster" data-testid="nx-poster" aria-hidden="true">
            {(() => {
              const r = screenRect(cam, vp, posterLandEast());
              return <span className="n12-poster-land" style={{ transform: `translate(${r.left}px, ${r.top}px)`, width: r.width, height: r.height }} />;
            })()}
            {posterImages(mode).map((img) => {
              const r = screenRect(cam, vp, img.bounds);
              // eslint-disable-next-line @next/next/no-img-element
              return <img key={img.url} src={img.url} alt="" style={{ transform: `translate(${r.left}px, ${r.top}px)`, width: r.width, height: r.height }} decoding="async" />;
            })}
          </div>
        ) : null}
        <div className="n12-map" ref={mapEl} data-testid="nx-map" />
        {/* the static tier's data images (a page's baked frame) lie here, on the baked chart and under its names */}
        <div id="nx-chart-underlay" className="n12-underlay" aria-hidden="true" />
        {opening}
        <div className="n12-overlay" data-testid="nx-overlay">
          <RegulationLayer />
          <CanyonName />
          <svg className="n12-leaders" aria-hidden="true" />
          <ChartLabels ports={ports} />
          <div id="nx-chart-marks" className="n12-marks" />
        </div>
        {scale ? (
          <div className="n12-scale" aria-label={`Scale bar ${scale.label}. Not for navigation.`}>
            <span className="n12-scale-bar" style={{ width: scale.px }}>
              <i />
              <i />
              <i />
              <i />
            </span>
            <span className="n12-scale-text">{scale.label}. Not for navigation.</span>
          </div>
        ) : null}
        <p className="n12-attrib">
          Depth NOAA ETOPO, CRM · Coast © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>
        </p>
        <Neatline camera={cam} viewport={vp} band={neat.band} label={neat.label} pane={{ left: coverRef.current.left, right: coverRef.current.right }} />
      </div>
    </div>
  );
}

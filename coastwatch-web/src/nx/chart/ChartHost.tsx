"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Map as MapLibreMap } from "maplibre-gl";
import { capability, noteContextLoss, type GraphicsTier } from "@/lib/capability";
import { EASE_FLIGHT, flightSeconds, layerIn, layerOut, pan, zoomStep } from "@/nx/ds/motion";
import { paneJustChanged } from "@/nx/ds/layout/pane";
import { alignIn, coverCamera, fillZoom, fitAtZoom, fitBounds, keepInFrame, NO_INSETS, scaleBar, screenRect, slideWithin, unproject, type Bounds, type Camera, type Insets, type Viewport, latY, lngX, worldSize, xLng, yLat } from "./mercator";
import { placeAll, type Anchor, type Rect } from "./placement";
import { allMarks, getFrame, onMarksChange, publishFrame, staticProjector, type ChartFrame } from "./runtime";
import { getScene, subscribeScene, useScene, type ChartScene } from "./scene";
import { keepChartPointInView, onChartCommand, refitChart, setChartMap, zoomChart, fitChartBounds, type ChartCommand } from "./control";
import { drawnObstacles, measureCovers, onCoversChange } from "./cover";
import { Icon } from "@/nx/ds/Icon";
import { chartStyle, CHART_FRAME, DEPTH_LAYER_IDS, modePaint, posterImages, posterLandEast, readPalette } from "./style";
import { GraticuleLabels } from "./Neatline";
import { ChartLabels, CanyonName, type ChartPort } from "./Labels";
import { RegulationLayer, regLineBox, regTextBox } from "./RegulationLayer";
import { DepthLabels } from "./DepthLabels";
import { DeepWater } from "./DeepWater";
import { useChartHighlight } from "./highlight";
import { anchorShift, getCameraAnchor, onCameraAnchor, type CameraAnchor } from "./anchor";
import { flightDelay, noteFlight } from "./flight";
import { getMapLayers, LAYERS_DRAWN_EVENT, subscribeMapLayers, useMapLayer } from "./mapLayers";
import { useRegulation } from "./regulation";

/** The place lit from outside the chart (highlight.ts): a ring at its point, redrawn with the camera. */
function Highlight({ cam }: { cam: Camera | null }) {
  const h = useChartHighlight();
  const f = getFrame();
  if (!h || !f || !cam) return null;
  const p = f.project(h.lon, h.lat);
  return (
    <span className="n12-highlight" style={{ transform: `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)` }} data-testid="nx-highlight" aria-hidden="true">
      {h.label ? <span className="n12-port-name">{h.label}</span> : null}
    </span>
  );
}

/**
 * The chart's own buttons: zoom in, zoom out, back to the page's view, and a page's "fit the data" when its
 * scene names one. Only where the reader may move the chart; each is a 44 px target with its name.
 */
function ChartControls() {
  const scene = useScene();
  if (!scene || scene.interactive === false || scene.controls === false || "cover" in scene.view) return null;
  const fit = scene.controls?.fit;
  return (
    <div className="n12-controls" role="group" aria-label="Chart view" data-testid="nx-chart-controls">
      <button type="button" className="n12-controls-zoom" onClick={() => zoomChart(1)} aria-label="Zoom in" title="Zoom in" data-testid="nx-zoom-in">
        <Icon name="plus" size={18} />
      </button>
      <button type="button" className="n12-controls-zoom" onClick={() => zoomChart(-1)} aria-label="Zoom out" title="Zoom out" data-testid="nx-zoom-out">
        <Icon name="minus" size={18} />
      </button>
      <button type="button" onClick={refitChart} aria-label={scene.focus ? `Back to ${scene.focus.name}` : "Back to this page's view"} title={scene.focus ? `Back to ${scene.focus.name}` : "Back to this page's view"} data-testid="nx-refit">
        <Icon name="recentre" size={18} />
      </button>
      {fit ? (
        <button type="button" className="n12-controls-word" onClick={() => fitChartBounds(fit.bounds)} aria-label={fit.label} title={fit.label} data-testid="nx-fit-data">
          <span className="n12-controls-words">{fit.label}</span>
          <Icon name="layers" size={20} className="n12-controls-icon" />
        </button>
      ) : null}
    </div>
  );
}

type Win = Window & { __cwMap?: MapLibreMap; __nxChart?: { inits: number; map: MapLibreMap | null; tier: GraphicsTier; camera: Camera | null; scene: string | null; keep?: typeof keepChartPointInView; anchor?: CameraAnchor | null; target?: Camera } };

const bezier = (t: number) => {
  // cubic-bezier(EASE_FLIGHT) solved for y at x = t: the camera's curve between places (slow out, long glide in)
  const [x1, y1, x2, y2] = EASE_FLIGHT;
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

export function cameraFor(scene: ChartScene, vp: Viewport, cover: Insets, anchor: CameraAnchor | null = null): Camera {
  if ("cover" in scene.view) {
    // the box covers the free part of the sheet (the phone's resting margin excluded), as the still's CSS does
    const free = { width: vp.width, height: Math.max(1, vp.height - cover.bottom) };
    const c = coverCamera(scene.view.cover, free, scene.view.position);
    return { center: unproject(c, free, free.width / 2, free.height / 2 + cover.bottom / 2), zoom: c.zoom };
  }
  const ins = sceneInsets(vp, cover);
  // a tall, narrow free area (a desk page's chart pane) takes the view's own box for that shape when it has one
  const freeW = vp.width - ins.left - ins.right, freeH = vp.height - ins.top - ins.bottom;
  const box = scene.view.tall && freeH >= freeW * 1.25 ? scene.view.tall : scene.view.bounds;
  let cam = fitBounds(box, vp, ins, scene.view.maxZoom ?? 12.5);
  if (scene.view.fill) cam = fitAtZoom(box, vp, ins, Math.min(scene.view.maxZoom ?? 12.5, fillZoom(box, vp, ins)));
  // a box too large for this sheet (the whole coast on a phone) is shown from the zoom the chart's extent allows,
  // centred on the box (then aligned), rather than left for the map's clamp to crop at random
  const frame = frameFor(cover, minZoomOf(scene));
  const inFrame = keepInFrame(cam, vp, frame);
  if (inFrame.zoom > cam.zoom) cam = fitAtZoom(box, vp, ins, inFrame.zoom);
  if (scene.view.align) cam = alignIn(cam, box, vp, ins, scene.view.align);
  if (scene.view.within) cam = slideWithin(cam, vp, box, scene.view.within, cover);
  // a ticket stands beside its place (ds Ticket): bring the place to the ticket's edge, a stub's length away
  if (anchor) {
    const ws = worldSize(cam.zoom);
    const cx = lngX(cam.center[0]) * ws, cy = latY(cam.center[1]) * ws;
    const p = { x: vp.width / 2 + (lngX(anchor.lon) * ws - cx), y: vp.height / 2 + (latY(anchor.lat) * ws - cy) };
    const { dx, dy } = anchorShift(anchor, p, vp);
    cam = { center: [xLng((cx + dx) / ws), yLat((cy + dy) / ws)], zoom: cam.zoom };
  }
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
/**
 * How high the chart's own furniture (zoom, scale) stands: above a dock or pass that reaches the right-hand corner
 * where it lives, else a gutter from the foot (the prototype's `--foot`).
 */
function writeFoot(frame: HTMLElement) {
  const fr = frame.getBoundingClientRect();
  // the furniture's column: the zoom buttons where they stand (the bottom-right corner by default)
  const c = frame.querySelector<HTMLElement>(".n12-controls")?.getBoundingClientRect();
  const x0 = c && c.width ? c.left - 6 : fr.right - 96, x1 = c && c.width ? c.right + 6 : fr.right;
  let foot = 0;
  document.querySelectorAll<HTMLElement>("[data-nx-dock], [data-nx-pass], .n12-dock-slot").forEach((el) => {
    const r = el.getBoundingClientRect();
    // where it stands at rest: an arriving dock is still sliding up (a transform) when this runs
    const tf = getComputedStyle(el).transform;
    const m = /\((.+)\)/.exec(tf);
    const ty = m ? parseFloat(m[1].split(",")[tf.startsWith("matrix3d") ? 13 : 5]) || 0 : 0;
    const top = r.top - ty;
    if (!r.width || !r.height || r.right < x0 || r.left > x1 || top >= fr.bottom) return;
    foot = Math.max(foot, Math.round(fr.bottom - top + 16));
  });
  frame.style.setProperty("--nx-foot", foot ? `${foot}px` : "");
}
const sameInsets = (a: Insets, b: Insets) => a.top === b.top && a.right === b.right && a.bottom === b.bottom && a.left === b.left;

/** True while the opening plays and the chart's lettering waits for its beat (chart.css hides the overlay then). */
const openingHidesMarks = () => {
  const h = document.documentElement;
  const op = h.getAttribute("data-nx-opening");
  return (op === "on" || op === "run") && !h.hasAttribute("data-nx-marks");
};

/** How long after the map's "load" it counts as ready when "idle" has not come (ms). */
const READY_AFTER_LOAD_MS = 1500;

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
  const coverRef = useRef<Insets>(NO_INSETS);
  const coverObstacles = useRef<Rect[]>([]);
  const sceneId = useRef<string | null>(null);
  const prevAnchors = useRef(new Map<string, Anchor>());
  const raf = useRef(0);
  /** when the pan now travelling lands (performance.now() ms); a refinement of the same view glides on until then */
  const panUntil = useRef(0);
  const frameRefLatest = useRef<ChartFrame | null>(null);
  /** where the zoom buttons are heading while the camera travels */
  const zoomGoal = useRef<number | null>(null);
  /** a flight waiting for the leaving ticket to finish shrinking */
  const flightTimer = useRef(0);
  const sheetRef = useRef<HTMLDivElement>(null);
  const depthOn = useMapLayer("depth");
  const regulationOn = useMapLayer("regulation");
  const regulationSpec = useRegulation();

  /**
   * What the chart actually draws, for tests and QA (`data-layers` on the sheet): Depth from the map style's own layer
   * visibility, Regulation when its layer has something to draw, Sea now and Toxin tests when marks of theirs are on
   * the chart. Never copied from the chips.
   */
  const writeLayers = useCallback(() => {
    const el = sheetRef.current;
    if (!el) return;
    // not before the chart can say: on the live tier, once the map has drawn (the attribute appears then)
    if (el.dataset.tier !== "static" && el.dataset.mapReady !== "1") return;
    const m = mapRef.current;
    const on: string[] = [];
    if (m && m.getLayer("contours") && m.getLayoutProperty("contours", "visibility") !== "none" && getMapLayers().on.depth) on.push("depth");
    if (el.querySelector(".n12-reg-wrap:not([data-leaving])")) on.push("regulation");
    // a page's marks (buoys, tests, stations): a layer is drawn while its marks are on the chart (they mount with the
    // chip and leave with it); a layer that has nothing to draw at this place counts as drawn while its chip is on
    const layers = getMapLayers();
    const registered = new Set(allMarks().map((e) => e.spec.layer).filter(Boolean) as string[]);
    for (const id of Object.keys(layers.on)) {
      if (id === "depth" || id === "regulation" || on.includes(id)) continue;
      if (registered.has(id) || layers.on[id]) on.push(id);
    }
    const v = on.join(" ");
    if (el.getAttribute("data-layers") !== v) el.setAttribute("data-layers", v);
  }, []);

  // ---------------------------------------------------------------- placement of everything on the chart
  const placeNow = useCallback(() => {
    raf.current = 0;
    // while the opening plays the lettering is hidden until its beat: nothing is placed (the page's first seconds on
    // a slow machine belong to the opening); the attribute's change places everything at once
    if (openingHidesMarks()) return;
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
    // the chart's own buttons, and whatever a page drew on the chart and asked lettering to keep clear of
    // (a transect line, a read cell: cover.ts registerChartObstacle)
    const controls = frameEl.querySelector<HTMLElement>(".n12-controls");
    if (controls && controls.offsetWidth) {
      const fr = frameEl.getBoundingClientRect(), cr = controls.getBoundingClientRect();
      obstacles.push({ x0: cr.left - fr.left - 6, y0: cr.top - fr.top - 6, x1: cr.right - fr.left + 6, y1: cr.bottom - fr.top + 6 });
    }
    obstacles.push(...drawnObstacles(f.project));
    const canyon = frameEl.querySelector<SVGGraphicsElement>(".n12-canyon-name text");
    if (canyon && canyon.getAttribute("visibility") !== "hidden") {
      const b = canyon.getBBox();
      if (b.width) obstacles.push({ x0: b.x, y0: b.y, x1: b.x + b.width, y1: b.y + b.height });
    }
    // a regulation line's or band's name holds its place (RegulationLayer.tsx draws a line only with its name)
    frameEl.querySelectorAll<SVGTextElement>(".n12-reg text[data-reg-hold]").forEach((t) => {
      if (t.getAttribute("visibility") === "hidden") return;
      const b = regTextBox(t);
      if (b.width) obstacles.push({ x0: b.x - 2, y0: b.y - 2, x1: b.x + b.width + 2, y1: b.y + b.height + 2 });
    });
    // a zone's boundary line (a parallel of latitude): a name is never set across it, struck through
    frameEl.querySelectorAll<SVGPathElement>(".n12-reg path.n12-reg-line, .n12-reg path.n12-reg-limitline").forEach((ln) => {
      if (ln.getAttribute("visibility") === "hidden") return;
      const b = regLineBox(ln);
      if (b && b.width > 24 && b.height < 3) obstacles.push({ x0: b.x, y0: b.y - 7, x1: b.x + b.width, y1: b.y + b.height + 7 });
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
    const reg = [...frameEl.querySelectorAll<SVGTextElement>(".n12-reg text:not([data-reg-hold])")].filter((t) => t.getAttribute("visibility") !== "hidden" || t.hasAttribute("data-reg-yield")).map((t) => ({ t, b: regTextBox(t) }));
    if (reg.length) {
      const hits = (i: number, b: { x: number; y: number; width: number; height: number }) => b.x < placed[i].left + items[i].width && b.x + b.width > placed[i].left && b.y < placed[i].top + items[i].height && b.y + b.height > placed[i].top;
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
    const keptDots = new Set<string>();
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
        // the reader's port keeps its symbol where its name gave way (to a transect, a locator)
        const keepDot = !!e.spec.keep && it.x >= area.x0 && it.x <= area.x1 && it.y >= area.y0 && it.y <= area.y1;
        pt.style.visibility = pl.visible || keepDot ? "visible" : "hidden";
        if (keepDot) keptDots.add(e.spec.id);
      }
    });
    const svg = frameEl.querySelector(".n12-leaders");
    if (svg) svg.innerHTML = leaders.join("");
    writeLayers();
    for (const e of allMarks()) {
      if (shown.has(e.spec.id)) continue;
      e.el.style.visibility = "hidden";
      const pt = e.el.previousElementSibling as HTMLElement | null;
      if (pt?.dataset.nxPoint === e.spec.id && !keptDots.has(e.spec.id)) pt.style.visibility = "hidden";
    }
  }, [writeLayers]);

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
    // for tests: the page-facing camera request, as a page would call it
    w.__nxChart.keep = keepChartPointInView;
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
      writeFoot(el);
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
      const start = scene ? cameraFor(scene, viewport, startCover, getCameraAnchor()) : { center: [-120.5, 37] as [number, number], zoom: 5 };
      const map = new ml.Map({
        container: mapEl.current,
        style: chartStyle(palette, location.origin, { hidden, depth: getMapLayers().on.depth !== false }),
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
      // ready when the map is first idle (style, sources and tiles drawn), or, if it never settles that far (software
      // GL on a slow machine, a page that keeps the camera or a layer moving), a moment after its style and first
      // frame have loaded: the chart never waits on "idle" alone
      let ready = false;
      let fallback = 0;
      const markReady = () => {
        if (ready || cancelled) return;
        ready = true;
        window.clearTimeout(fallback);
        setMapReady(true);
        setChartMap(map, true);
        onMove();
      };
      map.once("idle", markReady);
      map.once("load", () => {
        fallback = window.setTimeout(markReady, READY_AFTER_LOAD_MS);
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
        window.clearTimeout(fallback);
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
      const target = cameraFor(scene, vp, coverRef.current, getCameraAnchor());
      const m = mapRef.current;
      const w = window as Win;
      if (w.__nxChart) {
        w.__nxChart.scene = scene.id;
        // for tests: the anchor and the camera the scene asked for
        w.__nxChart.anchor = getCameraAnchor();
        w.__nxChart.target = target;
      }
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
        // a refinement that arrives while the flight is still waiting for a leaving ticket (the new ticket's anchor, a
        // dock's cover) re-aims that flight, keeping its wait: the camera never starts before the old ticket has gone
        const waiting = !animate && left > 60 && flightDelay() > 4;
        const glide = !animate && left > 60 && !waiting;
        const still = (!animate && !glide && !waiting) || reduced || opening === "on" || opening === "run";
        window.clearTimeout(flightTimer.current);
        if (still) {
          noteFlight(0, 0);
          m.jumpTo({ center: target.center, zoom: target.zoom });
        } else if (glide) {
          m.easeTo({ center: target.center, zoom: target.zoom, duration: left, easing: (t: number) => 1 - (1 - t) ** 2, essential: false });
        } else {
          // a flight between places (v14 README §5): 0.5 to 1.15 s by distance, on the flight curve, after the old
          // ticket has shrunk back to its pin; tickets arriving wait for 62 % of it (chart/flight)
          const from = m.getCenter();
          const ws = worldSize(Math.min(m.getZoom(), target.zoom));
          const dist = Math.hypot((lngX(target.center[0]) - lngX(from.lng)) * ws, (latY(target.center[1]) - latY(from.lat)) * ws) + Math.abs(target.zoom - m.getZoom()) * 160;
          const duration = tier === "reduced" ? 300 : flightSeconds(dist) * 1000;
          const delay = flightDelay();
          noteFlight(delay, duration);
          panUntil.current = performance.now() + delay + duration;
          const go = () => (dist < 2 ? m.easeTo({ center: target.center, zoom: target.zoom, duration: Math.min(duration, 400), easing: bezier, essential: false }) : m.flyTo({ center: target.center, zoom: target.zoom, duration, curve: 1.25, easing: bezier, essential: false }));
          if (delay > 4) flightTimer.current = window.setTimeout(go, delay);
          else go();
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
    // the reader resized or reopened the chart pane: the same view glides to its new fit (`pan`); any other
    // change of size (the window, a rotation) refits at once
    applyScene(paneJustChanged());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vp?.width, vp?.height]);

  useEffect(() => onMarksChange(schedule), [schedule]);
  // the opening ending (or reaching its marks' beat): place the lettering now
  useEffect(() => {
    const mo = new MutationObserver(() => !openingHidesMarks() && schedule());
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-nx-opening", "data-nx-marks"] });
    return () => mo.disconnect();
  }, [schedule]);
  // what a page drew on the chart and asked lettering to keep clear of can change without the camera moving (a
  // transect arriving after its reading): the names are placed again when those shapes change
  useEffect(() => {
    let last = "";
    const t = window.setInterval(() => {
      const f = frameRefLatest.current;
      if (!f) return;
      const sig = drawnObstacles(f.project)
        .map((r) => `${Math.round(r.x0)},${Math.round(r.y0)},${Math.round(r.x1)},${Math.round(r.y1)}`)
        .join(";");
      if (sig !== last) {
        last = sig;
        schedule();
      }
    }, 250);
    return () => window.clearInterval(t);
  }, [schedule]);
  // what the chart draws, kept current: a chip switched, regulation arriving or leaving (RegulationLayer's event)
  useEffect(() => {
    let r = 0;
    const later = () => {
      cancelAnimationFrame(r);
      r = requestAnimationFrame(writeLayers);
    };
    const off = subscribeMapLayers(later);
    window.addEventListener(LAYERS_DRAWN_EVENT, later);
    return () => {
      off();
      window.removeEventListener(LAYERS_DRAWN_EVENT, later);
      cancelAnimationFrame(r);
    };
  }, [writeLayers]);

  // the chart's own buttons arrive with the map: the furniture's height is measured against them. What changed while
  // the map was loading (a ticket's anchor, a dock's cover) is applied now, as a refinement of the same view.
  useEffect(() => {
    const el = frameRef.current;
    if (!el || !mapReady) return;
    requestAnimationFrame(() => {
      writeFoot(el);
      writeLayers();
    });
    applyScene(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapReady]);

  // the page's ticket arrived or moved: the camera glides on to stand its place beside it
  useEffect(() => onCameraAnchor(() => applyScene(false)), [applyScene]);

  // ---------------------------------------------------------------- the Depth layer: contours fade in and out (320 / 220 ms)
  useEffect(() => {
    const m = mapRef.current;
    if (!m || !mapReady) return;
    const dur = reduced ? 0 : ((depthOn ? layerIn : layerOut) as { duration: number }).duration * 1000;
    const full: Record<string, number> = { contours: 0.42, "contours-shelf": 0.62 };
    for (const id of DEPTH_LAYER_IDS) {
      if (!m.getLayer(id)) continue;
      m.setPaintProperty(id, "line-opacity-transition", { duration: dur, delay: 0 });
      if (depthOn) m.setLayoutProperty(id, "visibility", "visible");
      m.setPaintProperty(id, "line-opacity", depthOn ? full[id] : 0);
    }
    const t = window.setTimeout(() => {
      if (!getMapLayers().on.depth) for (const id of DEPTH_LAYER_IDS) if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", "none");
      writeLayers();
    }, depthOn ? 0 : dur + 20);
    writeLayers();
    return () => window.clearTimeout(t);
  }, [depthOn, mapReady, reduced, writeLayers]);
  useEffect(() => {
    const r = requestAnimationFrame(writeLayers);
    return () => cancelAnimationFrame(r);
  }, [regulationOn, regulationSpec, writeLayers]);

  // ---------------------------------------------------------------- commands (control.ts): the chart's buttons and a page's requests
  useEffect(
    () =>
      onChartCommand((c: ChartCommand) => {
        const m = mapRef.current;
        const el = frameRef.current;
        if (!m || !el || !vp) return;
        const duration = reduced ? 0 : tier === "reduced" ? 300 : (pan.duration as number) * 1000;
        const go = (center: [number, number], zoom: number) => {
          // a view the reader or the page asked for may lie outside the scene's own zoom limits
          m.setMinZoom(Math.min(m.getMinZoom(), zoom));
          m.setMaxZoom(Math.max(m.getMaxZoom(), zoom));
          if (duration) m.easeTo({ center, zoom, duration, easing: bezier, essential: true });
          else m.jumpTo({ center, zoom });
        };
        if (c.type === "refit") {
          zoomGoal.current = null;
          return applyScene(true);
        }
        if (c.type === "zoom") {
          // steps add up while the camera is still travelling (two quick presses are two steps), inside the
          // view's own zoom limits, so a view that keeps the coast in sight still does
          const from = zoomGoal.current ?? m.getZoom();
          const z = Math.min(m.getMaxZoom(), Math.max(m.getMinZoom(), from + c.by));
          zoomGoal.current = z;
          m.once("moveend", () => {
            if (Math.abs(m.getZoom() - z) < 0.01) zoomGoal.current = null;
          });
          const ctr = m.getCenter();
          if (duration) m.easeTo({ center: [ctr.lng, ctr.lat], zoom: z, duration: Math.min(duration, (zoomStep as { duration: number }).duration * 1000), easing: bezier, essential: true });
          else m.jumpTo({ center: [ctr.lng, ctr.lat], zoom: z });
          return;
        }
        zoomGoal.current = null;
        if (c.type === "fit") {
          const cover = covered(el).insets;
          const cam = fitBounds(c.bounds, vp, sceneInsets(vp, cover), c.maxZoom ?? 12.5);
          return go(cam.center, cam.zoom);
        }
        // keep: the free area is what the covers leave now. The phone sheet is measured where it stands (its
        // present detent), not at its resting height, so a reading under a raised sheet is brought above it
        const fr = el.getBoundingClientRect();
        const cover = { ...covered(el).insets };
        const sheetEl = document.querySelector<HTMLElement>(".n12-margin[data-detent]");
        if (sheetEl) cover.bottom = Math.max(cover.bottom, Math.min(fr.height * 0.8, fr.bottom - sheetEl.getBoundingClientRect().top));
        const pad = c.margin ?? 56;
        const p = m.project([c.lon, c.lat]);
        const x0 = cover.left + pad, x1 = fr.width - cover.right - pad, y0 = cover.top + pad, y1 = fr.height - cover.bottom - pad;
        // brought a few px inside the edge, so a place already brought clear is clear when asked again
        const IN = 6;
        const dx = p.x < x0 ? p.x - x0 - IN : p.x > x1 ? p.x - x1 + IN : 0;
        const dy = p.y < y0 ? p.y - y0 - IN : p.y > y1 ? p.y - y1 + IN : 0;
        if (!dx && !dy) return;
        // the chart's extent reaches past its drawing by what covers it now (a raised sheet covers more than a resting one)
        m.setMaxBounds(frameFor(cover, m.getMinZoom()));
        const ctr = m.unproject([fr.width / 2 + dx, fr.height / 2 + dy]);
        go([ctr.lng, ctr.lat], m.getZoom());
      }),
    [applyScene, vp, reduced, tier],
  );

  // a panel or the phone sheet arriving, leaving or changing size: the camera refits into the free part of the
  // sheet (gliding, since it is the same view) and the names are placed again
  useEffect(
    () =>
      onCoversChange(() => {
        const el = frameRef.current;
        if (!el) return;
        const c = covered(el);
        coverObstacles.current = c.obstacles;
        writeFoot(el);
        if (sameInsets(c.insets, coverRef.current)) return schedule();
        applyScene(true);
      }),
    [applyScene, schedule],
  );

  const showPoster = tier === "static" || !mapReady || lost;
  // the bar is sized to the free part of the sheet, so beside a panel it is never cut by it
  const freeW = vp ? vp.width - coverRef.current.left - coverRef.current.right : 0;
  const scale = cam && vp ? scaleBar(cam.center[1], cam.zoom, Math.min(120, Math.max(60, freeW * 0.12))) : null;

  return (
    <div className="n12-sheet" ref={sheetRef} data-testid="nx-sheet" data-tier={tier ?? "pending"} data-map-ready={mapReady ? "1" : "0"}>
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
        {tier && tier !== "static" && tier !== "reduced" && mapReady && !lost ? <DeepWater ready /> : null}
        {/* the static tier's data images (a page's baked frame) lie here, on the baked chart and under its names */}
        <div id="nx-chart-underlay" className="n12-underlay" aria-hidden="true" />
        {opening}
        <div className="n12-overlay" data-testid="nx-overlay">
          <RegulationLayer />
          <DepthLabels />
          <CanyonName />
          <svg className="n12-leaders" aria-hidden="true" />
          <ChartLabels ports={ports} />
          <div id="nx-chart-marks" className="n12-marks" />
          <Highlight cam={cam} />
        </div>
        {scale ? (
          <div className="n12-scale" aria-label={`Scale bar ${scale.label}. Not for navigation.`}>
            <span className="n12-scale-bar" style={{ width: scale.px }} />
            <span className="n12-scale-text">{scale.label}</span>
            <span className="n12-scale-nfn">Not for navigation</span>
          </div>
        ) : null}
        {tier !== "static" && mapReady ? <ChartControls /> : null}
        <p className="n12-attrib">
          Depth NOAA ETOPO, CRM · Coast © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>
        </p>
        <GraticuleLabels camera={cam} viewport={vp} insets={coverRef.current} />
      </div>
    </div>
  );
}

"use client";

import { useEffect } from "react";
import { animate } from "motion/react";
import type { Map as MapLibreMap } from "maplibre-gl";
import { EASE_CHART, ink as inkPreset, OPENING, register } from "@/nx/ds/motion";
import { getChartMap, whenChartReady } from "@/nx/chart/control";
import { DEPTH_THRESHOLDS, STILL } from "@/nx/chart/style";

/**
 * The one-time opening (README §8, journeys.md Journey 1). It is the home screen from frame one: the statement
 * and Find your port are server HTML in the margin, and the sheet shows a dithered still of the lit Monterey
 * Canyon, registered exactly to the live chart beneath it (both are `cover`ed to the same box). Then, on
 * motion.ts OPENING's clock, the still settles away, the isobaths ink, the depth tints flood from the shore
 * outward, the neatline closes and the names register: the opening becomes the chart in place. At most 5 s;
 * any key, pointer or the Skip button ends it on the final frame. It never plays twice (nx-opening-seen-v1),
 * never on a deep link, under reduced motion or on the static tier (src/nx/shell/entry.ts decides pre-paint).
 */
export function OpeningStill() {
  return (
    <div className="n12-still" aria-hidden="true" data-testid="nx-still">
      <div className="n12-still-ink" style={{ maskImage: `url(${STILL.url})`, WebkitMaskImage: `url(${STILL.url})` }} />
      <svg className="n12-ink-lines" />
      <p className="n12-still-caption">Monterey Canyon, lit from the north-west. Depth exaggerated {STILL.exaggeration}×. NOAA CRM; not for navigation.</p>
    </div>
  );
}

const html = () => document.documentElement;
const state = () => html().getAttribute("data-nx-opening");
const MINOR = 0.42;
const INDEX = 0.78;

function finalPaint(map: MapLibreMap) {
  for (const d of DEPTH_THRESHOLDS) {
    const id = `depth-${d}`;
    if (!map.getLayer(id)) continue;
    map.setPaintProperty(id, "fill-opacity-transition", { duration: 0, delay: 0 });
    map.setPaintProperty(id, "fill-opacity", 1);
  }
  if (map.getLayer("isobaths-minor")) map.setPaintProperty("isobaths-minor", "line-opacity", MINOR);
  if (map.getLayer("isobaths-index")) map.setPaintProperty("isobaths-index", "line-opacity", INDEX);
  if (map.getLayer("isobath-labels")) map.setPaintProperty("isobath-labels", "text-opacity", 1);
}

type Iso = { type: "Feature"; properties: { d: number; i: number }; geometry: { type: "LineString"; coordinates: [number, number][] } };
let isoCache: Promise<Iso[]> | null = null;
const isobaths = () => (isoCache ??= fetch("/chart/isobaths.geojson").then((r) => r.json()).then((j) => j.features as Iso[]).catch(() => []));

/** The isobaths in view as SVG paths, projected through the live map (the camera holds still during the opening). */
async function drawInk(svg: SVGSVGElement, map: MapLibreMap) {
  const feats = await isobaths();
  const b = map.getBounds();
  const parts: string[] = [];
  for (const f of feats) {
    const c = f.geometry.coordinates;
    if (!c.some(([x, y]) => x >= b.getWest() && x <= b.getEast() && y >= b.getSouth() && y <= b.getNorth())) continue;
    let d = "";
    let last = "";
    for (const [x, y] of c) {
      const p = map.project([x, y]);
      const s = `${p.x.toFixed(0)},${p.y.toFixed(0)}`;
      if (s === last) continue;
      d += (d ? "L" : "M") + s;
      last = s;
    }
    parts.push(`<path pathLength="1" class="${f.properties.i ? "n12-ink-index" : "n12-ink-minor"}" d="${d}"/>`);
  }
  svg.innerHTML = parts.join("");
}

export function OpeningController() {
  useEffect(() => {
    if (state() !== "on") return;
    html().setAttribute("data-nx-opening", "run");
    const stops: Array<{ stop: () => void }> = [];
    const timers: number[] = [];
    let done = false;
    const paint = performance.getEntriesByName("first-contentful-paint")[0]?.startTime ?? 0;
    const elapsed = () => (performance.now() - paint) / 1000;

    const finish = () => {
      if (done) return;
      done = true;
      stops.forEach((a) => a.stop());
      timers.forEach((t) => clearTimeout(t));
      const map = getChartMap();
      if (map) finalPaint(map);
      else whenChartReady().then(finalPaint);
      const r = html();
      r.removeAttribute("data-nx-opening");
      r.removeAttribute("data-nx-neat");
      r.removeAttribute("data-nx-marks");
      document.querySelector<SVGElement>(".n12-neat")?.style.removeProperty("clip-path");
      const overlay = document.querySelector<HTMLElement>(".n12-overlay");
      if (overlay) overlay.style.opacity = "";
    };
    // any input (the inline script marks it), or the Skip button
    const mo = new MutationObserver(() => {
      if (state() === "skip") finish();
    });
    mo.observe(html(), { attributes: true, attributeFilter: ["data-nx-opening"] });

    const still = document.querySelector<HTMLElement>(".n12-still");
    const stillInk = document.querySelector<HTMLElement>(".n12-still-ink");
    const caption = document.querySelector<HTMLElement>(".n12-still-caption");
    const svg = document.querySelector<SVGSVGElement>(".n12-ink-lines");
    const at = (sec: number, f: () => void) => timers.push(window.setTimeout(() => !done && f(), Math.max(0, (sec - elapsed()) * 1000)));

    // the hard cap: whatever is late, the opening is over by OPENING.total (+0.2 s grace)
    at(OPENING.total + 0.2, finish);

    const run = (map: MapLibreMap | null) => {
      if (done) return;
      // the chart may be ready after the canyon's hold: the rest of the timeline is compressed into what is left
      const start = Math.max(OPENING.settleToChart[0], elapsed());
      const k = Math.max(0.35, (OPENING.total - start) / (OPENING.total - OPENING.settleToChart[0]));
      const T = (t: number) => start + (t - OPENING.settleToChart[0]) * k;
      const D = (r: readonly [number, number]) => (r[1] - r[0]) * k;

      at(T(OPENING.settleToChart[0]), () => {
        if (still) stops.push(animate(still, { opacity: [1, 0] }, { duration: D(OPENING.settleToChart), ease: EASE_CHART }));
        if (stillInk) stops.push(animate(stillInk, { scale: [1.025, 1] }, { duration: D(OPENING.settleToChart), ease: EASE_CHART }));
        if (caption) stops.push(animate(caption, { opacity: [1, 0] }, { duration: 0.4 }));
      });
      if (map && svg) {
        at(T(OPENING.isobathsInk[0]) - 0.15, () => void drawInk(svg, map));
        at(T(OPENING.isobathsInk[0]), () => {
          svg.style.opacity = "1";
          const paths = [...svg.querySelectorAll("path")];
          const dur = Math.min(D(OPENING.isobathsInk), inkPreset(D(OPENING.isobathsInk)).duration as number);
          paths.forEach((p, i) => stops.push(animate(p, { strokeDashoffset: [1, 0] }, { duration: dur * 0.8, delay: (i % 12) * (dur * 0.2) / 12, ease: EASE_CHART })));
        });
        at(T(OPENING.isobathsInk[1]), () => {
          // the live lines take over from the drawn ones
          map.setPaintProperty("isobaths-minor", "line-opacity", MINOR);
          map.setPaintProperty("isobaths-index", "line-opacity", INDEX);
          stops.push(animate(svg, { opacity: [1, 0] }, { duration: 0.3 }));
        });
        at(T(OPENING.tintsFlood[0]), () => {
          const span = D(OPENING.tintsFlood) * 1000;
          const each = Math.min(700, span * 0.5);
          // shallowest first: the water fills from the shore outward
          [...DEPTH_THRESHOLDS].reverse().forEach((d, i, all) => {
            const id = `depth-${d}`;
            map.setPaintProperty(id, "fill-opacity-transition", { duration: each, delay: Math.round(((span - each) * i) / (all.length - 1)) });
            map.setPaintProperty(id, "fill-opacity", 1);
          });
        });
      }
      at(T(OPENING.neatlineClose[0]), () => {
        html().setAttribute("data-nx-neat", "closing");
        const neat = document.querySelector<SVGElement>(".n12-neat");
        if (neat) stops.push(animate(neat, { clipPath: ["inset(50% 50% 50% 50%)", "inset(0% 0% 0% 0%)"] }, { duration: D(OPENING.neatlineClose), ease: EASE_CHART }));
      });
      at(T(OPENING.soundingsRegister[0]), () => {
        html().setAttribute("data-nx-marks", "on");
        const overlay = document.querySelector<HTMLElement>(".n12-overlay");
        if (overlay) stops.push(animate(overlay, { opacity: [0, 1] }, register));
        if (map) map.setPaintProperty("isobath-labels", "text-opacity", 1);
      });
      at(T(OPENING.total), finish);
    };

    // the static tier has no map: the still settles onto the baked chart
    const sheet = document.querySelector<HTMLElement>("[data-testid='n12-sheet']");
    const isStatic = () => sheet?.dataset.tier === "static";
    let started = false;
    const go = (m: MapLibreMap | null) => {
      if (started) return;
      started = true;
      run(m);
    };
    whenChartReady().then(go);
    const poll = window.setInterval(() => {
      if (isStatic()) go(null);
      // the canyon has held as long as it may: settle onto whatever is drawn (the poster) and finish
      if (elapsed() > OPENING.neatlineClose[0]) go(getChartMap());
    }, 100);
    timers.push(poll);
    return () => {
      // the layout never unmounts in use; a development double-run re-arms the opening for the second pass
      clearInterval(poll);
      mo.disconnect();
      stops.forEach((a) => a.stop());
      timers.forEach((t) => clearTimeout(t));
      if (!done && state() === "run") html().setAttribute("data-nx-opening", "on");
      done = true;
    };
  }, []);
  return null;
}

/** The margin's Skip control: visible only while the opening plays. */
export function SkipOpening() {
  return (
    <button type="button" className="n12-skip-opening" data-testid="nx-skip-opening" onClick={() => html().setAttribute("data-nx-opening", "skip")}>
      Skip
    </button>
  );
}


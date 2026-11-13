"use client";

import { useEffect, useRef } from "react";
import { getMapLayers, subscribeMapLayers } from "./mapLayers";
import { getFrame, onFrame, type ChartFrame } from "./runtime";
import { unproject } from "./mercator";

/**
 * Depth labels set along their line, as on a chart (v14 README §4, after the approved prototype's map.js): fathoms
 * first, metres small after ("55 fm 100 m"), italic, each breaking its own line. 1 fm = 1.8288 m. The data holds the
 * 20, 50, 100 and 200 m edges (the nearest levels to 10/20/30/50/100 fm): each is labelled with its true fathom value
 * (11, 27, 55, 109 fm), with the Depth layer, fading in as the camera nears the shelf. The 1,000 and 2,000 m index
 * lines (547 and 1,094 fm) are always labelled, quieter close in.
 *
 * Labels are placed when the camera settles, near the open water beside the page's objects and never under them; while
 * the camera flies they ride on the chart at the place they had (they fade, never vanish). Wide screens only, as in
 * the prototype: a phone's strip has no room for them.
 */
const FM = (m: number) => Math.round(m / 1.8288);
const CONTOURS = [20, 50, 100, 200];
const INDEX = [1000, 2000];
type Line = { d: number; pts: [number, number][]; box: [number, number, number, number] };
type Placed = { d: number; at: [number, number]; a: [number, number]; b: [number, number]; index: boolean };

let cache: Promise<Line[]> | null = null;
function lines(): Promise<Line[]> {
  return (cache ??= Promise.all([
    fetch("/chart/contours.geojson").then((r) => r.json()),
    fetch("/chart/isobaths.geojson").then((r) => r.json()),
  ])
    .then(([c, i]) => {
      type F = { properties: { d: number; i?: number }; geometry: { coordinates: [number, number][] } };
      const out: Line[] = [];
      const box = (pts: [number, number][]): Line["box"] => {
        let w = 180, s0 = 90, e = -180, n = -90;
        for (const [x, y] of pts) {
          if (x < w) w = x;
          if (x > e) e = x;
          if (y < s0) s0 = y;
          if (y > n) n = y;
        }
        return [w, s0, e, n];
      };
      for (const f of c.features as F[]) if (CONTOURS.includes(f.properties.d) && f.geometry.coordinates.length > 20) out.push({ d: f.properties.d, pts: f.geometry.coordinates, box: box(f.geometry.coordinates) });
      for (const f of i.features as F[]) if (f.properties.i === 1 && INDEX.includes(f.properties.d) && f.geometry.coordinates.length > 20) out.push({ d: f.properties.d, pts: f.geometry.coordinates, box: box(f.geometry.coordinates) });
      return out;
    })
    .catch(() => []));
}

const smooth = (a: number, b: number, v: number) => {
  const t = Math.max(0, Math.min(1, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** The rects lettering keeps clear of: the bar, the chips, the page's objects, the chart's own furniture. */
function blockers(frame: DOMRect): DOMRect[] {
  return [...document.querySelectorAll<HTMLElement>(".n12-bar > *, .nx-layers, .n12-zonepin, .n12-reg text[visibility=visible], .n12-labels .n12-mark[style*=visible], [data-nx-ticket], [data-nx-dock], [data-nx-pass], .n12-controls, .n12-scale, .n12-dock-slot, .n12-app:not(:has([data-nx-layout=stage])) .n12-margin")]
    .map((e) => e.getBoundingClientRect())
    .filter((r) => r.width && r.height)
    .map((r) => new DOMRect(r.left - frame.left, r.top - frame.top, r.width, r.height));
}

export function DepthLabels() {
  const svgRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    let data: Line[] = [];
    let placed: Placed[] = [];
    let settle = 0;
    let alive = true;

    const write = (f: ChartFrame) => {
      const z = f.camera.zoom;
      const depthOn = !!getMapLayers().on.depth;
      // fade by zoom as the prototype did (px per degree 170 → 330 is about zoom 6.9 → 7.9)
      const contourA = depthOn ? smooth(6.9, 7.9, z) : 0;
      const indexA = 1 - 0.6 * smooth(8.5, 9.3, z);
      const nodes = svg.children;
      placed.forEach((p, i) => {
        const g = nodes[i] as SVGGElement | undefined;
        if (!g) return;
        const c = f.project(p.at[0], p.at[1]);
        const a = f.project(p.a[0], p.a[1]);
        const b = f.project(p.b[0], p.b[1]);
        let ang = (Math.atan2(a.y - b.y, a.x - b.x) * 180) / Math.PI;
        if (ang > 90) ang -= 180;
        if (ang < -90) ang += 180;
        g.setAttribute("transform", `translate(${c.x.toFixed(1)} ${c.y.toFixed(1)}) rotate(${ang.toFixed(1)})`);
        g.style.opacity = String(p.index ? indexA : contourA);
      });
    };

    const place = (f: ChartFrame) => {
      // not while the opening plays (its lettering waits for its beat); placed when it ends
      const op = document.documentElement.getAttribute("data-nx-opening");
      if ((op === "on" || op === "run") && !document.documentElement.hasAttribute("data-nx-marks")) return;
      const { width: W, height: H } = f.viewport;
      svg.replaceChildren();
      placed = [];
      if (W <= 640 || !data.length) return;
      const fr = svg.getBoundingClientRect();
      const keep = blockers(fr);
      const bottom = H - Math.max(30, f.insets.bottom);
      const blocked = (x: number, y: number) => keep.some((r) => x > r.left - 10 && x < r.right + 10 && y > r.top - 12 && y < r.bottom + 12);
      const used: [number, number][] = [];
      // only the lines whose box reaches the view are walked (the whole coast's lines are thousands of points)
      const [vw, vn] = unproject(f.camera, f.viewport, 0, 0);
      const [ve, vs] = unproject(f.camera, f.viewport, W, H);
      const inView = data.filter((l) => l.box[2] >= vw && l.box[0] <= ve && l.box[3] >= vs && l.box[1] <= vn);
      const target0: [number, number] = [W * 0.26, H * 0.34];
      const pick = (ds: number[], n: number, at: [number, number]) => {
        let best: { i: number; line: Line; p: [number, number] } | null = null;
        let bd = 1e12;
        for (const line of inView) {
          if (!ds.includes(line.d)) continue;
          const pts = line.pts;
          for (let i = 6; i < pts.length - 8; i += 3) {
            const q = f.project(pts[i][0], pts[i][1]);
            if (q.x < 40 || q.x > W - 70 || q.y < 100 || q.y > bottom - 12 || blocked(q.x, q.y)) continue;
            const a = f.project(pts[i + 6][0], pts[i + 6][1]), b = f.project(pts[i - 6][0], pts[i - 6][1]);
            const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy) || 1, ux = (dx / l) * 46, uy = (dy / l) * 46;
            if (blocked(q.x + ux, q.y + uy) || blocked(q.x - ux, q.y - uy)) continue;
            const dd = (q.x - at[0]) ** 2 + (q.y - at[1]) ** 2;
            if (dd < bd) {
              bd = dd;
              best = { i, line, p: [q.x, q.y] };
            }
          }
        }
        if (!best || bd > (W * 0.5) ** 2 || used.some((u) => Math.hypot(u[0] - best!.p[0], u[1] - best!.p[1]) < 70)) return;
        used.push(best.p);
        const pts = best.line.pts;
        placed.push({ d: best.line.d, at: pts[best.i], a: pts[best.i + 6], b: pts[best.i - 6], index: n < 0 });
      };
      if (getMapLayers().on.depth && f.camera.zoom > 6.7) CONTOURS.forEach((d, n) => pick([d], n, [target0[0] + n * 0.06 * W, target0[1] + n * 0.05 * H]));
      INDEX.forEach((d) => pick([d], -1, target0));
      const ns = "http://www.w3.org/2000/svg";
      for (const p of placed) {
        const g = document.createElementNS(ns, "g");
        g.setAttribute("class", "n12-depth");
        const t = document.createElementNS(ns, "text");
        t.setAttribute("text-anchor", "middle");
        t.setAttribute("dominant-baseline", "middle");
        const a = document.createElementNS(ns, "tspan");
        a.setAttribute("class", "n12-depth-fm");
        a.textContent = `${FM(p.d).toLocaleString("en-US")} fm`;
        const b = document.createElementNS(ns, "tspan");
        b.setAttribute("class", "n12-depth-m");
        b.setAttribute("dx", "5");
        b.textContent = `${p.d.toLocaleString("en-US")} m`;
        t.append(a, b);
        g.append(t);
        svg.append(g);
      }
      write(f);
    };

    const onF = (f: ChartFrame) => {
      write(f);
      window.clearTimeout(settle);
      settle = window.setTimeout(() => alive && place(getFrame() ?? f), 160);
    };
    lines().then((d) => {
      if (!alive) return;
      data = d;
      const f = getFrame();
      if (f) place(f);
    });
    const off = onFrame(onF);
    const mo = new MutationObserver(() => {
      const f = getFrame();
      if (f && alive) place(f);
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-nx-opening", "data-nx-marks"] });
    // a chip switched (or a page's defaults): fade at once, place again a moment later, off the page change's frames
    let relayout = 0;
    const offL = subscribeMapLayers(() => {
      const f = getFrame();
      if (f) write(f);
      window.clearTimeout(relayout);
      relayout = window.setTimeout(() => {
        const g = getFrame();
        if (alive && g) place(g);
      }, 420);
    });
    return () => {
      alive = false;
      off();
      offL();
      mo.disconnect();
      window.clearTimeout(settle);
      window.clearTimeout(relayout);
    };
  }, []);
  return <svg ref={svgRef} className="n12-depths" aria-hidden="true" data-testid="nx-depth-labels" />;
}

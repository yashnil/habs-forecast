"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cssColour, onModeChange, type RGBA } from "@/nx/chart/layers/colors";
import { decodeValue, gridBounds, mercY } from "@/nx/chart/layers/grid";
import { screenRect, visibleBounds } from "@/nx/chart/mercator";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { loadCodes } from "./codes";
import type { GridSpec } from "./model";

type Kind = "ramp" | "outlook" | "classes";

/**
 * The static tier's data layer (README §8: "static: baked chart image of the same view, same notes"): with no
 * WebGL the frame is baked once on the device into an image, the same marks as the live layer in the same
 * tokens: the chlorophyll ramp with its "no view" hatch, the outlook hatched as a model, C-HARM in its ten
 * display classes under the model hatch. It lies on the baked chart under the chart's names; a new frame is a
 * new bake (a cut, T26). Temperature is drawn as isotherms live and has no static bake (the legend says so).
 */
export function StaticField({ base, grid, kind, past, domain }: { base: string; grid: GridSpec; kind: Kind; past?: boolean; domain: [number, number] }) {
  const [url, setUrl] = useState<string | null>(null);
  const [mode, setMode] = useState(0);
  const imgRef = useRef<HTMLImageElement>(null);
  const [target, setTarget] = useState<HTMLElement | null>(null);
  useEffect(() => setTarget(document.getElementById("nx-chart-underlay")), []);
  useEffect(() => onModeChange(() => setMode((m) => m + 1)), []);

  useEffect(() => {
    let live = true;
    const f = getFrame();
    const box = f ? (visibleBounds(f.camera, f.viewport).flat() as [number, number, number, number]) : null;
    loadCodes(base, grid, box)
      .then((c) => {
        if (!live) return;
        return landRings().then((land) => live && setUrl(bake(grid, c.codes, kind, !!past, domain, land)));
      })
      .catch(() => live && setUrl(null));
    return () => {
      live = false;
    };
  }, [base, grid, kind, past, domain, mode]);

  // placed by the chart's own projection, like the baked chart under it
  useEffect(() => {
    const [w, s, e, n] = gridBounds(grid);
    const place = (f: ChartFrame | null) => {
      const img = imgRef.current;
      if (!f || !img) return;
      const r = screenRect(f.camera, f.viewport, [
        [w, s],
        [e, n],
      ]);
      Object.assign(img.style, { transform: `translate(${r.left}px, ${r.top}px)`, width: `${r.width}px`, height: `${r.height}px` });
    };
    place(getFrame());
    return onFrame(place);
  }, [grid, url]);

  if (!target || !url) return null;
  // eslint-disable-next-line @next/next/no-img-element
  return createPortal(<img ref={imgRef} src={url} alt="" className="n12-static-field" data-testid="nx-static-field" />, target);
}

const tokenList = (prefix: string, n: number): RGBA[] => {
  const cs = getComputedStyle(document.documentElement);
  return Array.from({ length: n }, (_, i) => cssColour(cs.getPropertyValue(`--${prefix}-${i}`).trim()));
};
const tok = (name: string) => cssColour(getComputedStyle(document.documentElement).getPropertyValue(name).trim());

/** Composite `top` (straight alpha `a`) over `under` (premultiplied), returning premultiplied. */
function over(under: [number, number, number, number], top: RGBA, a: number): [number, number, number, number] {
  const ta = top[3] * a;
  return [top[0] * ta + under[0] * (1 - ta), top[1] * ta + under[1] * (1 - ta), top[2] * ta + under[2] * (1 - ta), ta + under[3] * (1 - ta)];
}

type Ring = [number, number][];
let landP: Promise<Ring[]> | null = null;
/** The chart's own land (the same asset the live chart draws): a grid's empty cells over land are land, not "no view". */
function landRings(): Promise<Ring[]> {
  landP ??= fetch("/chart/land.geojson")
    .then((r) => r.json())
    .then((j: { features: Array<{ geometry: { type: string; coordinates: unknown } }> }) =>
      j.features.flatMap((f) => (f.geometry.type === "Polygon" ? (f.geometry.coordinates as Ring[]) : f.geometry.type === "MultiPolygon" ? (f.geometry.coordinates as Ring[][]).flat() : [])),
    )
    .catch(() => []);
  return landP;
}

function bake(g: GridSpec, codes: Uint16Array, kind: Kind, past: boolean, domain: [number, number], land: Ring[]): string | null {
  const chl = tokenList("chl", 8);
  const prob = tokenList("prob", 10);
  const ink = tok("--ink"), ink2 = tok("--ink-2"), hatchCol = tok("--hatch"), paper = tok("--paper");
  const [w, s, e, n] = gridBounds(g);
  // the image is drawn in Web Mercator rows, so it registers with the baked chart at every latitude
  const W = Math.min(1600, g.width * 2);
  const yN = mercY(n), yS = mercY(s);
  const H = Math.max(1, Math.round((W * (yS - yN)) / ((e - w) / 360)));
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(W, H);
  const [lo, hi] = [Math.log10(domain[0]), Math.log10(domain[1])];
  const ramp = (v: number): RGBA => {
    const t = Math.min(1, Math.max(0, (Math.log10(Math.max(v, 1e-9)) - lo) / (hi - lo))) * 7;
    const i = Math.min(6, Math.floor(t));
    const f = t - i;
    return [0, 1, 2, 3].map((k) => chl[i][k] + (chl[i + 1][k] - chl[i][k]) * f) as RGBA;
  };
  for (let y = 0; y < H; y++) {
    const my = yN + ((y + 0.5) / H) * (yS - yN);
    const lat = (Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180) / Math.PI;
    const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
    for (let x = 0; x < W; x++) {
      const lon = w + ((x + 0.5) / W) * (e - w);
      const col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
      if (row < 0 || col < 0 || row >= g.height || col >= g.width) continue;
      const v = decodeValue(g, codes[row * g.width + col]);
      const line = (spacing: number) => ((x + y) % spacing) < 1.2;
      // the grid's own extent ends at a neatline; "no view" comes in over the first pixels inside it
      const edge = Math.min(x, y, W - 1 - x, H - 1 - y);
      let px: [number, number, number, number] = [0, 0, 0, 0];
      if (kind === "ramp" && edge < 1) px = over(px, ink2, 0.7);
      else if (v === null) {
        if (kind !== "ramp") {
          // "outside the model": level rules over a paper veil, as the live layer draws it (FieldLayer outsideModel)
          px = over(px, paper, 0.3);
          if (y % 7 < 1.2) px = over(px, ink2, 0.42);
          const k = (y * W + x) * 4;
          img.data[k] = (px[0] / px[3]) * 255;
          img.data[k + 1] = (px[1] / px[3]) * 255;
          img.data[k + 2] = (px[2] / px[3]) * 255;
          img.data[k + 3] = px[3] * 255;
          continue;
        }
        const inA = Math.min(1, edge / 24);
        px = over(px, paper, 0.22 * inA);
        if (line(11)) px = over(px, hatchCol, 0.32 * inA);
      } else if (kind === "classes") {
        px = over(px, prob[Math.max(0, Math.min(9, Math.floor(v * 10)))], past ? 0.42 : 0.72);
        if (line(8)) px = over(px, past ? ink2 : ink, 0.2);
      } else {
        px = over(px, ramp(v), kind === "outlook" ? 0.9 : 1);
        if (kind === "outlook" && line(8)) px = over(px, ink, 0.28);
      }
      const k = (y * W + x) * 4;
      const a = px[3] || 1;
      img.data[k] = (px[0] / a) * 255;
      img.data[k + 1] = (px[1] / a) * 255;
      img.data[k + 2] = (px[2] / a) * 255;
      img.data[k + 3] = px[3] * 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // land masks the data, as the live chart's opaque land does
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  for (const ring of land) {
    ring.forEach(([lon, lat], i) => {
      const px = ((lon - w) / (e - w)) * W, py = ((mercY(lat) - yN) / (yS - yN)) * H;
      if (i) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    });
    ctx.closePath();
  }
  ctx.fill("evenodd");
  return cv.toDataURL("image/png");
}

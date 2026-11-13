"use client";

import { unproject, project, type Camera, type Insets, type Viewport } from "./mercator";

/** The graticule's latitude step at a zoom (the map style's own graticule filter, style.ts): 1°, ½°, ¼°. */
export const graticuleStep = (zoom: number) => (zoom >= 8.6 ? 0.25 : zoom >= 7.2 ? 0.5 : 1);

/** "39°30′N": a parallel as the chart's border writes it. */
export function parallelLabel(lat: number): string {
  const d = Math.floor(lat + 1e-9);
  const m = Math.round((lat - d) * 60);
  return `${d}°${String(m).padStart(2, "0")}′N`;
}

/**
 * The chart's border, v14 (the synthesis has no neatline): each parallel of the graticule named at the screen's right
 * edge, over the land, as on a printed chart's border. A label is written whole or not at all: never under the bar,
 * the chart's foot furniture, a ticket or a dock standing at that edge.
 */
export function GraticuleLabels({ camera, viewport, insets }: { camera: Camera | null; viewport: Viewport | null; insets: Insets }) {
  if (!camera || !viewport || viewport.width <= 640) return null;
  const { width: W, height: H } = viewport;
  const step = graticuleStep(camera.zoom);
  const [, n] = unproject(camera, viewport, 0, 0);
  const [, s] = unproject(camera, viewport, 0, H);
  const blocks =
    typeof document === "undefined"
      ? []
      : [...document.querySelectorAll<HTMLElement>("[data-nx-ticket], [data-nx-dock], [data-nx-pass], .n12-controls, .n12-margin")]
          .map((e) => e.getBoundingClientRect())
          .filter((r) => r.width && r.right > W - 90 && r.left < W - 6);
  const labels: { y: number; t: string }[] = [];
  for (let la = Math.ceil(s / step) * step; la < n; la += step) {
    const y = project(camera, viewport, 0, la).y;
    if (y < 84 || y > H - Math.max(60, insets.bottom + 30)) continue;
    if (blocks.some((r) => y > r.top - 6 && y < r.bottom + 14)) continue;
    labels.push({ y, t: parallelLabel(la) });
  }
  return (
    <svg className="n12-grat" width={W} height={H} aria-hidden="true" data-testid="nx-neatline">
      {labels.map((l) => (
        <text key={l.t} x={W - 10} y={l.y - 5} textAnchor="end">
          {l.t}
        </text>
      ))}
    </svg>
  );
}

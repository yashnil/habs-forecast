"use client";

import { formatDegrees, ticks, unproject, type Camera, type Viewport } from "./mercator";

/**
 * The graduated neatline (README §6): a narrow band around the sheet, alternating every major interval like a
 * chart's latitude and longitude scale, with real ticks and labels that follow the camera. Labels sit above
 * the top band and beside the left band; the band itself is drawn on all four sides.
 *
 * `pane` is the part of the sheet a panel takes on either side (Research's record): the neatline then frames only
 * the chart's own pane, and paper fills the rest, so no graduation runs on over the panel (critic finding 7).
 */
export function Neatline({ camera, viewport, band, label, pane = { left: 0, right: 0 } }: { camera: Camera | null; viewport: Viewport | null; band: number; label: number; pane?: { left: number; right: number; bottom?: number } }) {
  if (!camera || !viewport || viewport.width < 10) return null;
  const { width: SW, height: H } = viewport;
  const px0 = Math.max(0, pane.left), px1 = Math.max(px0 + 10, SW - Math.max(0, pane.right));
  const W = px1 - px0;
  const [w, n] = unproject(camera, viewport, px0, 0);
  const [e, s] = unproject(camera, viewport, px1, H);
  const lon = ticks(w, e, W, W < 500 ? 120 : 150);
  const lat = ticks(s, n, H, H < 400 ? 90 : 120);
  const x = (v: number) => ((v - w) / (e - w)) * W;
  // latitude is not linear on Mercator: place each by projection
  const y = (v: number) => {
    const t = (Math.log(Math.tan(Math.PI / 4 + (v * Math.PI) / 360)) - Math.log(Math.tan(Math.PI / 4 + (n * Math.PI) / 360))) / (Math.log(Math.tan(Math.PI / 4 + (s * Math.PI) / 360)) - Math.log(Math.tan(Math.PI / 4 + (n * Math.PI) / 360)));
    return t * H;
  };
  // alternating fills between major ticks (the "graduated" band)
  const lonEdges = [w, ...lon.major, e];
  const latEdges = [n, ...[...lat.major].reverse(), s];
  const parity = (v: number, step: number) => Math.floor(v * 60 / step + 1e-6) % 2 === 0;
  const T = band + label;
  return (
    <svg className="n12-neat" width={SW + band * 2 + label} height={H + band * 2 + label} style={{ left: -band - label, top: -band - label }} aria-hidden="true" data-testid="nx-neatline">
      {/* paper over the sheet outside the chart's pane (under the panel, and the gap beside it) */}
      {px0 > 0 ? <rect className="n12-neat-paper" x={0} y={0} width={T + px0 - band} height={H + band * 2 + label} /> : null}
      {px1 < SW ? <rect className="n12-neat-paper" x={T + px1 + band} y={0} width={SW - px1 + label} height={H + band * 2 + label} /> : null}
      <g transform={`translate(${T + px0} ${T})`}>
        {/* the band: alternate segments filled, the others left as paper, outlined in a hairline */}
        {lonEdges.slice(0, -1).map((a, i) => {
          const b = lonEdges[i + 1];
          if (!parity(a, lon.stepMin)) return null;
          return (
            <g key={`x${i}`} className="n12-neat-fill">
              <rect x={x(a)} y={-band} width={x(b) - x(a)} height={band} />
              <rect x={x(a)} y={H} width={x(b) - x(a)} height={band} />
            </g>
          );
        })}
        {latEdges.slice(0, -1).map((a, i) => {
          const b = latEdges[i + 1];
          if (!parity(b, lat.stepMin)) return null;
          return (
            <g key={`y${i}`} className="n12-neat-fill">
              <rect x={-band} y={y(a)} width={band} height={y(b) - y(a)} />
              <rect x={W} y={y(a)} width={band} height={y(b) - y(a)} />
            </g>
          );
        })}
        <rect className="n12-neat-line" x={-band} y={-band} width={W + band * 2} height={H + band * 2} />
        <rect className="n12-neat-line" x={0} y={0} width={W} height={H} />
        {lon.minor.map((v) => (
          <line key={`lm${v}`} className="n12-neat-tick" x1={x(v)} x2={x(v)} y1={-band} y2={-band - 3} />
        ))}
        {lat.minor.map((v) => (
          <line key={`am${v}`} className="n12-neat-tick" y1={y(v)} y2={y(v)} x1={-band} x2={-band - 3} />
        ))}
        {lon.major.map((v) => (
          <g key={`lM${v}`}>
            <line className="n12-neat-tick" x1={x(v)} x2={x(v)} y1={-band} y2={-band - 6} />
            {x(v) > 40 && x(v) < W - 40 ? (
              <text className="n12-neat-label" x={x(v)} y={-band - 7} textAnchor="middle">
                {formatDegrees(v, "lon")}
              </text>
            ) : null}
          </g>
        ))}
        {lat.major.map((v) => (
          <g key={`aM${v}`}>
            <line className="n12-neat-tick" y1={y(v)} y2={y(v)} x1={-band} x2={-band - 6} />
            {/* a label is written whole or not at all: never cut by the frame's corner or by a sheet resting over the chart's foot */}
            {y(v) > 44 && y(v) < H - Math.max(0, pane.bottom ?? 0) - 44 ? (
              <text className="n12-neat-label" transform={`translate(${-band - 7} ${y(v)}) rotate(-90)`} textAnchor="middle">
                {formatDegrees(v, "lat")}
              </text>
            ) : null}
          </g>
        ))}
      </g>
    </svg>
  );
}

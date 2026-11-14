"use client";
import { svgId, useElementWidth } from "@/nx/ds";
import { curveCatmullRom, line } from "d3-shape";
import { useId, useMemo, type ReactNode } from "react";
import { plates, type PlateData } from "./data";
import s from "./catalog.module.css";

/**
 * A static chart plate for the catalog: real coastline and isobaths projected to SVG. It stands in for the
 * persistent chart host, which owns projection in the product; here `children(k)` receives the scale from
 * plate pixels to screen pixels so overlays land where they would.
 */
export function Plate({ which = "bay", children, under, rail, label }: { which?: "bay" | "chl"; children?: (k: number) => ReactNode; under?: ReactNode; rail?: boolean; label: string }) {
  const p: PlateData = plates[which];
  const [ref, w] = useElementWidth<HTMLDivElement>(640);
  const k = w / p.width;
  const clip = svgId(useId());
  // ETOPO isobaths are smoothed and simplified (T30); the curve only removes the corners of simplification
  const iso = useMemo(() => {
    const g = line<[number, number]>().curve(curveCatmullRom.alpha(0.5));
    return p.isobaths.map((i) => ({ depth: i.depth_m, d: (g(i.pts) ?? "").replace(/-?\d+\.\d+/g, (m) => (+m).toFixed(1)) }));
  }, [p]);
  return (
    <figure className={s.plate} aria-label={label}>
      <div ref={ref} className={s.plateInner} style={{ aspectRatio: `${p.width} / ${p.height}` }}>
        <svg className={s.plateSvg} viewBox={`0 0 ${p.width} ${p.height}`} preserveAspectRatio="xMidYMid slice" aria-hidden focusable="false">
          <rect width={p.width} height={p.height} className={s.land} />
          <defs>
            <clipPath id={clip}>
              <path d={p.water} clipRule="evenodd" />
            </clipPath>
          </defs>
          <path d={p.water} className={s.water} fillRule="evenodd" />
          <g clipPath={`url(#${clip})`}>
            <path d={p.coast} className={s.shoalMid} />
            <path d={p.coast} className={s.shoalNear} />
            {under}
          </g>
          {iso.map((i, n) => (
            <path key={n} d={i.d} className={i.depth === 200 ? s.iso200 : s.iso} />
          ))}
          <path d={p.coast} className={s.coast} />
          {rail ? <rect x={p.width - 4} y={0} width={4} height={p.height} className={s.regRail} /> : null}
        </svg>
        <div className={s.overlay}>{children ? children(k) : null}</div>
      </div>
      <figcaption className={s.plateCap}>Coastline OpenMapTiles; isobaths NOAA ETOPO 2022, 200 to 2,000 m. Not for navigation.</figcaption>
    </figure>
  );
}

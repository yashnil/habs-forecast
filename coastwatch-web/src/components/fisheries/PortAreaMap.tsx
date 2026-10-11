"use client";

import { useMemo, useState } from "react";
import geo from "@/content/fisheries-geo.json";

type Port = (typeof geo.ports)[number];

const LON0 = -124.7;
const LON1 = -114.0;
const LAT0 = 32.4;
const LAT1 = 42.1;
const K = 25; // px per degree of latitude
const COS = Math.cos((37.3 * Math.PI) / 180);
const LABEL_W = 116;
const MAP_W = (LON1 - LON0) * COS * K;
const W = Math.ceil(LABEL_W + MAP_W);
const H = Math.ceil((LAT1 - LAT0) * K) + 8;
const px = (lon: number) => LABEL_W + (lon - LON0) * COS * K;
const py = (lat: number) => 4 + (LAT1 - lat) * K;
/** CDFW's two port areas away from the open coast; drawn lighter and named in the note. */
const INLAND = new Set([10, 11]);

/**
 * Where landings are reported (M7, decision F2): CDFW's Marine Landing Data System ports, coloured by
 * nothing but labelled by CDFW port area. CDFW publishes points, not area boundaries, so no areas are
 * drawn, and no value is attached to any port or area: NOAA's figures on this page are statewide.
 */
export function PortAreaMap() {
  const [hot, setHot] = useState<number | null>(null);
  const areas = useMemo(() => {
    const by = new Map<number, { code: number; name: string; ports: Port[] }>();
    for (const p of geo.ports) {
      const a = by.get(p.area_code) ?? { code: p.area_code, name: p.area, ports: [] };
      a.ports.push(p);
      by.set(p.area_code, a);
    }
    const coastal = [...by.values()].filter((a) => !INLAND.has(a.code)).sort((a, b) => a.code - b.code);
    // label at the area's northernmost-to-southernmost midpoint, then pushed apart (≥ 17 px)
    let prev = -Infinity;
    const placed = coastal.map((a) => {
      const lats = a.ports.map((p) => p.lat);
      const mid = (Math.max(...lats) + Math.min(...lats)) / 2;
      const target = py(mid);
      const y = Math.max(target, prev + 17);
      prev = y;
      const anchor = a.ports.reduce((w, p) => (p.lon < w.lon ? p : w), a.ports[0]); // westernmost port
      return { ...a, y, ax: px(anchor.lon), ay: py(mid) };
    });
    return { coastal: placed, inland: [...by.values()].filter((a) => INLAND.has(a.code)) };
  }, []);
  const outline = useMemo(() => geo.outline.map((ring) => `M${ring.map(([x, y]) => `${px(x).toFixed(1)},${py(y).toFixed(1)}`).join("L")}Z`).join(""), []);

  return (
    <figure className="space-y-2" data-testid="port-area-map">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className="block h-auto max-w-full"
        role="img"
        aria-label={`Map of California with ${geo.ports.length} CDFW landing ports in ${areas.coastal.length + areas.inland.length} port areas: ${areas.coastal.map((a) => a.name).join(", ")}, plus ${areas.inland.map((a) => a.name).join(" and ")}. No landings values are shown.`}
      >
        <path d={outline} fill="var(--cw-surface-3)" stroke="var(--cw-hairline-strong)" strokeWidth={1} />
        {geo.ports.map((p) => {
          const on = hot === p.area_code;
          return (
            <circle
              key={p.code}
              cx={px(p.lon)}
              cy={py(p.lat)}
              r={on ? 3.2 : 2.2}
              fill={INLAND.has(p.area_code) ? "var(--cw-ink-3)" : on ? "var(--cw-ink)" : "var(--cw-history)"}
              opacity={INLAND.has(p.area_code) ? 0.45 : hot != null && !on ? 0.35 : 0.9}
            />
          );
        })}
        {areas.coastal.map((a) => {
          const on = hot === a.code;
          return (
            <g key={a.code} onMouseEnter={() => setHot(a.code)} onMouseLeave={() => setHot(null)} data-testid={`port-area-${a.code}`}>
              <path d={`M${LABEL_W - 8},${a.y - 4} L${a.ax - 5},${a.ay}`} stroke={on ? "var(--cw-ink)" : "var(--cw-hairline-strong)"} fill="none" />
              <text x={LABEL_W - 11} y={a.y} textAnchor="end" className={`text-[12.5px] ${on ? "fill-[var(--cw-ink)] font-semibold" : "fill-[var(--cw-ink-2)]"}`}>
                {a.name}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="text-[12.5px] leading-snug text-ink-3">
        Dots: {geo.ports.length} ports where commercial landings are reported to CDFW, labelled by CDFW port area; {areas.inland.map((a) => a.name).join(" and ")} (lighter) are inland. CDFW publishes these as approximate points and no area boundaries, so none are drawn. Data:{" "}
        <a href={geo.ports_source.url} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-ink">
          CDFW Marine Region, BIOS ds3081
        </a>{" "}
        ({geo.ports_source.license}); outline: Natural Earth.
      </figcaption>
      <details className="text-[13px] text-ink-2" data-testid="port-area-list">
        <summary className="cursor-pointer font-medium text-ink">Ports in each area</summary>
        <dl className="mt-2 space-y-1.5">
          {[...areas.coastal, ...areas.inland].map((a) => (
            <div key={a.code}>
              <dt className="inline font-medium text-ink">{a.name}: </dt>
              <dd className="inline">{a.ports.map((p) => p.name).join(", ")}</dd>
            </div>
          ))}
        </dl>
      </details>
    </figure>
  );
}

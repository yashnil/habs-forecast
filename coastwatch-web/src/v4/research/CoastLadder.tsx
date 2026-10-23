"use client";

import { useState } from "react";

/**
 * The coast ladder (v7): every shore station at its true latitude and longitude along a stylised
 * California coast, its name set in a column beside it with a leader line to its dot. It is the
 * station picker and the spatial context in one, with no second map. Longitude is compressed (the
 * coast keeps its shape, not its proportions); latitude is true, with whole-degree ticks.
 */

export type LadderStation = { id: string; name: string; lat: number; lon: number; last: string };

/** A simplified California shoreline, north to south: (lat, lon). Drawn, not measured. */
const COAST: [number, number][] = [
  [42.0, -124.21], [41.75, -124.2], [41.06, -124.14], [40.8, -124.2], [40.44, -124.41], [39.8, -123.85],
  [38.95, -123.74], [38.3, -123.07], [38.0, -123.02], [37.81, -122.52], [37.5, -122.49], [37.1, -122.33],
  [36.96, -122.02], [36.8, -121.79], [36.6, -121.89], [36.3, -121.9], [35.67, -121.28], [35.37, -120.86],
  [35.17, -120.74], [34.58, -120.64], [34.45, -120.47], [34.41, -119.69], [34.1, -119.1], [34.03, -118.75],
  [34.01, -118.5], [33.75, -118.41], [33.6, -117.93], [33.46, -117.71], [33.0, -117.28], [32.87, -117.25],
  [32.53, -117.12],
];

const LAT_N = 42.2;
const LAT_S = 32.3;
const LON_W = -124.6;
const LON_E = -116.9;
const TICK_W = 28; // latitude labels
const BAND = 54; // the coast drawing
const LABEL_X = TICK_W + BAND + 12;
const PAD = 10;
/** lastHabWords' lead: said to a screen reader; the column's heading says it to the eye */
const LAST = "last HAB sample ";

export function CoastLadder({ stations, selected, onSelect, height = 640 }: { stations: LadderStation[]; selected: string; onSelect: (id: string) => void; height?: number }) {
  const [hover, setHover] = useState<string | null>(null);
  const H = height;
  const y = (lat: number) => PAD + ((LAT_N - lat) / (LAT_N - LAT_S)) * (H - 2 * PAD);
  const x = (lon: number) => TICK_W + 6 + ((lon - LON_W) / (LON_E - LON_W)) * (BAND - 12);
  const rowH = Math.min(36, Math.floor((H - 2 * PAD) / Math.max(1, stations.length)));
  // labels at their latitude where there is room, pushed apart where stations crowd (Tomales, Morro)
  const sorted = [...stations].sort((a, b) => b.lat - a.lat);
  const top: number[] = [];
  sorted.forEach((s, i) => {
    const ideal = y(s.lat) - rowH / 2;
    top.push(Math.max(PAD - 4, ideal, i ? top[i - 1] + rowH : -Infinity));
  });
  for (let i = sorted.length - 1; i >= 0; i--) top[i] = Math.min(top[i], i === sorted.length - 1 ? H - PAD - rowH + 4 : top[i + 1] - rowH);
  const coast = COAST.map(([la, lo], i) => `${i ? "L" : "M"}${x(lo).toFixed(1)},${y(la).toFixed(1)}`).join(" ");
  const ticks = [42, 40, 38, 36, 34, 32].filter((t) => t <= LAT_N && t >= LAT_S);
  return (
    <div className="relative w-full" style={{ height: H }} data-testid="coast-ladder">
      <svg aria-hidden className="absolute inset-0 h-full w-full overflow-visible" width="100%" height={H}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={TICK_W - 6} x2={TICK_W} y1={y(t)} y2={y(t)} stroke="var(--nx-line-strong)" />
            <text x={TICK_W - 9} y={y(t) + 3.5} textAnchor="end" fontSize="10" fill="var(--nx-ink-3)" fontFamily="var(--nx-font-data)">
              {t}°
            </text>
          </g>
        ))}
        <line x1={TICK_W} x2={TICK_W} y1={y(LAT_N)} y2={y(LAT_S)} stroke="var(--nx-line)" />
        <path d={coast} fill="none" stroke="var(--nx-ink-3)" strokeWidth={1} strokeLinejoin="round" strokeLinecap="round" opacity={0.6} />
        {sorted.map((s, i) => {
          const on = s.id === selected;
          const hot = on || s.id === hover;
          const dx = x(s.lon);
          const dy = y(s.lat);
          const ly = top[i] + rowH / 2;
          return (
            <g key={s.id}>
              <path d={`M${dx},${dy} H${TICK_W + BAND} L${LABEL_X - 4},${ly}`} fill="none" stroke={on ? "var(--nx-measured)" : hot ? "var(--nx-ink-2)" : "var(--nx-line-strong)"} strokeWidth={on ? 1.2 : 0.8} opacity={hot ? 1 : 0.7} />
              {on && <circle cx={dx} cy={dy} r={8} fill="var(--nx-measured-bg)" stroke="var(--nx-measured-line)" strokeWidth={1} />}
              <circle cx={dx} cy={dy} r={on ? 4.2 : hot ? 3.4 : 2.6} fill={on ? "var(--nx-measured)" : hot ? "var(--nx-ink)" : "var(--nx-ink-3)"} />
            </g>
          );
        })}
      </svg>
      <ul className="absolute inset-0">
        {sorted.map((s, i) => {
          const on = s.id === selected;
          return (
            <li key={s.id} className="absolute right-0" style={{ top: top[i], left: LABEL_X - 8, height: rowH }}>
              <button
                type="button"
                onClick={() => onSelect(s.id)}
                onPointerEnter={() => setHover(s.id)}
                onPointerLeave={() => setHover((h) => (h === s.id ? null : h))}
                onFocus={() => setHover(s.id)}
                onBlur={() => setHover((h) => (h === s.id ? null : h))}
                aria-current={on ? "true" : undefined}
                className={`nx-focus relative flex h-full w-full min-w-0 flex-col justify-center rounded-[3px] px-2 text-left transition-colors duration-150 ${on ? "bg-n-measured-bg" : "hover:bg-n-surface-2"}`}
                data-testid={`ladder-${s.id}`}
              >
                {on && <span aria-hidden className="absolute inset-y-1 left-0 w-[2px] rounded-full bg-n-measured" />}
                <span className={`block truncate text-[13px] leading-[15px] ${on ? "font-medium text-n-ink" : "text-n-ink-2"}`}>{s.name}</span>
                <span className="block truncate font-[family-name:var(--nx-font-data)] text-[10.5px] leading-[13px] text-n-ink-3">
                  {s.last.startsWith(LAST) ? (
                    <>
                      <span className="sr-only">{LAST}</span>
                      {s.last.slice(LAST.length)}
                    </>
                  ) : (
                    s.last
                  )}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

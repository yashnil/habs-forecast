"use client";

import { useEffect, useRef, useState } from "react";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import type { CurrentField } from "@/lib/currents";
import { currentsAt, forecastAt, satelliteAt, type Readout } from "@/lib/readout";

export type HoverSource =
  | { kind: "forecast"; layer: LayerArtifact }
  | { kind: "satellite"; layer: LayerArtifact }
  | { kind: "currents"; layer: LayerArtifact; field: CurrentField }
  | null;

/**
 * Desktop hover readout (M5): the exact published value of the cell under the pointer, with
 * its date and source, in a small label beside the cursor. The latest pointer position wins;
 * grids are fetched once and cached. Click opens the full inspector, which is also the
 * keyboard and screen-reader route to the same values, so this label is aria-hidden.
 */
export function HoverReadout({ manifest, baseUrl, source, point, screen, bounds }: { manifest: Manifest; baseUrl: string; source: HoverSource; point: { lat: number; lon: number } | null; screen: { x: number; y: number } | null; bounds: { w: number; h: number } }) {
  const [r, setR] = useState<Readout | null>(null);
  const seq = useRef(0);
  useEffect(() => {
    const n = ++seq.current;
    if (!source || !point) {
      setR(null);
      return;
    }
    const run = async (): Promise<Readout> => {
      if (source.kind === "forecast") return forecastAt(baseUrl, source.layer, point.lat, point.lon);
      if (source.kind === "satellite") return satelliteAt(manifest, baseUrl, source.layer, point.lat, point.lon);
      return currentsAt(source.field, source.layer, point.lat, point.lon);
    };
    run()
      .then((x) => n === seq.current && setR(x))
      .catch(() => n === seq.current && setR(null));
  }, [manifest, baseUrl, source, point]);

  if (!r || !screen || !point) return null;
  const W = 236;
  const left = screen.x + 16 + W > bounds.w ? screen.x - 16 - W : screen.x + 16;
  const top = Math.min(Math.max(8, screen.y + 16), bounds.h - 96);
  return (
    <div
      aria-hidden
      data-testid="hover-readout"
      className="pointer-events-none absolute z-30 rounded-lg bg-navy-950/[0.93] px-2.5 py-1.5 text-[12px] leading-snug text-on-navy-2 shadow-[0_6px_18px_rgba(4,11,23,0.4)] ring-1 ring-white/10 backdrop-blur-sm"
      style={{ left, top, width: W }}
    >
      {r.value != null ? (
        <>
          <p className="flex items-baseline gap-1.5 text-white">
            <span className="text-[16px] font-semibold tabular" data-testid="hover-value">
              {r.value}
            </span>
            {r.unit && <span className="text-[12px] text-on-navy-2">{r.unit}</span>}
          </p>
          <p className="text-[12px] leading-snug text-white/85">{r.what}</p>
        </>
      ) : (
        <p className="text-[12px] text-white" data-testid="hover-none">
          {r.none}
        </p>
      )}
      <p className="truncate text-[11.5px]">
        <span aria-hidden className={`mr-1 inline-block h-1.5 w-1.5 -translate-y-px rounded-full ${r.kind === "model" ? "bg-[#b9a6f5]" : "bg-[#7fd4c1]"}`} />
        {r.when}
      </p>
      <p className="text-[11px] text-on-navy-2/70 tabular">
        {point.lat.toFixed(3)}°N {Math.abs(point.lon).toFixed(3)}°W · click for details
      </p>
    </div>
  );
}

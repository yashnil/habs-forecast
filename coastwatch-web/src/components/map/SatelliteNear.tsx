"use client";

import { useEffect, useState } from "react";
import type { Manifest } from "@/generated/schema";
import { CHLOROPHYLL_COPY } from "@/content/copy";
import { sampleChunked, type ChunkedSample } from "@/lib/grid";
import { nativeLabel, satelliteLatest, type SatProduct } from "@/lib/layers";
import { formatDate } from "@/lib/time";

/**
 * "Latest clear satellite view near here": the newest valid chlorophyll pixel at (or, if
 * cloud covers the point, within about 1 km of) a location, with the date that pixel was
 * observed. OLCI 300 m first; the VIIRS 750 m fallback only if OLCI is not published.
 */
export function SatelliteNear({ manifest, baseUrl, lat, lon, place }: { manifest: Manifest; baseUrl: string; lat: number; lon: number; place: string }) {
  const olci = satelliteLatest(manifest, "olci300");
  const layer = olci ?? satelliteLatest(manifest, "viirs750");
  const product: SatProduct | null = olci ? "olci300" : layer ? "viirs750" : null;
  const [s, setS] = useState<ChunkedSample | null | "error">(null);

  useEffect(() => {
    let cancelled = false;
    setS(null);
    if (!layer?.grid?.chunks) return;
    const search = product === "olci300" ? 4 : 2; // about 1-1.5 km either way
    sampleChunked(baseUrl, layer.grid, lat, lon, layer.composite?.age_grid ?? null, search)
      .then((r) => !cancelled && setS(r))
      .catch(() => !cancelled && setS("error"));
    return () => {
      cancelled = true;
    };
  }, [baseUrl, layer, lat, lon, product]);

  if (!layer) return null;
  const ref = layer.composite?.reference_date;
  const observed = s && s !== "error" && s.kind === "value" && ref && s.ageDays != null ? new Date(Date.parse(`${ref}T12:00:00Z`) - s.ageDays * 86400000).toISOString().slice(0, 10) : null;
  return (
    <section data-testid="satellite-near" className="space-y-1.5" aria-labelledby="sat-near-h">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="sat-near-h" className="text-[13px] font-semibold text-ink">
          Satellite chlorophyll, latest clear view
        </h3>
        <span className="text-[11px] text-ink-3">
          {product === "olci300" ? "Sentinel-3 OLCI" : "VIIRS"} · native {nativeLabel(layer)}
        </span>
      </div>
      {s === null ? (
        <p className="text-[12px] text-ink-3">Reading the satellite grid…</p>
      ) : s === "error" ? (
        <p className="text-[12px] text-ink-3">The satellite grid could not be read.</p>
      ) : s.kind === "none" ? (
        <p className="text-[12.5px] text-ink-2" data-testid="satellite-near-none">
          No clear observation near {place} in the last {layer.composite?.window_days ?? 7} days. Cloud or fog, not low chlorophyll.
        </p>
      ) : (
        <p className="text-[12.5px] text-ink-2">
          <span className="text-[20px] font-semibold text-ink tabular" data-testid="satellite-near-value">
            {s.value >= 10 ? s.value.toFixed(0) : s.value.toPrecision(2)}
          </span>{" "}
          mg/m³
          {observed && (
            <>
              {" "}
              · observed <span className="font-medium text-ink" data-testid="satellite-near-date">{formatDate(observed)}</span>
            </>
          )}
          {s.nearest && <span className="text-ink-3"> · nearest clear pixel {s.distanceKm.toFixed(1)} km away</span>}
        </p>
      )}
      <p className="text-[11.5px] text-ink-3">{CHLOROPHYLL_COPY.biomass}</p>
    </section>
  );
}

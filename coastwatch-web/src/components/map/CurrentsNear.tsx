"use client";

import { useEffect, useState } from "react";
import type { LayerArtifact } from "@/generated/schema";
import { compass, KNOTS_PER_MS, loadField, sampleField, type CurrentSample } from "@/lib/currents";
import { formatDateTimePT } from "@/lib/time";

/** Observed surface current at a point for the hour (or 24 h mean) shown on the map. */
export function CurrentsNear({ layer, baseUrl, lat, lon, place }: { layer: LayerArtifact | null; baseUrl: string; lat: number; lon: number; place: string }) {
  const [s, setS] = useState<CurrentSample | null | "none" | "error">(null);
  useEffect(() => {
    let cancelled = false;
    setS(null);
    if (!layer?.vectors) return;
    loadField(baseUrl, layer)
      .then((f) => !cancelled && setS(sampleField(f, lat, lon, 1) ?? "none"))
      .catch(() => !cancelled && setS("error"));
    return () => {
      cancelled = true;
    };
  }, [baseUrl, layer, lat, lon]);
  if (!layer?.vectors) return null;
  const mean = layer.layer_id.endsWith("mean24h");
  const ot = layer.time.observed_times ?? [];
  const when = mean
    ? `24-hour mean to ${formatDateTimePT(ot[ot.length - 1] ?? "")}`
    : `hour of ${formatDateTimePT(layer.time.valid_time ?? "")}`;
  return (
    <section data-testid="currents-near" className="space-y-1.5" aria-labelledby="cur-near-h">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="cur-near-h" className="text-[13px] font-semibold text-ink">
          Observed surface current
        </h3>
        <span className="text-[11px] text-ink-3">HF radar · native 2 km</span>
      </div>
      {s === null ? (
        <p className="text-[12px] text-ink-3">Reading the current field…</p>
      ) : s === "error" ? (
        <p className="text-[12px] text-ink-3">The current field could not be read.</p>
      ) : s === "none" ? (
        <p className="text-[12.5px] text-ink-2" data-testid="currents-near-none">
          No radar observation near {place} for this {mean ? "24-hour mean" : "hour"}. No data, not calm water.
        </p>
      ) : (
        <>
          <p className="text-[12.5px] text-ink-2">
            <span className="text-[20px] font-semibold text-ink tabular" data-testid="currents-near-speed">
              {s.speed.toFixed(2)}
            </span>{" "}
            m/s ({(s.speed * KNOTS_PER_MS).toFixed(1)} knots) toward{" "}
            <span className="font-medium text-ink" data-testid="currents-near-dir">
              {compass(s.dir)} ({Math.round(s.dir)}°)
            </span>
          </p>
          <p className="text-[12px] text-ink-3" data-testid="currents-near-time">
            Observed, {when}
            {s.distanceKm > 0 && <> · nearest radar cell {s.distanceKm.toFixed(1)} km away</>}
          </p>
        </>
      )}
      <p className="text-[11.5px] text-ink-3">Surface water movement observed in the past. Not a forecast, and not where a bloom will go.</p>
    </section>
  );
}

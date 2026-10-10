"use client";

import { useEffect, useState } from "react";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import { CHLOROPHYLL_COPY } from "@/content/copy";
import { sampleChunked, type ChunkedSample } from "@/lib/grid";
import { nativeLabel, satelliteLatest, type SatProduct } from "@/lib/layers";
import { formatDate } from "@/lib/time";
import { multiSensorLayer, multiSensorMembers, multiSensorPick, observedDate, type SensorObs } from "@/lib/multisensor";

/**
 * "Latest clear satellite view near here": the newest valid chlorophyll pixel at (or, if
 * cloud covers the point, within about 1 km of) a location, with the date that pixel was
 * observed. OLCI 300 m first; the VIIRS 750 m fallback only if OLCI is not published.
 */
export function SatelliteNear(props: { manifest: Manifest; baseUrl: string; lat: number; lon: number; place: string }) {
  return multiSensorLayer(props.manifest) ? <MultiSensorNear {...props} /> : <SingleSensorNear {...props} />;
}

function SingleSensorNear({ manifest, baseUrl, lat, lon, place }: { manifest: Manifest; baseUrl: string; lat: number; lon: number; place: string }) {
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

type Read = { obs: SensorObs; nearest: ChunkedSample | null };

async function readMember(baseUrl: string, l: LayerArtifact | null, lat: number, lon: number, search: number): Promise<Read> {
  if (!l?.grid?.chunks || !l.composite) return { obs: null, nearest: null };
  const here = await sampleChunked(baseUrl, l.grid, lat, lon, l.composite.age_grid, 0);
  if (here.kind === "value" && here.ageDays != null) return { obs: { value: here.value, date: observedDate(l.composite.reference_date, here.ageDays) }, nearest: null };
  const near = await sampleChunked(baseUrl, l.grid, lat, lon, l.composite.age_grid, search);
  return { obs: null, nearest: near.kind === "value" ? near : null };
}

const fmtV = (v: number) => (v >= 10 ? v.toFixed(0) : v.toPrecision(2));

/**
 * At a selected point: which sensor the multi-sensor map shows there, with that sensor's own
 * value, resolution and observation date, and what the other sensor has at the same place.
 * Values come from the members' published grids; the rule is the pipeline's.
 */
function MultiSensorNear({ manifest, baseUrl, lat, lon, place }: { manifest: Manifest; baseUrl: string; lat: number; lon: number; place: string }) {
  const layer = multiSensorLayer(manifest)!;
  const ms = layer.multisensor!;
  const [primary, secondary] = multiSensorMembers(manifest, layer);
  const labels = [...ms.members].sort((a, b) => a.order - b.order).map((m) => m.label);
  const [r, setR] = useState<[Read, Read] | null | "error">(null);

  useEffect(() => {
    let cancelled = false;
    setR(null);
    Promise.all([readMember(baseUrl, primary, lat, lon, 4), readMember(baseUrl, secondary, lat, lon, 2)])
      .then((x) => !cancelled && setR(x))
      .catch(() => !cancelled && setR("error"));
    return () => {
      cancelled = true;
    };
  }, [baseUrl, primary, secondary, lat, lon]);

  const pick = r && r !== "error" ? multiSensorPick(r[0].obs, r[1].obs, ms.prefer_primary_within_days) : null;
  const shown = r && r !== "error" && pick ? r[pick - 1].obs : null;
  const other = r && r !== "error" && pick ? r[2 - pick].obs : null;
  const nearest = r && r !== "error" && !pick ? (r[0].nearest ?? r[1].nearest) : null;
  return (
    <section data-testid="satellite-near" className="space-y-1.5" aria-labelledby="sat-near-h">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="sat-near-h" className="text-[13px] font-semibold text-ink">
          Satellite chlorophyll, latest view
        </h3>
        <span className="text-[11px] text-ink-3">multi-sensor display</span>
      </div>
      {r === null ? (
        <p className="text-[12px] text-ink-3">Reading the satellite grids…</p>
      ) : r === "error" ? (
        <p className="text-[12px] text-ink-3">The satellite grids could not be read.</p>
      ) : !pick || !shown ? (
        <p className="text-[12.5px] text-ink-2" data-testid="satellite-near-none">
          Neither sensor observed {place} in the last 7 days. Cloud or fog, not low chlorophyll.
          {nearest && nearest.kind === "value" && (
            <span className="text-ink-3"> Nearest clear pixel {nearest.distanceKm.toFixed(1)} km away: {fmtV(nearest.value)} mg/m³.</span>
          )}
        </p>
      ) : (
        <>
          <p className="text-[12.5px] text-ink-2">
            <span className="text-[20px] font-semibold text-ink tabular" data-testid="satellite-near-value">
              {fmtV(shown.value)}
            </span>{" "}
            mg/m³ · observed <span className="font-medium text-ink" data-testid="satellite-near-date">{formatDate(shown.date)}</span>
          </p>
          <p className="text-[12px] text-ink-2" data-testid="satellite-near-sensor">
            Shown: <span className="font-medium text-ink">{labels[pick - 1]}</span>
            {pick === 2 && r[0].obs && <> (Sentinel-3&apos;s pixel here is older: {formatDate(r[0].obs.date)})</>}
            {pick === 2 && !r[0].obs && <> (no Sentinel-3 observation here in 7 days)</>}
          </p>
          {other && (
            <p className="text-[12px] text-ink-3" data-testid="satellite-near-other">
              {labels[2 - pick]} here: {fmtV(other.value)} mg/m³ · observed {formatDate(other.date)}
            </p>
          )}
        </>
      )}
      <p className="text-[11.5px] text-ink-3">{CHLOROPHYLL_COPY.biomass}</p>
    </section>
  );
}

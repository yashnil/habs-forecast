"use client";

import { useEffect, useState } from "react";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import { CHL_NEAR_COPY, CHLOROPHYLL_COPY } from "@/content/copy";
import { CHL_PLACE_RADIUS_KM, chlStaleWord, fmtChlValue as fmtV, readChlNear, readPlaceChl, type ChlNear } from "@/lib/chlNear";
import { useNow } from "@/lib/useNow";
import { nativeLabel, satelliteLatest } from "@/lib/layers";
import { viirsPlatformLabel } from "@/lib/stamp";
import { formatDate } from "@/lib/time";
import { multiSensorLayer, multiSensorMembers, multiSensorPick, type SensorObs } from "@/lib/multisensor";

/**
 * "Latest clear satellite view near here": the newest valid chlorophyll pixel at a location or,
 * if that pixel is missing (cloud, land) or set aside by the quality rules (lib/chlNear), the
 * median of the clear water pixels within CHL_PLACE_RADIUS_KM, with their count, range and date.
 * The same reading (lib/chlNear readPlaceChl) as My Coast's briefing: one place, one number.
 */
export const NEAR_RADIUS_KM = CHL_PLACE_RADIUS_KM;

/** What the component found, for a summary elsewhere on the page (one rule, one answer):
 *  the date of the headline value, and how far its nearest pixel is when it is not at the point. */
export type SatelliteResult = { date: string; nearestKm: number | null } | { date: null; nearestKm: null; flagged: boolean } | null;
type NearProps = { manifest: Manifest; baseUrl: string; lat: number; lon: number; place: string; onRead?: (r: SatelliteResult) => void };

export function SatelliteNear(props: NearProps) {
  return multiSensorLayer(props.manifest) ? <MultiSensorNear {...props} /> : <SingleSensorNear {...props} />;
}

/** The map's freshness word for the reading's own date (nothing when current). */
function StaleWord({ layer, date }: { layer: LayerArtifact | null; date: string | null }) {
  const now = useNow();
  const word = chlStaleWord(layer, date, now);
  return word ? (
    <span className="ml-1 font-medium text-[#9b2c2c]" data-testid="satellite-near-stale">
      · {word}
    </span>
  ) : null;
}

const result = (n: ChlNear | null): SatelliteResult =>
  n && (n.kind === "at" || n.kind === "median") && n.date ? { date: n.date, nearestKm: n.kind === "median" ? n.nearest.km : null } : { date: null, nearestKm: null, flagged: n?.kind === "flagged" };

/** How a neighbourhood headline was made, and pixels set aside next to it. */
function Basis({ n }: { n: ChlNear }) {
  return (
    <>
      {n.kind === "median" && (
        <p className="text-[13px] text-ink-3" data-testid="satellite-near-basis">
          {CHL_NEAR_COPY.basis({ n: n.n, radiusKm: n.radiusKm, min: n.min, max: n.max, nearestKm: n.nearest.km })}
        </p>
      )}
      {(n.kind === "at" || n.kind === "median") && n.aside && (
        <p className="text-[13px] text-ink-2" data-testid="satellite-near-aside">
          {CHL_NEAR_COPY.aside(n.aside)}
        </p>
      )}
      {!n.water && <p className="text-[13px] text-ink-3">{CHL_NEAR_COPY.noCoastline}</p>}
    </>
  );
}

function Flagged({ place, n }: { place: string; n: ChlNear }) {
  return (
    <>
      <p className="text-[13px] text-ink-2" data-testid="satellite-near-flagged">
        {CHL_NEAR_COPY.flagged(place, n.radiusKm)}
      </p>
      <Basis n={n} />
    </>
  );
}

function SingleSensorNear({ manifest, baseUrl, lat, lon, place, onRead }: NearProps) {
  const [layer, setLayer] = useState<LayerArtifact | null>(null);
  const [s, setS] = useState<ChlNear | null | "error">(null);

  useEffect(() => {
    let cancelled = false;
    setS(null);
    readPlaceChl(manifest, baseUrl, lat, lon)
      .then((r) => {
        if (cancelled || !r) return;
        setLayer(r.layer);
        setS(r.read);
      })
      .catch(() => !cancelled && setS("error"));
    return () => {
      cancelled = true;
    };
  }, [manifest, baseUrl, lat, lon]);
  const product = layer?.layer_id.startsWith("olci300") ? "olci300" : "viirs750";

  useEffect(() => {
    if (!onRead || s === null) return;
    if (s === "error") return onRead({ date: null, nearestKm: null, flagged: false });
    // a value without a date is not summarised elsewhere (as before)
    if ((s.kind === "at" || s.kind === "median") && !s.date) return;
    onRead(result(s));
  }, [s, onRead]);
  if (!satelliteLatest(manifest, "olci300") && !satelliteLatest(manifest, "viirs750")) return null;
  return (
    <section data-testid="satellite-near" className="space-y-1.5" aria-labelledby="sat-near-h">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="sat-near-h" className="text-[13px] font-semibold text-ink">
          Satellite chlorophyll, latest clear view
        </h3>
        <span className="text-[13px] text-ink-3">
          {layer && (
            <>
              {product === "olci300" ? "Sentinel-3 OLCI" : viirsPlatformLabel(layer)} · native {nativeLabel(layer)}
            </>
          )}
        </span>
      </div>
      {s === null ? (
        <p className="text-[13px] text-ink-3">Reading the satellite grid…</p>
      ) : s === "error" ? (
        <p className="text-[13px] text-ink-3">The satellite grid could not be read.</p>
      ) : s.kind === "none" ? (
        <p className="text-[13px] text-ink-2" data-testid="satellite-near-none">
          No clear observation near {place} in the last {layer?.composite?.window_days ?? 7} days. Cloud or fog, not low chlorophyll.
        </p>
      ) : s.kind === "flagged" ? (
        <Flagged place={place} n={s} />
      ) : (
        <>
          <p className="text-[13px] text-ink-2">
            <span className="text-[20px] font-semibold text-ink tabular" data-testid="satellite-near-value">
              {fmtV(s.value)}
            </span>{" "}
            mg/m³
            {s.date && (
              <>
                {" "}
                · {s.kind === "median" ? "newest pixel observed" : "observed"} <span className="font-medium text-ink" data-testid="satellite-near-date">{formatDate(s.date)}</span>
                <StaleWord layer={layer} date={s.date} />
              </>
            )}
          </p>
          <Basis n={s} />
        </>
      )}
      <p className="text-[13px] text-ink-3">{CHLOROPHYLL_COPY.biomass}</p>
    </section>
  );
}

async function readMember(baseUrl: string, l: LayerArtifact | null, lat: number, lon: number): Promise<ChlNear | null> {
  if (!l?.grid?.chunks || !l.composite) return null;
  return readChlNear(baseUrl, l, lat, lon, { radiusKm: NEAR_RADIUS_KM });
}

const atObs = (n: ChlNear | null): SensorObs => (n?.kind === "at" && n.date ? { value: n.value, date: n.date } : null);
const nearObs = (n: ChlNear | null): SensorObs => (n?.kind === "median" && n.date ? { value: n.value, date: n.date } : null);

/**
 * At a selected point: which sensor the multi-sensor map shows there, with that sensor's own
 * value, resolution and observation date, and what the other sensor has at the same place.
 * Values come from the members' published grids; the rule is the pipeline's. Where neither
 * sensor has a usable pixel at the point, the same rule picks between the sensors' medians of
 * the clear water pixels nearby.
 */
function MultiSensorNear({ manifest, baseUrl, lat, lon, place, onRead }: NearProps) {
  const layer = multiSensorLayer(manifest)!;
  const ms = layer.multisensor!;
  const [primary, secondary] = multiSensorMembers(manifest, layer);
  const labels = [...ms.members].sort((a, b) => a.order - b.order).map((m) => m.label);
  const [r, setR] = useState<[ChlNear | null, ChlNear | null] | null | "error">(null);

  useEffect(() => {
    let cancelled = false;
    setR(null);
    Promise.all([readMember(baseUrl, primary, lat, lon), readMember(baseUrl, secondary, lat, lon)])
      .then((x) => !cancelled && setR(x))
      .catch(() => !cancelled && setR("error"));
    return () => {
      cancelled = true;
    };
  }, [baseUrl, primary, secondary, lat, lon]);

  const ok = r && r !== "error" ? r : null;
  const pick = ok ? multiSensorPick(atObs(ok[0]), atObs(ok[1]), ms.prefer_primary_within_days) : 0;
  const pickNear = ok && !pick ? multiSensorPick(nearObs(ok[0]), nearObs(ok[1]), ms.prefer_primary_within_days) : 0;
  const which = pick || pickNear;
  const shown = ok && which ? ok[which - 1] : null;
  const other = ok && pick ? atObs(ok[2 - pick]) : null;
  const flagged = ok ? (ok.find((n) => n?.kind === "flagged") ?? null) : null;
  useEffect(() => {
    if (!onRead || r === null) return;
    if (r === "error") onRead({ date: null, nearestKm: null, flagged: false });
    else onRead(shown ? result(shown) : { date: null, nearestKm: null, flagged: !!flagged });
  }, [r, shown, flagged, onRead]);
  return (
    <section data-testid="satellite-near" className="space-y-1.5" aria-labelledby="sat-near-h">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="sat-near-h" className="text-[13px] font-semibold text-ink">
          Satellite chlorophyll, latest view
        </h3>
        <span className="text-[13px] text-ink-3">multi-sensor display</span>
      </div>
      {r === null ? (
        <p className="text-[13px] text-ink-3">Reading the satellite grids…</p>
      ) : r === "error" ? (
        <p className="text-[13px] text-ink-3">The satellite grids could not be read.</p>
      ) : !shown || (shown.kind !== "at" && shown.kind !== "median") ? (
        flagged ? (
          <Flagged place={place} n={flagged} />
        ) : (
          <p className="text-[13px] text-ink-2" data-testid="satellite-near-none">
            Neither sensor observed {place} in the last 7 days. Cloud or fog, not low chlorophyll.
          </p>
        )
      ) : (
        <>
          <p className="text-[13px] text-ink-2">
            <span className="text-[20px] font-semibold text-ink tabular" data-testid="satellite-near-value">
              {fmtV(shown.value)}
            </span>{" "}
            mg/m³ · {shown.kind === "median" ? "newest pixel observed" : "observed"} <span className="font-medium text-ink" data-testid="satellite-near-date">{shown.date ? formatDate(shown.date) : "—"}</span>
            <StaleWord layer={ok && which ? [primary, secondary][which - 1] : null} date={shown.date} />
          </p>
          <p className="text-[13px] text-ink-2" data-testid="satellite-near-sensor">
            Shown: <span className="font-medium text-ink">{labels[which - 1]}</span>
            {pick === 2 && atObs(ok![0]) && <> (Sentinel-3&apos;s pixel here is older: {formatDate(atObs(ok![0])!.date)})</>}
            {pick === 2 && !atObs(ok![0]) && (ok![0]?.atPoint === "set-aside" ? <> (Sentinel-3&apos;s pixel here is not used: on land, right against the shore, or far out of line with the water around it)</> : <> (no Sentinel-3 observation here in 7 days)</>)}
          </p>
          {other && (
            <p className="text-[13px] text-ink-3" data-testid="satellite-near-other">
              {labels[2 - pick]} here: {fmtV(other.value)} mg/m³ · observed {formatDate(other.date)}
            </p>
          )}
          <Basis n={shown} />
        </>
      )}
      <p className="text-[13px] text-ink-3">{CHLOROPHYLL_COPY.biomass}</p>
    </section>
  );
}

"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import type { Manifest } from "@/generated/schema";
import type { PortIntel } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { CHARM_LEADS, artifactUrl, charmLayer } from "@/lib/layers";
import { charmDays, roleWords, runNotes } from "@/lib/charmDays";
import { loadGrid } from "@/lib/grid";
import { CHL_PLACE_RADIUS_KM, chlStaleWord, fmtChlValue, readPlaceChl, type ChlAside } from "@/lib/chlNear";
import { CHL_NEAR_COPY } from "@/content/copy";
import { formatDate } from "@/lib/time";
import { Mark } from "@/components/ui/Mark";
import { ageWords, fmtMeasured, fmtPct, measuredEvidence, neighbourhood, stationsAround, EVIDENCE_RADIUS_KM, NEIGHBOURHOOD_KM, RECENT_DAYS, type Neighbourhood, type StationEvidence } from "@/lib/hab";
export type { Neighbourhood };

/**
 * "HAB conditions" for chosen waters: the latest evidence of bloom activity around them, each piece
 * with its value, date, distance and source, and NOAA's model outlook for the nearby cells. Selecting
 * a piece of evidence shows its source on the map (onFocus). Observed, measured and modelled stay
 * apart; nothing here is a risk score, a seafood-safety statement or a fishing prediction.
 */
export type Focus = "toxin" | "cells" | "chl" | "satellite" | "model" | "official" | "conditions" | null;

// ---------------------------------------------------------------- data hooks

export type OutlookDay = { lead: number; valid: string | null; pda: Neighbourhood | null; pn: Neighbourhood | null };

/** C-HARM for the waters, every issued day: the waters' own cell and the real cells around them. */
export function useOutlook(manifest: Manifest, baseUrl: string, lat: number | null, lon: number | null): OutlookDay[] | null | "error" {
  const [r, setR] = useState<OutlookDay[] | null | "error">(null);
  useEffect(() => {
    let cancelled = false;
    setR(null);
    if (lat == null || lon == null) return;
    Promise.all(
      CHARM_LEADS.map(async (lead): Promise<OutlookDay> => {
        const a = charmLayer(manifest, "particulate_domoic", lead);
        const b = charmLayer(manifest, "pseudo_nitzschia", lead);
        const [ca, cb] = await Promise.all([a?.grid ? loadGrid(artifactUrl(baseUrl, a.grid.url)) : null, b?.grid ? loadGrid(artifactUrl(baseUrl, b.grid.url)) : null]);
        return { lead, valid: (a ?? b)?.time.valid_date ?? null, pda: a?.grid && ca ? neighbourhood(a.grid, ca, lat, lon) : null, pn: b?.grid && cb ? neighbourhood(b.grid, cb, lat, lon) : null };
      }),
    )
      .then((x) => !cancelled && setR(x.filter((d) => d.valid)))
      .catch(() => !cancelled && setR("error"));
    return () => {
      cancelled = true;
    };
  }, [manifest, baseUrl, lat, lon]);
  return r;
}

export type SatReading =
  | {
      value: number;
      date: string | null;
      /** 0 when the value is the pixel at the waters; otherwise the distance to the nearest pixel used */
      km: number;
      sensor: string;
      /** the pixel at the waters, or the nearest pixel used */
      pixelLat: number;
      pixelLon: number;
      layerId: string;
      /** set when the value is a median of the clear water pixels nearby */
      basis: { n: number; min: number; max: number; radiusKm: number } | null;
      aside: ChlAside | null;
      /** whether pixels on land and against the shore could be set aside */
      water: boolean;
    }
  | { value: null; sensor: string; windowDays: number; flagged: boolean; radiusKm: number; water: boolean }
  | null;

export const SAT_RADIUS_KM = CHL_PLACE_RADIUS_KM;

/** The place's satellite chlorophyll: the same reading as the Science tab and the map (lib/chlNear readPlaceChl). */
export function useSatellite(manifest: Manifest, baseUrl: string, lat: number | null, lon: number | null): SatReading | "error" {
  const [r, setR] = useState<SatReading | "error">(null);
  useEffect(() => {
    let cancelled = false;
    setR(null);
    if (lat == null || lon == null) return;
    readPlaceChl(manifest, baseUrl, lat, lon)
      .then((pc) => {
        if (cancelled || !pc) return;
        const { layer, read: s } = pc;
        const sensor = layer.layer_id.startsWith("olci300") ? "Sentinel-3 OLCI, 300 m" : "VIIRS, 750 m";
        if (s.kind === "none" || s.kind === "flagged") return setR({ value: null, sensor, windowDays: layer.composite?.window_days ?? 7, flagged: s.kind === "flagged", radiusKm: s.radiusKm, water: s.water });
        const px = s.kind === "at" ? s.pixel : s.nearest;
        setR({
          value: s.value,
          date: s.date,
          km: s.kind === "at" ? 0 : s.nearest.km,
          sensor,
          pixelLat: px.lat,
          pixelLon: px.lon,
          layerId: layer.layer_id,
          basis: s.kind === "median" ? { n: s.n, min: s.min, max: s.max, radiusKm: s.radiusKm } : null,
          aside: s.aside,
          water: s.water,
        });
      })
      .catch(() => !cancelled && setR("error"));
    return () => {
      cancelled = true;
    };
  }, [manifest, baseUrl, lat, lon]);
  return r;
}

// ---------------------------------------------------------------- small visuals

/** A year of samples at one station: dots on a log axis, reported zeros on their own floor, the newest ringed. */
function Timeline({ series, now, color = "var(--cw-chl)", testid }: { series: [string, number | null][]; now: Date; color?: string; testid?: string }) {
  const W = 220;
  const H = 46;
  const t1 = now.getTime();
  const t0 = t1 - 365 * 86400000;
  const pts = series.filter(([d]) => Date.parse(d) >= t0);
  const pos = pts.filter(([, v]) => v != null && v > 0).map(([, v]) => v as number);
  const lo = pos.length ? Math.min(...pos) : 1;
  const hi = pos.length ? Math.max(...pos) : 10;
  const l0 = Math.log10(lo) - 0.1;
  const l1 = Math.log10(hi) + 0.1;
  const x = (d: string) => 4 + ((Date.parse(d) - t0) / (t1 - t0)) * (W - 8);
  const y = (v: number) => (v <= 0 ? H - 4 : 6 + (1 - (Math.log10(v) - l0) / Math.max(0.2, l1 - l0)) * (H - 18));
  const last = pts.at(-1);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 block h-[46px] w-full" role="img" aria-label={`${pts.length} samples in the last 12 months`} data-testid={testid}>
      <line x1="4" x2={W - 4} y1={H - 4} y2={H - 4} stroke="var(--cw-hairline-strong)" strokeDasharray="2 3" />
      {pts.map(([d, v], i) =>
        v != null && v > 0 ? (
          <motion.circle key={d + i} cx={x(d)} cy={y(v)} r={2.4} fill={color} initial={{ opacity: 0 }} animate={{ opacity: 0.85 }} transition={{ delay: i * 0.01 }} />
        ) : (
          <circle key={d + i} cx={x(d)} cy={H - 4} r={1.8} fill="none" stroke={color} strokeOpacity={0.55} />
        ),
      )}
      {last && last[1] != null && <circle cx={x(last[0])} cy={y(last[1])} r={5} fill="none" stroke="var(--cw-ink)" strokeWidth={1.3} />}
    </svg>
  );
}

/**
 * Four days of C-HARM for the waters: the waters' own cell as a solid bar when it has a value;
 * otherwise the offshore cells' median as an outlined, hatched bar with the range as a whisker and
 * the word "offshore" under it, so a probability that is not for these waters is never drawn as
 * if it were. Days are named by their date and distance from today (lib/charmDays).
 */
function OutlookBars({ days, pick, active, now }: { days: OutlookDay[]; pick: "pda" | "pn"; active: number; now: Date | null }) {
  const chips = charmDays(
    [0, 1, 2, 3].map((l) => {
      const d = days.find((x) => x.lead === l);
      return d?.valid ? { time: { valid_date: d.valid } } : null;
    }),
    now,
  );
  return (
    <div className="mt-2 grid grid-cols-4 gap-2" data-testid={`outlook-${pick}`}>
      {days.map((d) => {
        const nb = d[pick];
        const own = nb?.here != null;
        const v = nb?.here ?? nb?.median ?? null;
        const chip = chips[d.lead];
        return (
          <div key={d.lead} className={`rounded-lg px-1.5 pb-1.5 pt-1 ${d.lead === active ? "bg-model-bg" : ""}`} data-testid={`outlook-${pick}-${d.lead}`} data-basis={own ? "own-cell" : v != null ? "offshore-median" : "none"}>
            <div className="relative h-14">
              {nb && nb.min != null && nb.max != null && !own && (
                <span aria-hidden className="absolute left-1/2 w-px -translate-x-1/2 bg-model/50" style={{ bottom: `${nb.min * 100}%`, height: `${(nb.max - nb.min) * 100}%` }} />
              )}
              {v != null && (
                <motion.span
                  aria-hidden
                  className={`absolute bottom-0 left-1/2 w-3.5 -translate-x-1/2 rounded-t-sm ${own ? "bg-model" : "cw-hatch-model border border-model/70"}`}
                  initial={{ height: 0 }}
                  animate={{ height: `${v * 100}%` }}
                  transition={{ type: "spring", stiffness: 160, damping: 22, delay: d.lead * 0.06 }}
                />
              )}
            </div>
            <p className={`mt-1 text-center text-[13px] tabular ${own ? "font-semibold text-ink" : "text-ink-2"}`}>
              {v != null && !own ? "~" : ""}
              {fmtPct(v)}
            </p>
            <p className="text-center text-[12px] leading-tight text-ink-3">{own ? "own cell" : v != null ? "offshore" : "no value"}</p>
            <p className="text-center text-[12px] leading-tight text-ink-3">{chip.short ?? "—"}</p>
            <p className="text-center text-[12px] leading-tight text-ink-3">{chip.relative ?? roleWords(d.lead)}</p>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- the section

const KIND_LABEL: Record<"toxin" | "cells" | "chl", string> = {
  toxin: "Toxin in seawater · measured",
  cells: "Pseudo-nitzschia cells · measured",
  chl: "Chlorophyll in the water · measured",
};

function EvidenceCard({ ev, kind, focus, onFocus, now }: { ev: StationEvidence | null; kind: "toxin" | "cells"; focus: Focus; onFocus: (f: Focus) => void; now: Date }) {
  const on = focus === kind;
  if (!ev) {
    return (
      <div className="rounded-2xl border border-dashed border-hairline-strong p-4" data-testid={`ev-${kind}`}>
        <p className="cw-kind text-ink-3">
          <Mark kind="gap" />
          {KIND_LABEL[kind]}
        </p>
        <p className="mt-2 text-[15px] text-ink-2">No shore station measures this near here.</p>
      </div>
    );
  }
  const recent = ev.near && ev.ageDays <= RECENT_DAYS;
  const main = ev.values[0];
  return (
    <motion.button
      type="button"
      onClick={() => onFocus(on ? null : kind)}
      aria-pressed={on}
      whileTap={{ scale: 0.985 }}
      className={`group w-full rounded-2xl border p-4 text-left transition-colors ${on ? "border-measured bg-measured-bg/60 shadow-[0_0_0_1px_var(--cw-chl)]" : "border-hairline bg-surface hover:border-hairline-strong"}`}
      data-testid={`ev-${kind}`}
      data-recent={recent || undefined}
    >
      <p className="cw-kind text-measured">
        <Mark kind="measured" aged={!recent} />
        {KIND_LABEL[kind]}
      </p>
      <p className="mt-1.5 text-[22px] font-semibold leading-tight tabular text-ink">{fmtMeasured(main.variable, main.value)}</p>
      {kind === "cells" && ev.values.length > 1 && <p className="text-[13px] text-ink-2">{ev.values.map((v) => `${v.variable === "pn_seriata" ? "larger cells" : "smaller cells"} ${fmtMeasured(v.variable, v.value)}`).join(" · ")}</p>}
      <p className={`mt-1 text-[14px] ${recent ? "text-ink-2" : "font-medium text-ink"}`}>
        {formatDate(ev.date, { year: ev.ageDays > 300 })} · {ageWords(ev.ageDays)}
      </p>
      <p className="text-[13px] text-ink-3">
        {ev.station.name}, {ev.km < 10 ? ev.km.toFixed(1) : Math.round(ev.km)} km {ev.bearing}
        {!ev.near && " (too far to describe these waters)"}
      </p>
      {Object.values(ev.series)[0] && (
        <>
          <Timeline series={Object.values(ev.series)[0]} now={now} testid={`ev-${kind}-timeline`} />
          <p className="flex justify-between text-[11.5px] text-ink-3">
            <span>12 months ago</span>
            <span>{kind === "cells" ? "○ none counted" : "○ not quantified"} · ◎ latest</span>
          </p>
        </>
      )}
      <p className="mt-1.5 text-[13px] font-medium text-accent group-hover:underline">{on ? "Shown on the map" : "Show the station on the map"}</p>
    </motion.button>
  );
}

function HistoryLine({ points }: { points: { date: string; value: number | null }[] }) {
  const W = 220;
  const H = 40;
  const pts = points.filter((p) => p.value != null) as { date: string; value: number }[];
  if (pts.length < 2) return null;
  const t0 = Date.parse(pts[0].date);
  const t1 = Date.parse(pts[pts.length - 1].date);
  const x = (d: string) => 4 + ((Date.parse(d) - t0) / Math.max(1, t1 - t0)) * (W - 8);
  const y = (v: number) => 4 + (1 - v) * (H - 8);
  // a line joins only consecutive daily runs; a missing day breaks it (no invented values)
  const segs: { date: string; value: number }[][] = [];
  for (const p of pts) {
    const last = segs.at(-1)?.at(-1);
    if (last && Date.parse(p.date) - Date.parse(last.date) <= 1.5 * 86400000) segs.at(-1)!.push(p);
    else segs.push([p]);
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-10 w-full" role="img" aria-label={`Nowcast median over the last ${pts.length} runs`} data-testid="model-history">
      <line x1="4" x2={W - 4} y1={y(0.5)} y2={y(0.5)} stroke="var(--cw-hairline)" />
      {segs.map((sg, i) =>
        sg.length > 1 ? (
          <motion.path key={i} d={sg.map((p, k) => `${k ? "L" : "M"}${x(p.date)},${y(p.value)}`).join("")} fill="none" stroke="var(--cw-forecast)" strokeWidth={1.6} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.9, ease: [0.2, 0, 0, 1] }} />
        ) : (
          <circle key={i} cx={x(sg[0].date)} cy={y(sg[0].value)} r={1.8} fill="var(--cw-forecast)" />
        ),
      )}
    </svg>
  );
}

export function HabBriefing({
  manifest,
  placeName,
  stations,
  outlook,
  sat,
  port,
  lead,
  focus,
  onFocus,
  now,
  lat,
  lon,
  between,
  pick,
  onPick,
  modelStale = null,
}: {
  /** the newest C-HARM run is stale or historical by its own policy */
  modelStale?: { state: "stale" | "historical"; issued: string | null; ageDays: number } | null;
  /** the model quantity shown (and drawn on the map) */
  pick: "pda" | "pn";
  onPick: (p: "pda" | "pn") => void;
  manifest: Manifest;
  placeName: string;
  stations: StationLite[] | null;
  outlook: OutlookDay[] | null | "error";
  sat: SatReading | "error";
  port: PortIntel | null;
  lead: number;
  focus: Focus;
  onFocus: (f: Focus) => void;
  now: Date | null;
  lat: number;
  lon: number;
  /** shown between the summary and the evidence cards (the phone's map) */
  between?: React.ReactNode;
}) {
  const ev = useHabEvidence(stations, lat, lon, now);
  const days = outlook && outlook !== "error" ? outlook : [];
  const day = days.find((d) => d.lead === lead) ?? days[0] ?? null;
  const tox = day?.pda ?? null;
  const thrPda = charmLayer(manifest, "particulate_domoic", lead)?.threshold_text ?? null;
  const thrPn = charmLayer(manifest, "pseudo_nitzschia", lead)?.threshold_text ?? null;
  const forDay = day?.valid ? `for ${formatDate(day.valid)}` : "";

  // the summary: facts only, in the order a reader needs them
  const summary: string[] = [];
  if (ev) {
    const c = ev.cells;
    const t = ev.toxin;
    const recentC = c && c.near && c.ageDays <= RECENT_DAYS;
    const recentT = t && t.near && t.ageDays <= RECENT_DAYS;
    if (recentC) {
      const v = c!.values.reduce((a, x) => a + x.value, 0);
      summary.push(v > 0 ? `Pseudo-nitzschia, the algae that can produce domoic acid, was counted at ${c!.station.name} ${ageWords(c!.ageDays)} (${fmtMeasured("pn_seriata", v)}).` : `No Pseudo-nitzschia was counted at ${c!.station.name} ${ageWords(c!.ageDays)}.`);
    }
    if (recentT) {
      const v = t!.values[0].value;
      summary.push(v > 0 ? `Domoic acid was measured in seawater at ${t!.station.name} ${recentC && t!.station.station_id === c!.station.station_id ? "in that sample" : ageWords(t!.ageDays)}: ${fmtMeasured("pDA", v)}.` : `No domoic acid was quantified in seawater at ${t!.station.name} ${ageWords(t!.ageDays)}.`);
    }
    // the nearest station, when the evidence above came from farther away: its own last sample, with its age
    const used = [recentC ? c : null, recentT ? t : null].filter(Boolean).map((e) => e!.station.station_id);
    if (used.length && ev.nearest && !used.includes(ev.nearest.station.station_id) && ev.nearest.km < Math.min(...[c, t].filter(Boolean).map((e) => e!.km)))
      summary.push(`The nearest station, ${ev.nearest.station.name} (${ev.nearest.km < 10 ? ev.nearest.km.toFixed(1) : Math.round(ev.nearest.km)} km), last reported Pseudo-nitzschia or toxin ${ageWords(ev.nearest.ageDays)}.`);
    if (!recentC && !recentT) {
      const older = [c, t].filter(Boolean).sort((a, b) => a!.km - b!.km)[0];
      const nearOld = [c, t].filter((e) => e && e.near).sort((a, b) => a!.ageDays - b!.ageDays)[0];
      summary.push(
        nearOld
          ? `No shore station within ${EVIDENCE_RADIUS_KM} km has sampled in the last ${RECENT_DAYS} days; the latest sample nearby was ${ageWords(nearOld.ageDays)}, at ${nearOld.station.name} (${Math.round(nearOld.km)} km).`
          : older
            ? `No shore station within ${EVIDENCE_RADIUS_KM} km measures bloom algae or toxin; the nearest that does, ${older.station.name}, is ${Math.round(older.km)} km away, too far to describe these waters.`
            : "No shore station measures bloom algae or toxin near these waters.",
      );
    }
  }
  if (tox?.here != null) summary.push(`NOAA's model gives these waters' 3 km cell a ${fmtPct(tox.here)} chance that particulate domoic acid exceeds its threshold ${forDay}.`);
  else if (tox && tox.n > 0) summary.push(`NOAA's model has no value for these nearshore waters; its ${tox.n} offshore cells within ${NEIGHBOURHOOD_KM} km have a median ${fmtPct(tox.median)} chance that particulate domoic acid exceeds its threshold ${forDay}.`);
  else if (tox?.nearest) summary.push(`These waters are outside NOAA's model: it has no cells within ${NEIGHBOURHOOD_KM} km. Its nearest cell, ${tox.nearest.km.toFixed(0)} km away, has ${fmtPct(tox.nearest.value)}; that describes that cell, not these waters.`);
  else if (tox) summary.push("These waters are outside NOAA's model: it has no cell within 80 km.");

  const hist = port?.charm?.history?.[pick === "pda" ? "particulate_domoic" : "pseudo_nitzschia"] ?? null;
  const satOn = focus === "satellite";
  const modelOn = focus === "model";
  return (
    <section aria-labelledby="hab-briefing-h" data-testid="hab-briefing" className="space-y-4">
      <div>
        <p className="cw-kind text-model-ink">
          <Mark kind="model" />
          HAB conditions · measured, observed and modelled
        </p>
        <h2 id="hab-briefing-h" className="mt-1 font-display text-[26px] font-medium leading-tight text-ink">
          Bloom and toxin evidence near {placeName}
        </h2>
        <p className="mt-2 max-w-[64ch] text-[16px] leading-relaxed text-ink-2" data-testid="hab-summary" aria-live="polite">
          {ev ? summary.join(" ") : "Reading the measurements, the model and the satellite…"}
        </p>
      </div>
      {between}

      <div className="grid gap-3 sm:grid-cols-2">
        {now && ev ? (
          <>
            <EvidenceCard ev={ev.cells} kind="cells" focus={focus} onFocus={onFocus} now={now} />
            <EvidenceCard ev={ev.toxin} kind="toxin" focus={focus} onFocus={onFocus} now={now} />
          </>
        ) : (
          <>
            <div className="h-[200px] rounded-2xl bg-surface-3" aria-hidden />
            <div className="h-[200px] rounded-2xl bg-surface-3" aria-hidden />
          </>
        )}

        <motion.button
          type="button"
          onClick={() => onFocus(satOn ? null : "satellite")}
          aria-pressed={satOn}
          whileTap={{ scale: 0.985 }}
          className={`group w-full rounded-2xl border p-4 text-left transition-colors sm:col-span-2 ${satOn ? "border-measured bg-measured-bg/60 shadow-[0_0_0_1px_var(--cw-chl)]" : "border-hairline bg-surface hover:border-hairline-strong"}`}
          data-testid="ev-satellite"
        >
          <p className="cw-kind text-measured">
            <Mark kind="satellite" />
            Chlorophyll from space · observed
          </p>
          {sat === null ? (
            <p className="mt-2 h-7 w-40 rounded-md bg-surface-3" aria-hidden />
          ) : sat === "error" ? (
            <p className="mt-2 text-[15px] text-ink-2">The satellite grid could not be read.</p>
          ) : sat.value == null ? (
            sat.flagged ? (
              <p className="mt-2 text-[15px] text-ink-2" data-testid="ev-satellite-flagged">
                {CHL_NEAR_COPY.flagged("these waters", sat.radiusKm)}
              </p>
            ) : (
              <p className="mt-2 text-[15px] text-ink-2">No clear view within {sat.radiusKm} km in the last {sat.windowDays} days: cloud, fog or no overpass, not low chlorophyll.</p>
            )
          ) : (
            <>
              <p className="mt-1.5 text-[22px] font-semibold leading-tight tabular text-ink" data-testid="ev-satellite-value">
                {fmtChlValue(sat.value)} mg/m³
              </p>
              <p className="mt-1 text-[14px] text-ink-2">
                {(() => {
                  const word = chlStaleWord(manifest.layers.find((l) => l.layer_id === sat.layerId), sat.date, now);
                  return word ? (
                    <span className="mr-1 font-medium text-[#9b2c2c]" data-testid="ev-satellite-stale">
                      {word}:
                    </span>
                  ) : null;
                })()}
                {sat.date ? `${sat.basis ? "Newest pixel " : ""}${formatDate(sat.date)} · ${ageWords(Math.floor(((now?.getTime() ?? Date.now()) - Date.parse(`${sat.date}T12:00:00Z`)) / 86400000))}` : "latest view"}
              </p>
              <p className="text-[13px] text-ink-3" data-testid="ev-satellite-basis">
                {sat.basis ? CHL_NEAR_COPY.basis({ ...sat.basis, nearestKm: sat.km }) : "At these waters"} · {sat.sensor}
              </p>
              {sat.aside && (
                <p className="mt-1 text-[13px] text-ink-2" data-testid="ev-satellite-aside">
                  {CHL_NEAR_COPY.aside(sat.aside)}
                </p>
              )}
            </>
          )}
          {sat && sat !== "error" && !sat.water && <p className="mt-1 text-[13px] text-ink-3">{CHL_NEAR_COPY.noCoastline}</p>}
          <p className="mt-2 text-[13px] text-ink-2">Algae of every kind, not toxin: high chlorophyll does not confirm a harmful bloom, and low does not rule one out.</p>
          <p className="mt-1.5 text-[13px] font-medium text-accent group-hover:underline">{satOn ? "Shown on the map" : "Show the satellite view"}</p>
        </motion.button>

        <div className={`rounded-2xl border p-4 transition-colors sm:col-span-2 ${modelOn ? "border-model bg-model-bg/50 shadow-[0_0_0_1px_var(--cw-forecast)]" : "border-hairline bg-surface"}`} data-testid="ev-model">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <p className="cw-kind text-model-ink">
              <Mark kind="model" />
              NOAA C-HARM model · a probability, not a measurement
            </p>
            {modelStale &&
              runNotes({ state: modelStale.state, basisDate: modelStale.issued, ageDays: modelStale.ageDays }, null).map((n) => (
                <p key={n.testid} className={`basis-full text-[14px] ${n.tone === "problem" ? "font-medium text-serious" : "text-ink-2"}`} data-testid="ev-model-stale" data-tone={n.tone}>
                  <span className="font-medium text-ink">{n.head}.</span> {n.text}
                </p>
              ))}
            <div role="radiogroup" aria-label="Model quantity" className="flex rounded-xl bg-surface-3 p-0.5 text-[13px]">
              {(
                [
                  ["pda", "Toxin"],
                  ["pn", "Bloom"],
                ] as const
              ).map(([id, label]) => (
                <button key={id} type="button" role="radio" aria-checked={pick === id} onClick={() => onPick(id)} data-testid={`outlook-pick-${id}`} className={`cw-press relative rounded-[10px] px-3 py-1 font-medium ${pick === id ? "text-ink" : "text-ink-3"}`}>
                  {pick === id && <motion.span layoutId="outlook-pill" className="absolute inset-0 rounded-[10px] bg-surface shadow-[0_1px_2px_rgba(13,27,42,0.12)]" aria-hidden />}
                  <span className="relative">{label}</span>
                </button>
              ))}
            </div>
          </div>
          {outlook === null ? (
            <p className="mt-3 h-20 rounded-lg bg-surface-3" aria-hidden />
          ) : outlook === "error" || !days.length ? (
            <p className="mt-2 text-[15px] text-ink-2">No C-HARM run could be read.</p>
          ) : (
            <>
              <p className="mt-2 text-[15px] leading-snug text-ink" data-testid="model-where">
                {day?.[pick]?.here != null ? (
                  <>
                    In these waters&apos; own 3 km cell: <b className="tabular">{fmtPct(day[pick]!.here)}</b>
                  </>
                ) : day?.[pick] && day[pick]!.n > 0 ? (
                  <>
                    No value for these nearshore waters. The <b>{day[pick]!.n} offshore cells within {NEIGHBOURHOOD_KM} km</b> (nearest {day[pick]!.nearest!.km.toFixed(1)} km): median <b className="tabular">{fmtPct(day[pick]!.median)}</b>, range {fmtPct(day[pick]!.min)}–{fmtPct(day[pick]!.max)}
                  </>
                ) : day?.[pick]?.nearest ? (
                  <>
                    No cells within {NEIGHBOURHOOD_KM} km. Nearest cell, {day[pick]!.nearest!.km.toFixed(0)} km away: {fmtPct(day[pick]!.nearest!.value)} (it describes that cell, not these waters)
                  </>
                ) : (
                  "No model cell near these waters."
                )}
              </p>
              <p className="text-[13px] text-ink-3">{(pick === "pda" ? thrPda : thrPn) ?? ""}</p>
              <OutlookBars days={days} pick={pick} active={lead} now={now} />
              {hist && hist.length > 1 && (
                <div className="mt-3">
                  <p className="text-[13px] text-ink-3">Nowcast median for the cells within {port?.charm?.radius_km ?? NEIGHBOURHOOD_KM} km, last {hist.length} runs</p>
                  <HistoryLine points={hist} />
                </div>
              )}
              <p className="mt-2 text-[13px] leading-snug text-ink-2">
                Bloom: the chance that Pseudo-nitzschia exceeds the model&apos;s threshold. Toxin: the chance that particulate domoic acid in the water does. Neither is a measurement, and neither is a test of seafood.
              </p>
              <button type="button" onClick={() => onFocus(modelOn ? null : "model")} aria-pressed={modelOn} className="cw-press mt-2 text-[13px] font-medium text-accent hover:underline" data-testid="ev-model-map">
                {modelOn ? "Shown on the map: the real 3 km cells and the nearshore gaps" : "Show the model's cells and its nearshore gaps on the map"}
              </button>
            </>
          )}
        </div>
      </div>
      <p className="text-[13px] leading-snug text-ink-3">
        No sample is not a clean result, and seawater is not seafood: only CDPH and CDFW test shellfish and fish themselves for toxin.
      </p>
    </section>
  );
}

/** Everything the section says, computed once (also used for the page's one-line answer). */
export function useHabEvidence(stations: StationLite[] | null, lat: number | null, lon: number | null, now: Date | null) {
  return useMemo(() => {
    if (lat == null || lon == null || !now) return null;
    return { toxin: measuredEvidence(stations, lat, lon, "toxin", now), cells: measuredEvidence(stations, lat, lon, "cells", now), nearest: stationsAround(stations, lat, lon, now, EVIDENCE_RADIUS_KM)[0] ?? null };
  }, [stations, lat, lon, now]);
}

export { NEIGHBOURHOOD_KM };

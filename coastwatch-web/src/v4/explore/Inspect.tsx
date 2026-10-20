"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { LayerArtifact, Manifest, PortsCollection, ValueGrid } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { artifactUrl, charmRun, type CharmVariable } from "@/lib/layers";
import { areaCells, areaStats, type AreaCell } from "@/lib/area";
import { pageClaims, type Claim, type EvidenceGraph, type EvidenceNode } from "@/lib/harborEvidence";
import { stationsNear, coastHref } from "@/lib/coast";
import { distanceKm, loadGrid } from "@/lib/grid";
import { DAY, unitLabel } from "@/lib/observations";
import { OFFICIAL_STATUS } from "@/content/copy";
import { disputeOf, officialAt } from "@/lib/official";
import { currentsAt, forecastAt, pnSibling, satelliteAt, type Readout } from "@/lib/readout";
import { KNOTS_PER_MS, type CurrentField } from "@/lib/currents";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, kmLabel, lastHabWords, relTime } from "@/v4/ui/time";
import { lastHabSample } from "@/lib/provenance";
import { SeriesChart, recentSamples } from "@/v4/charts/SeriesChart";
import { useConditions, useEvidence, type CellRead } from "./reads";
import { VARIABLE_WORDS, runDays } from "./layers";
import { packForPort } from "@/v4/offline/packs";
import { useSavedSnapshot } from "@/v4/offline/snapshot";
import type { Selection } from "@/v4/state/selection";

type Ctx = {
  manifest: Manifest;
  baseUrl: string;
  ports: PortsCollection | null;
  portIntel: PortIntelCollection | null;
  stations: StationLite[] | null;
  now: Date | null;
  variable: CharmVariable;
  lead: number;
  /** the map's layer: the C-HARM day is named only while the forecast is on the map */
  layer?: string;
  onSelect: (s: Selection) => void;
  official?: OfficialDataset | null;
  /** what can be read at a point: the forecast for the chosen variable and day, the satellite view, and the currents hour once loaded */
  reads?: { forecast: LayerArtifact | null; satellite: LayerArtifact | null; currents: { layer: LayerArtifact; field: CurrentField } | null };
};

const KIND: Record<string, { word: string; tone: string; dot: string; hollow: string }> = {
  official: { word: "Official", tone: "text-n-official", dot: "bg-n-official", hollow: "border-2 border-n-official" },
  // the agencies' standing pages in Neritic's unreviewed copy: the official colour, drawn hollow, never a notice's mark
  agency_page: { word: "Agency page · Neritic's copy", tone: "text-n-official", dot: "border-2 border-n-official", hollow: "border-2 border-n-official" },
  measured: { word: "Measured", tone: "text-n-measured", dot: "bg-n-measured", hollow: "border-2 border-n-measured" },
  model: { word: "Model", tone: "text-n-model", dot: "bg-n-model", hollow: "border-2 border-n-model" },
  satellite: { word: "Satellite", tone: "text-n-measured", dot: "bg-n-measured", hollow: "border-2 border-n-measured" },
  conditions: { word: "Conditions", tone: "text-n-conditions", dot: "bg-n-conditions", hollow: "border-2 border-n-conditions" },
  gap: { word: "Not known", tone: "text-n-ink-3", dot: "bg-n-gap", hollow: "border-2 border-n-gap" },
};

export function Header({ icon, kind, title, sub, onClose, onBack, canBack, action }: { icon: string; kind: string; title: string; sub?: string; onClose: () => void; onBack?: () => void; canBack?: boolean; action?: React.ReactNode }) {
  return (
    <header className="flex items-start gap-2">
      {canBack && onBack && (
        <button type="button" onClick={onBack} aria-label="Back to the previous view" data-testid="inspect-back" className="nx-press nx-focus -ml-1 grid h-11 w-11 shrink-0 place-items-center rounded-lg text-n-ink-2 hover:bg-n-surface-2">
          <Icon name="back" />
        </button>
      )}
      <div className="min-w-0 flex-1 pt-1">
        <p className="nx-eyebrow flex items-center gap-1.5">
          <Icon name={icon} className="h-3.5 w-3.5" />
          {kind}
        </p>
        <h2 className="mt-1 text-balance font-serif text-[24px] font-semibold leading-tight text-n-ink" data-testid="inspect-title">
          {title}
        </h2>
        {sub && <p className="mt-1 text-[14px] text-n-ink-2">{sub}</p>}
      </div>
      {action}
      <button type="button" onClick={onClose} aria-label="Close" data-testid="inspect-close" className="nx-press nx-focus grid h-11 w-11 shrink-0 place-items-center rounded-lg text-n-ink-2 hover:bg-n-surface-2">
        <Icon name="close" />
      </button>
    </header>
  );
}

function Section({ title, children, testid }: { title: string; children: React.ReactNode; testid?: string }) {
  return (
    <section className="flex flex-col gap-2.5 border-t border-n-line pt-4" data-testid={testid}>
      <h3 className="nx-eyebrow">{title}</h3>
      {children}
    </section>
  );
}

/** A claim's supporting node, as a selection on the map (when it has a place). */
export function nodeSelection(n: EvidenceNode | undefined, stations: StationLite[] | null): Selection | null {
  if (!n) return null;
  const sid = n.id?.split(":")[1];
  if (n.class === "measurement" && sid && stations?.some((s) => s.station_id === sid)) return { kind: "station", id: sid };
  if (n.class === "published_forecast" && n.source.id === "charm") {
    const w = n.where;
    if (w?.lat != null && w?.lon != null) return { kind: "cell", lat: w.lat, lon: w.lon };
  }
  if (n.where?.lat != null && n.where?.lon != null && n.class !== "measurement") return { kind: "point", lat: n.where.lat, lon: n.where.lon };
  return null;
}

/** `fallback`: where "Show on map" goes when no supporting node has a place of its own (an official notice: the harbour's waters). */
export function ClaimRow({ c, graph, stations, onSelect, fallback = null }: { c: Claim; graph: EvidenceGraph; stations: StationLite[] | null; onSelect?: (s: Selection) => void; fallback?: Selection | null }) {
  const k = KIND[c.kind] ?? KIND.gap;
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const target = (c.supports ?? []).map((id) => nodeSelection(nodes.get(id), stations)).find(Boolean) ?? fallback;
  return (
    // an alert keeps its dot, its "Official" label and a 2 px official rule; no filled card
    <li className={`flex gap-3 ${c.tone === "alert" ? "-ml-3 border-l-2 border-n-official pl-[10px]" : ""}`} data-testid={`claim-${c.id}`} data-kind={c.kind} data-tone={c.tone} data-supports={(c.supports ?? []).join(" ")}>
      {/* older evidence is drawn hollow (its age is in the line itself), as in v2 */}
      <span className={`mt-2 h-2.5 w-2.5 shrink-0 rounded-full ${c.aged ? k.hollow : k.dot}`} data-aged={c.aged ? "true" : undefined} aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-[15.5px] font-semibold leading-snug text-n-ink" data-headline>
          {c.headline}
        </p>
        {/* the agency's own words: quoted in full, never paraphrased into the headline */}
        {c.quote && (
          <blockquote className="mt-1 border-l-2 border-n-official-line pl-2.5 font-serif text-[15.5px] leading-snug text-n-ink" data-testid="claim-quote">
            &ldquo;{c.quote}&rdquo;
          </blockquote>
        )}
        {c.also && (
          <p className="mt-1.5 text-[14px] leading-snug text-n-ink-2" data-testid="claim-also">
            <span className="font-semibold text-n-ink">{c.also.label}:</span> &ldquo;{c.also.quote}&rdquo;{" "}
            <a href={c.also.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex items-center gap-1 font-semibold text-n-accent">
              source <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          </p>
        )}
        {c.detail && <p className="mt-0.5 text-[14px] leading-snug text-n-ink-2">{c.detail}</p>}
        {/* the measurement against the station's own record (its own node in the evidence); zeros said apart, never ranked */}
        {c.history && (
          <p className="mt-0.5 text-[14px] leading-snug text-n-ink-2" data-testid={`claim-${c.id}-history`}>
            {c.history}
          </p>
        )}
        {/* where the line's words come from: the species line is Neritic's unreviewed copy of the agencies' pages */}
        {c.provenance && (
          <p className="mt-0.5 text-[13px] leading-snug text-n-ink-3" data-testid={`claim-${c.id}-provenance`}>
            {c.provenance}
          </p>
        )}
        {/* every agency sentence one tap away, each quoted under its own agency and page */}
        {c.agencyWords && c.agencyWords.length > 0 && (
          <details className="group mt-0.5 text-[14px]" data-testid={`claim-${c.id}-agency`}>
            <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-semibold text-n-accent">
              <span className="group-open:hidden">What the agencies&rsquo; pages say</span>
              <span className="hidden group-open:inline">Hide the agencies&rsquo; words</span>
            </summary>
            <ul className="flex flex-col gap-2">
              {c.agencyWords.map((w) => (
                <li key={w.url + w.quote.slice(0, 24)} data-testid={`claim-${c.id}-word`} data-agency={w.agency}>
                  <blockquote className="border-l-2 border-n-official-line pl-2.5 font-serif text-[15.5px] leading-snug text-n-ink">&ldquo;{w.quote}&rdquo;</blockquote>
                  <a href={w.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-[13px] font-semibold text-n-accent">
                    {w.label} <Icon name="external" className="h-3.5 w-3.5" />
                  </a>
                </li>
              ))}
            </ul>
          </details>
        )}
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
          <span className={`font-semibold ${k.tone}`}>{k.word}</span>
          {c.source?.url && (
            <a href={c.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
              {c.source.label} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          {target && onSelect && (
            <button type="button" onClick={() => onSelect(target)} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-ink-2 hover:text-n-ink" data-testid={`claim-show-${c.id}`}>
              <Icon name="locate" className="h-3.5 w-3.5" /> Show on map
            </button>
          )}
        </div>
      </div>
    </li>
  );
}

function Loading({ what }: { what: string }) {
  return (
    <p className="flex items-center gap-2 text-[14px] text-n-ink-2" role="status">
      <svg width="28" height="10" viewBox="0 0 28 10" aria-hidden className="text-n-accent">
        <path className="nx-sweep" d="M1 6c4-4 7-4 11 0s7 4 11 0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      {what}
    </p>
  );
}

function Failed({ what, message, retry }: { what: string; message: string; retry?: () => void }) {
  return (
    <div className="rounded-xl border border-n-line-strong p-3 text-[14px]" role="alert">
      <p className="font-semibold text-n-down">{what} could not be loaded</p>
      <p className="mt-0.5 text-n-ink-2">{message}. Everything else on this page is unaffected.</p>
      {retry && (
        <button type="button" onClick={retry} className="nx-press nx-focus mt-2 flex min-h-11 items-center gap-2 rounded-lg border border-n-line-strong px-3 font-semibold">
          <Icon name="refresh" className="h-4 w-4" /> Try again
        </button>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------- harbour

export function HarborView({ code, ctx }: { code: number; ctx: Ctx }) {
  const intel = ctx.portIntel?.ports.find((p) => p.port_code === code) ?? null;
  const f = ctx.ports?.features.find((x) => x.properties.port_code === code);
  const [lon, lat] = (f?.geometry.coordinates as number[] | undefined) ?? [intel?.lon ?? 0, intel?.lat ?? 0];
  const name = intel?.display_name ?? f?.properties.display_name ?? "Harbour";
  const [ev, retryEv] = useEvidence(`port-${code}`);
  const [cond, retryCond] = useConditions(lat ? { lat, lon } : null);
  const claims = useMemo(() => (ev?.status === "ready" ? pageClaims(ev.data) : []), [ev]);
  const near = useMemo(() => stationsNear(ctx.stations, lat, lon), [ctx.stations, lat, lon]);
  const harbor = cond?.status === "ready" ? cond.data.harbor : null;
  // served from a saved field pack (offline): the evidence is dated by when it was saved, never "now"
  const saved = useSavedSnapshot(packForPort(code)?.id ?? null);
  // the day the lines are for (the evidence's own day) and the model day chosen on the map's timebar
  const evDay = ev?.status === "ready" ? ev.data.window.to : null;
  const mapDay = useMemo(() => runDays(ctx.manifest, ctx.now).find((d) => d.lead === ctx.lead) ?? null, [ctx.manifest, ctx.now, ctx.lead]);
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-harbor">
      <Link href={coastHref({ place: `port-${code}` })} className="nx-press nx-focus flex min-h-[52px] items-center justify-between gap-3 rounded-xl bg-n-ink px-4 text-[15.5px] font-semibold text-n-bg" data-testid="open-briefing">
        Open the {name} briefing <Icon name="chevronRight" />
      </Link>
      <Section title={saved ? `As saved ${clockLabel(saved.savedAt)}` : "What the evidence says now"} testid="harbor-claims">
        {evDay && (
          <p className="-mt-1 text-[13px] leading-snug text-n-ink-3" data-testid="harbor-day">
            For {dayLabel(evDay, true)}
            {mapDay?.date && ctx.layer === "forecast" ? (mapDay.date === evDay ? ", the C-HARM day on the map" : ` · the map shows C-HARM for ${dayLabel(mapDay.date, true)} (${mapDay.lead === 0 ? "nowcast" : "forecast"})`) : ""}
          </p>
        )}
        {!ev || ev.status === "loading" ? (
          <Loading what="Reading the evidence for these waters…" />
        ) : ev.status === "error" ? (
          <Failed what="The evidence" message={ev.message} retry={retryEv} />
        ) : (
          <ul className="flex flex-col divide-y divide-n-line [&>li]:py-3 [&>li:first-child]:pt-0 [&>li:last-child]:pb-0">
            {claims.map((c) => (
              <ClaimRow key={c.id} c={c} graph={ev.data} stations={ctx.stations} onSelect={ctx.onSelect} />
            ))}
          </ul>
        )}
      </Section>
      <Section title="Harbour water" testid="harbor-water">
        {!cond || cond.status === "loading" ? (
          <Loading what="Asking NOAA CO-OPS…" />
        ) : cond.status === "error" ? (
          <Failed what="Harbour conditions" message={cond.message} retry={retryCond} />
        ) : harbor?.status === "ok" && harbor.station ? (
          <div className="text-[14.5px] leading-snug">
            <p>
              <span className="nx-data text-[22px] font-semibold">{harbor.data.waterC.value.toFixed(1)} °C</span>{" "}
              <span className="text-n-ink-2">at {harbor.station.name} tide station, {kmLabel(harbor.station.km)}</span>
            </p>
            <p className="text-n-ink-3">
              {clockLabel(harbor.data.waterC.at)} · {harbor.station.sensorNote} · {harbor.data.flagNote}
            </p>
            <a href={harbor.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-[13px] font-semibold text-n-accent">
              NOAA CO-OPS {harbor.station.id} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          </div>
        ) : (
          <p className="text-[14px] text-n-ink-2">{harbor?.status === "none" || harbor?.status === "unavailable" ? harbor.reason : "No harbour water measurement."}</p>
        )}
      </Section>
      <Section title="Shore stations nearby" testid="harbor-stations">
        {near.near.length ? (
          <ul className="flex flex-col">
            {near.near.slice(0, 4).map((n) => (
              <li key={n.station.station_id}>
                <button type="button" onClick={() => ctx.onSelect({ kind: "station", id: n.station.station_id })} className="nx-focus flex min-h-12 w-full items-center justify-between gap-3 rounded-lg px-1 text-left hover:bg-n-surface-2">
                  <span className="flex min-w-0 flex-col">
                    <span className="text-[14.5px] font-semibold">{n.station.name}</span>
                    <span className="text-[12.5px] text-n-ink-3" data-testid="harbor-station-last">{lastHabWords(lastHabSample(n.station), ctx.now)}</span>
                  </span>
                  <span className="nx-data shrink-0 text-[13px] text-n-ink-3">
                    {kmLabel(n.km)} {n.bearing}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[14px] text-n-ink-2">No CalHABMAP shore station within 40 km. No sample is not a clean result.</p>
        )}
      </Section>
      {intel && <HarbourSummary intel={intel} lead={ctx.lead} />}
    </div>
  );
}

const PCT = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const CHL = (v: number | null | undefined) => (v == null ? "—" : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

/**
 * Advanced: what each product says in the harbour's neighbourhood, as the pipeline published it
 * (port intel): per variable the median, range and how many cells or pixels it rests on, with their
 * native size and dates. Neighbourhood statistics, never a value at the harbour itself.
 */
function HarbourSummary({ intel, lead }: { intel: PortIntelCollection["ports"][number]; lead: number }) {
  const c = intel.charm;
  const day = c?.leads.find((l) => l.lead_days === lead) ?? c?.leads[0] ?? null;
  const chl = intel.chlorophyll;
  return (
    <details className="rounded-xl border border-n-line px-3 py-2" data-testid="harbor-summary">
      <summary className="nx-focus flex min-h-11 cursor-pointer items-center justify-between text-[14px] font-semibold text-n-ink">
        Advanced: each product near {intel.display_name}
        <Icon name="chevronDown" className="h-4 w-4 text-n-ink-3" />
      </summary>
      <div className="flex flex-col gap-3 pb-2 text-[13.5px]">
        {c && day ? (
          <div data-testid="harbor-summary-charm">
            <p className="font-semibold text-n-model">NOAA C-HARM model · 3 km cells</p>
            <p className="text-[12.5px] text-n-ink-3">
              Cells whose centres lie within {c.radius_km} km of the port point: {c.cells_in_radius}{c.nearest_cell_km != null ? ` (nearest cell ${kmLabel(c.nearest_cell_km)})` : ""}. Valid {dayLabel(day.valid_date)}{c.issued_date ? `, run issued ${dayLabel(c.issued_date)}` : ""}.
            </p>
            <table className="mt-1 w-full text-left">
              <thead className="text-[12px] text-n-ink-3">
                <tr>
                  <th className="py-1 font-semibold">Variable</th>
                  <th className="py-1 text-right font-semibold">Median</th>
                  <th className="py-1 text-right font-semibold">Range</th>
                  <th className="py-1 text-right font-semibold">Cells</th>
                </tr>
              </thead>
              <tbody className="nx-data">
                {(Object.keys(VARIABLE_WORDS) as CharmVariable[]).map((v) => {
                  const x = day.variables[v as keyof typeof day.variables];
                  return (
                    <tr key={v} className="border-t border-n-line" data-testid={`harbor-summary-${v}`}>
                      <td className="py-1 font-sans text-n-ink-2">{VARIABLE_WORDS[v].short}</td>
                      <td className="py-1 text-right">{PCT(x?.median)}</td>
                      <td className="py-1 text-right text-n-ink-2">{x?.n ? `${PCT(x.min)}–${PCT(x.max)}` : "—"}</td>
                      <td className="py-1 text-right text-n-ink-2">{x?.n ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-1 text-[12.5px] text-n-ink-3">Probabilities over the neighbourhood&rsquo;s cells, not at the harbour, not a toxin measurement. A count below {c.cells_in_radius} means cells without a value (no value is never zero).</p>
          </div>
        ) : null}
        {chl && (
          <div data-testid="harbor-summary-chl">
            <p className="font-semibold text-n-measured">Satellite chlorophyll · VIIRS 750 m pixels</p>
            {chl.latest ? (
              <>
                <p>
                  Median <span className="nx-data font-semibold">{CHL(chl.latest.median)} mg/m³</span> (range {CHL(chl.latest.min)}–{CHL(chl.latest.max)}) over <span className="nx-data">{chl.latest.n}</span> pixels within {chl.radius_km} km
                  {chl.latest_valid_fraction != null ? `; ${Math.round(chl.latest_valid_fraction * 100)}% of the neighbourhood's pixels had a value` : ""}.
                </p>
                <p className="text-[12.5px] text-n-ink-3">
                  Each pixel&rsquo;s median over {chl.composite_days} days centred on {chl.latest_center_date ? dayLabel(chl.latest_center_date) : "—"}. Algae of every kind, not toxin; a pixel without a value is cloud or no overpass.
                </p>
              </>
            ) : (
              <p className="text-n-ink-2">Not available{chl.error ? `: ${chl.error.split(":")[0]}` : ""}. No value is not low chlorophyll.</p>
            )}
            {chl.source_url && (
              <a href={chl.source_url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
                NOAA CoastWatch ERDDAP <Icon name="external" className="h-3.5 w-3.5" />
              </a>
            )}
          </div>
        )}
        {intel.caveats?.length ? (
          <ul className="list-disc space-y-0.5 pl-4 text-[12.5px] text-n-ink-3">
            {intel.caveats.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
        ) : null}
      </div>
    </details>
  );
}

// ------------------------------------------------------------------------------------- station

const VARS = [
  { id: "pDA", label: "Particulate domoic acid", units: "ng/mL" },
  { id: "pn_seriata", label: "Pseudo-nitzschia seriata group", units: "cells/L" },
  { id: "pn_delicatissima", label: "Pseudo-nitzschia delicatissima group", units: "cells/L" },
  { id: "chl_extracted", label: "Chlorophyll (extracted)", units: "mg/m³" },
] as const;

export function StationView({ id, ctx }: { id: string; ctx: Ctx }) {
  const st = ctx.stations?.find((s) => s.station_id === id) ?? null;
  const avail = VARS.filter((v) => st?.recent?.[v.id]?.length);
  const [v, setV] = useState<string>(avail[0]?.id ?? "pDA");
  if (!st) return <p className="text-[14px] text-n-ink-2">This station is not in the published observations.</p>;
  const meta = VARS.find((x) => x.id === v) ?? VARS[0];
  const samples = recentSamples(st.recent?.[v]);
  const to = samples.length ? samples[samples.length - 1].t + 3 * DAY : Date.now();
  const from = to - 370 * DAY;
  const harbor = (() => {
    let best: { name: string; km: number } | null = null;
    for (const f of ctx.ports?.features ?? []) {
      const [plon, plat] = f.geometry.coordinates as number[];
      const km = distanceKm(st.lat, st.lon, plat, plon);
      if (!best || km < best.km) best = { name: f.properties.display_name ?? f.properties.name, km };
    }
    return best;
  })();
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-station">
      {/* the data themselves, one tap from the station's name (a researcher's first question) */}
      <div className="-mt-1 flex flex-wrap gap-2">
        {st.request_url && (
          <a href={st.request_url} target="_blank" rel="noreferrer" className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold text-n-ink" data-testid="station-csv">
            <Icon name="download" className="h-4 w-4" /> Original data (CSV)
          </a>
        )}
        <Link href={`/research?station=${encodeURIComponent(st.station_id)}`} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold text-n-ink" data-testid="station-research">
          <Icon name="research" className="h-4 w-4" /> Full record in Research
        </Link>
      </div>
      <Section title="Latest samples" testid="station-latest">
        <ul className="flex flex-col gap-2">
          {st.latest.map((l) => (
            <li key={l.variable} className="flex items-baseline justify-between gap-3 text-[14.5px]">
              <span className="text-n-ink-2">{l.label}</span>
              <span className="text-right">
                <span className="nx-data font-semibold text-n-ink">{l.value == null ? "—" : l.qualifier === "reported_zero" ? "0" : l.value.toLocaleString("en-US", { maximumSignificantDigits: 3 })}</span> <span className="text-n-ink-3">{unitLabel(l.units)}</span>
                <span className="block text-[12.5px] text-n-ink-3">
                  {dayLabel(l.date, true)}
                  {ctx.now ? ` · ${relTime(l.date, ctx.now)}` : ""}
                  {l.qualifier === "reported_zero" ? " · reported as zero" : ""}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </Section>
      <Section title="The last year of samples" testid="station-series">
        {avail.length > 1 && (
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Measurement">
            {avail.map((x) => (
              <button key={x.id} type="button" role="radio" aria-checked={v === x.id} onClick={() => setV(x.id)} className={`nx-press nx-focus min-h-11 rounded-lg px-2.5 text-[13px] font-semibold ${v === x.id ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink-2"}`}>
                {x.label.replace("Pseudo-nitzschia", "PN").replace("Particulate domoic acid", "pDA")}
              </button>
            ))}
          </div>
        )}
        <SeriesChart samples={samples} variable={meta.id} units={meta.units} label={meta.label} from={from} to={to} />
      </Section>
      <Section title="What this station is" testid="station-provenance">
        <p className="text-[14px] leading-snug text-n-ink-2">
          CalHABMAP shore station, sampled from a pier: seawater, not seafood. A value describes the water at the pier on that day.
          {harbor ? ` ${kmLabel(harbor.km)} from ${harbor.name}.` : ""} Sampling is roughly weekly; no sample is not a clean result.
        </p>
        <div className="flex flex-wrap gap-x-4">
          <a href={st.source_url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-[13.5px] font-semibold text-n-accent">
            SCCOOS ERDDAP record <Icon name="external" className="h-3.5 w-3.5" />
          </a>
        </div>
      </Section>
    </div>
  );
}

// ------------------------------------------------------------------------------------- cell

function pct(v: number | null) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

/** A probability as stored, to 3 decimals, with the grid's own rounding bound (16-bit storage). */
export function preciseProbability(v: number, quantErr: number | null | undefined): string {
  return `${v.toFixed(3)}${quantErr != null ? ` ± ${Number(quantErr.toPrecision(2)).toFixed(Math.max(0, -Math.floor(Math.log10(quantErr)) + 1))}` : ""}`;
}

/**
 * One C-HARM cell. `precise` (Research) adds the stored probability to 3 decimals with the
 * grid's quantisation error, and says plainly what the number is.
 */
export function CellBody({ read, ctx, variable, precise }: { read: CellRead | null; ctx: Ctx; variable: CharmVariable; precise?: { quantErr: number | null } }) {
  const run = charmRun(ctx.manifest);
  if (!read || read.status === "loading") return <Loading what="Reading the published grid…" />;
  if (read.status === "error") return <Failed what="The C-HARM grid" message={read.message} />;
  const own = read.leads[ctx.lead]?.value ?? null;
  const hr = read.harbor;
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-cell">
      <div>
        <p className="text-[14px] text-n-ink-2">{VARIABLE_WORDS[variable].what}</p>
        <p className="mt-1">
          <span className="nx-data text-[34px] font-semibold leading-none text-n-ink" data-testid="cell-value">
            {own == null ? "No value" : pct(own)}
          </span>
          {own != null && <span className="ml-2 text-[14px] text-n-ink-2">for {read.leads[ctx.lead]?.date ? dayLabel(read.leads[ctx.lead].date!, true) : "this day"}</span>}
        </p>
        {precise && own != null && (
          <p className="nx-data mt-1.5 text-[14px] text-n-ink" data-testid="cell-precise">
            probability {preciseProbability(own, precise.quantErr)}
            <span className="font-sans text-[12.5px] text-n-ink-3"> (± storage rounding)</span>
          </p>
        )}
        <p className="mt-1 text-[13px] text-n-ink-3">
          {precise ? `Model output for one ${read.sizeKm.toFixed(0)} km cell, not a toxin measurement` : `A published probability for one ${read.sizeKm.toFixed(1)} km cell, not a toxin measurement`}
          {run ? ` · run issued ${dayLabel(run.issued_date)}${run.issued_date_derived ? " (inferred)" : ""}` : ""}
        </p>
      </div>
      {own == null && (
        <div className="rounded-xl border border-dashed border-n-line-strong p-3 text-[14px] leading-snug" data-testid="cell-gap">
          {!read.gap.inDomain ? (
            <p>This point is outside C-HARM&rsquo;s domain.</p>
          ) : (
            <p>
              C-HARM publishes no value for this cell (nearshore and masked cells are not modelled).
              {read.gap.nearest && read.gap.nearestKm != null && (
                <>
                  {" "}The nearest cell with a value is {kmLabel(read.gap.nearestKm)} away.{" "}
                  <button type="button" className="nx-focus font-semibold text-n-accent underline underline-offset-2" onClick={() => ctx.onSelect({ kind: "cell", lat: read.gap.nearest!.lat, lon: read.gap.nearest!.lon })} data-testid="cell-nearest">
                    Select it
                  </button>
                </>
              )}
            </p>
          )}
        </div>
      )}
      {hr && (
        <Section title="Does it describe a harbour?" testid="cell-harbor">
          <p className="text-[14px] leading-snug text-n-ink-2" data-testid="cell-harbor-text">
            {hr.ownCell === "same"
              ? own == null
                ? `This is the cell containing ${hr.name}, and C-HARM publishes no value for it: there is no model estimate for the harbour.`
                : `Yes: this is the cell containing ${hr.name}.`
              : hr.ownCell === "no-value"
                ? `No. ${hr.name} is ${kmLabel(hr.km)} away and its own cell has no value; an offshore cell is context, never the harbour's value.`
                : hr.ownCell === "valued-other"
                  ? `No. ${hr.name} (${kmLabel(hr.km)} away) has its own cell with a value; this is a neighbouring cell.`
                  : `${hr.name} is ${kmLabel(hr.km)} away, outside the model domain.`}
          </p>
        </Section>
      )}
      <Section title="This run, day by day" testid="cell-leads">
        <ol className="grid grid-cols-4 gap-1.5">
          {read.leads.map((l) => (
            <li key={l.lead} className={`flex flex-col items-center rounded-lg px-1 py-2 ${l.lead === ctx.lead ? "bg-n-accent-bg ring-1 ring-n-accent" : "bg-n-surface-2"}`}>
              <span className="text-[12px] text-n-ink-3">{l.date ? dayLabel(l.date, true) : "—"}</span>
              <span className="nx-data mt-0.5 text-[16px] font-semibold text-n-ink">{l.issued ? pct(l.value) : "not issued"}</span>
              <span className="text-[11px] text-n-ink-3">{l.lead === 0 ? "nowcast" : "forecast"}</span>
            </li>
          ))}
        </ol>
        <p className="text-[12.5px] text-n-ink-3">The run&rsquo;s own four days; nothing beyond its horizon is shown, and values are never blended between days.</p>
      </Section>
      {read.cell && (
        <p className="nx-data text-[12.5px] text-n-ink-3">
          Cell centre {read.cell.lat.toFixed(3)}°N {Math.abs(read.cell.lon).toFixed(3)}°W · row {read.cell.row}, column {read.cell.col}
        </p>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------- point

export function PointView({ lat, lon, ctx }: { lat: number; lon: number; ctx: Ctx }) {
  const near = useMemo(() => stationsNear(ctx.stations, lat, lon), [ctx.stations, lat, lon]);
  const notices = useMemo(() => officialAt(ctx.official ?? null, lon, lat), [ctx.official, lat, lon]);
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-point">
      <p className="nx-data text-[13px] text-n-ink-3">
        {lat.toFixed(4)}°N {Math.abs(lon).toFixed(4)}°W
      </p>
      {ctx.reads && <PointReads lat={lat} lon={lon} ctx={ctx} />}
      <button type="button" onClick={() => ctx.onSelect({ kind: "cell", lat, lon })} className="nx-press nx-focus flex min-h-12 items-center justify-between rounded-xl border border-n-line-strong px-3 text-[15px] font-semibold">
        What C-HARM publishes for this cell <Icon name="chevronRight" />
      </button>
      {ctx.official !== undefined && (
        <Section title="Official notices at this point">
          {notices.length ? (
            <ul className="flex flex-col gap-2" data-testid="point-notices">
              {notices.map(({ record: r, reason }) => {
                const d = disputeOf(ctx.official ?? null, r.id);
                return (
                  <li key={r.id} className="text-[14px]" data-testid={`point-notice-${r.id}`} data-disputed={d ? "true" : undefined}>
                    <p className="font-semibold leading-snug">{r.title}</p>
                    <p className="text-[13px] text-n-ink-2">
                      {r.agency} · {reason} · <span className="text-n-official">not verified</span>
                      {d ? ` · ${d.agency} release ${d.release} appears to lift it; not shown as in effect` : ""}
                    </p>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="text-[14px] text-n-ink-2">No notice in Neritic&rsquo;s list covers this point.</p>
          )}
          <p className="mt-1 text-[13px] leading-snug text-n-ink-3" data-testid="point-missing-not-open">
            {OFFICIAL_STATUS.missingNotOpen}
          </p>
        </Section>
      )}
      <Section title="Nearest shore stations">
        {near.near.length || near.nearestBeyond ? (
          <ul>
            {(near.near.length ? near.near.slice(0, 3) : [near.nearestBeyond!]).map((n) => (
              <li key={n.station.station_id}>
                <button type="button" onClick={() => ctx.onSelect({ kind: "station", id: n.station.station_id })} className="nx-focus flex min-h-12 w-full items-center justify-between gap-3 rounded-lg px-1 text-left hover:bg-n-surface-2">
                  <span className="flex min-w-0 flex-col">
                    <span className="text-[14.5px] font-semibold">{n.station.name}</span>
                    <span className="text-[12.5px] text-n-ink-3" data-testid="harbor-station-last">{lastHabWords(lastHabSample(n.station), ctx.now)}</span>
                  </span>
                  <span className="nx-data shrink-0 text-[13px] text-n-ink-3">
                    {kmLabel(n.km)} {n.bearing}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[14px] text-n-ink-2">No shore station nearby.</p>
        )}
      </Section>
    </div>
  );
}

/** The published values at a point: the same reads as the desktop hover label, reachable by click,
 *  keyboard and screen reader. Each value is the cell or pixel that contains the point, never the
 *  nearest one and never interpolated, with its own product, resolution and time. */
function PointReads({ lat, lon, ctx }: { lat: number; lon: number; ctx: Ctx }) {
  const reads = ctx.reads!;
  const [rows, setRows] = useState<{ id: string; label: string; r: Readout | null; err?: boolean }[] | null>(null);
  useEffect(() => {
    let live = true;
    const jobs: { id: string; label: string; run: () => Promise<Readout> }[] = [];
    if (reads.forecast) jobs.push({ id: "forecast", label: "NOAA C-HARM model", run: () => forecastAt(ctx.baseUrl, reads.forecast!, lat, lon, pnSibling(ctx.manifest, reads.forecast!)) });
    if (reads.satellite) jobs.push({ id: "satellite", label: "Satellite chlorophyll", run: () => satelliteAt(ctx.manifest, ctx.baseUrl, reads.satellite!, lat, lon) });
    if (reads.currents) jobs.push({ id: "currents", label: "Surface current", run: async () => currentsAt(reads.currents!.field, reads.currents!.layer, lat, lon) });
    Promise.all(jobs.map((j) => j.run().then((r) => ({ id: j.id, label: j.label, r }), () => ({ id: j.id, label: j.label, r: null, err: true })))).then((x) => live && setRows(x));
    return () => {
      live = false;
    };
  }, [reads.forecast, reads.satellite, reads.currents, ctx.baseUrl, ctx.manifest, lat, lon]);
  return (
    <Section title="What is published here">
      {!rows ? (
        <p className="text-[14px] text-n-ink-2" role="status">
          Reading the published cells…
        </p>
      ) : (
        <ul className="flex flex-col divide-y divide-n-line" data-testid="point-reads">
          {rows.map(({ id, label, r, err }) => (
            <li key={id} className="py-2.5" data-testid={`point-${id}`}>
              <p className={`text-[12.5px] font-semibold uppercase tracking-[0.06em] ${r?.kind === "model" ? "text-n-model" : "text-n-measured"}`}>
                {label} · {r?.kind === "model" ? "model" : "observation"}
              </p>
              {r?.value != null ? (
                <>
                  <p className="mt-0.5 flex items-baseline gap-1.5">
                    <span className="nx-data text-[20px] font-semibold" data-testid="point-value">
                      {r.value}
                    </span>
                    {r.unit && <span className="text-[13.5px] text-n-ink-2">{r.unit}</span>}
                    {/* at sea, speeds are read in knots */}
                    {id === "currents" && <span className="text-[13.5px] text-n-ink-2" data-testid="point-knots">≈ {(Number(r.value) * KNOTS_PER_MS).toFixed(1)} kn</span>}
                  </p>
                  <p className="text-[13.5px] text-n-ink-2">{r.what}</p>
                  {r.flag && <p className="text-[13px] text-n-stale">{r.flag}</p>}
                </>
              ) : (
                <p className="mt-0.5 text-[14px] text-n-ink" data-testid="point-none">
                  {err ? "Could not be read; the rest of this panel is unaffected." : r?.none}
                </p>
              )}
              {r && (
                <p className="mt-0.5 text-[13px] text-n-ink-3" data-testid="point-when">
                  {r.when}
                </p>
              )}
            </li>
          ))}
          {!reads.currents && <li className="py-2.5 text-[13.5px] text-n-ink-2">Surface currents: choose the currents layer to read the radar here.</li>}
        </ul>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------------------------- area

export { areaStats } from "@/lib/area";

/**
 * The C-HARM cells of one layer inside a box: how many have a published value and the spread of
 * those values. Shared by Explore's area tool and Research; `footer` receives the cell list (e.g.
 * for an export). Nothing is filled or averaged into a harbour value.
 */
export function AreaModel({ layer, baseUrl, bbox, variable, footer }: { layer: LayerArtifact | null; baseUrl: string; bbox: [number, number, number, number]; variable: CharmVariable; footer?: (cells: AreaCell[]) => React.ReactNode }) {
  const [grid, setGrid] = useState<{ key: string; grid: ValueGrid; codes: Uint16Array } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    if (!layer?.grid) return;
    loadGrid(artifactUrl(baseUrl, layer.grid.url))
      .then((codes) => live && setGrid({ key: layer.layer_id, grid: layer.grid!, codes }))
      .catch((e) => live && setErr((e as Error).message));
    return () => {
      live = false;
    };
  }, [layer, baseUrl]);
  const read = grid && grid.key === layer?.layer_id ? grid : null;
  const cells = useMemo(() => areaCells(read, bbox), [read, bbox]);
  const stats = useMemo(() => areaStats(read, bbox), [read, bbox]);
  return (
    <section className="flex flex-col gap-2 border-t border-n-line pt-4" data-testid="area-model">
      <h3 className="nx-eyebrow">
        C-HARM cells in the box ({VARIABLE_WORDS[variable].short}, {layer?.time.valid_date ? dayLabel(layer.time.valid_date, true) : "—"})
      </h3>
      {err ? (
        <p className="text-[14px] text-n-down">The grid could not be loaded: {err}</p>
      ) : !layer?.grid ? (
        <p className="text-[14px] text-n-ink-2">No C-HARM grid is published for this variable and day.</p>
      ) : !stats ? (
        <p className="text-[14px] text-n-ink-2">Reading the grid…</p>
      ) : stats.valued === 0 ? (
        <p className="text-[14px] text-n-ink-2">None of the {stats.domain} grid cells in this box has a published value (land, nearshore or outside the model).</p>
      ) : (
        <>
          <p className="text-[14.5px] leading-snug">
            <span className="nx-data font-semibold">{stats.valued}</span> of {stats.domain} grid cells have a published value (land and nearshore cells have none).
          </p>
          <dl className="grid grid-cols-3 gap-2 text-center">
            {(
              [
                ["Lowest", stats.min],
                ["Median", stats.median],
                ["Highest", stats.max],
              ] as const
            ).map(([k, v]) => (
              <div key={k} className="rounded-lg bg-n-surface-2 py-2">
                <dt className="text-[12px] text-n-ink-3">{k}</dt>
                <dd className="nx-data text-[18px] font-semibold">{pct(v)}</dd>
              </div>
            ))}
          </dl>
          <p className="text-[12.5px] leading-snug text-n-ink-3">A summary of the published cell probabilities inside the box: not an average for any harbour and not a measurement. Other variables are not mixed in.</p>
        </>
      )}
      {cells && footer?.(cells)}
    </section>
  );
}

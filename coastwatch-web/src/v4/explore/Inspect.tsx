"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { charmRun, type CharmVariable } from "@/lib/layers";
import { pageClaims, type Claim, type EvidenceGraph, type EvidenceNode } from "@/lib/harborEvidence";
import { stationsNear, coastHref } from "@/lib/coast";
import { distanceKm } from "@/lib/grid";
import { DAY } from "@/lib/observations";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, kmLabel, relTime } from "@/v4/ui/time";
import { SeriesChart, recentSamples } from "@/v4/charts/SeriesChart";
import { useConditions, useEvidence, type CellRead } from "./reads";
import { VARIABLE_WORDS } from "./layers";
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
  onSelect: (s: Selection) => void;
};

const KIND: Record<string, { word: string; tone: string; dot: string }> = {
  official: { word: "Official", tone: "text-n-official", dot: "bg-n-official" },
  measured: { word: "Measured", tone: "text-n-measured", dot: "bg-n-measured" },
  model: { word: "Model", tone: "text-n-model", dot: "bg-n-model" },
  satellite: { word: "Satellite", tone: "text-n-measured", dot: "bg-n-measured" },
  conditions: { word: "Conditions", tone: "text-n-conditions", dot: "bg-n-conditions" },
  gap: { word: "Not known", tone: "text-n-ink-3", dot: "bg-n-gap" },
};

export function Header({ icon, kind, title, sub, onClose, onBack, canBack }: { icon: string; kind: string; title: string; sub?: string; onClose: () => void; onBack?: () => void; canBack?: boolean }) {
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
    <li className="flex gap-3" data-testid={`claim-${c.id}`} data-kind={c.kind} data-supports={(c.supports ?? []).join(" ")}>
      <span className={`mt-2 h-2.5 w-2.5 shrink-0 rounded-full ${k.dot}`} aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-[15.5px] font-semibold leading-snug text-n-ink" data-headline>
          {c.headline}
        </p>
        {c.detail && <p className="mt-0.5 text-[14px] leading-snug text-n-ink-2">{c.detail}</p>}
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
          <span className={`font-semibold ${k.tone}`}>{k.word}</span>
          {c.source?.url && (
            <a href={c.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-8 items-center gap-1 font-semibold text-n-accent">
              {c.source.label} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          {target && onSelect && (
            <button type="button" onClick={() => onSelect(target)} className="nx-focus inline-flex min-h-8 items-center gap-1 font-semibold text-n-ink-2 hover:text-n-ink" data-testid={`claim-show-${c.id}`}>
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
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-harbor">
      <Link href={coastHref({ place: `port-${code}` })} className="nx-press nx-focus flex min-h-[52px] items-center justify-between gap-3 rounded-xl bg-n-ink px-4 text-[15.5px] font-semibold text-n-bg" data-testid="open-briefing">
        Open the {name} briefing <Icon name="chevronRight" />
      </Link>
      <Section title="What the evidence says now" testid="harbor-claims">
        {!ev || ev.status === "loading" ? (
          <Loading what="Reading the evidence for these waters…" />
        ) : ev.status === "error" ? (
          <Failed what="The evidence" message={ev.message} retry={retryEv} />
        ) : (
          <ul className="flex flex-col gap-4">
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
            <a href={harbor.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 text-[13px] font-semibold text-n-accent">
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
                  <span className="text-[14.5px] font-semibold">{n.station.name}</span>
                  <span className="nx-data text-[13px] text-n-ink-3">
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
    </div>
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
      <Section title="Latest samples" testid="station-latest">
        <ul className="flex flex-col gap-2">
          {st.latest.map((l) => (
            <li key={l.variable} className="flex items-baseline justify-between gap-3 text-[14.5px]">
              <span className="text-n-ink-2">{l.label}</span>
              <span className="text-right">
                <span className="nx-data font-semibold text-n-ink">{l.value == null ? "—" : l.qualifier === "reported_zero" ? "0" : l.value.toLocaleString("en-US", { maximumSignificantDigits: 3 })}</span> <span className="text-n-ink-3">{l.units}</span>
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
              <button key={x.id} type="button" role="radio" aria-checked={v === x.id} onClick={() => setV(x.id)} className={`nx-press nx-focus min-h-10 rounded-lg px-2.5 text-[13px] font-semibold ${v === x.id ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink-2"}`}>
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
          <a href={st.source_url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-10 items-center gap-1 text-[13.5px] font-semibold text-n-accent">
            SCCOOS ERDDAP record <Icon name="external" className="h-3.5 w-3.5" />
          </a>
          {st.request_url && (
            <a href={st.request_url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-10 items-center gap-1 text-[13.5px] font-semibold text-n-accent">
              Original data (CSV) <Icon name="download" className="h-3.5 w-3.5" />
            </a>
          )}
        </div>
      </Section>
    </div>
  );
}

// ------------------------------------------------------------------------------------- cell

function pct(v: number | null) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

export function CellBody({ read, ctx, variable }: { read: CellRead | null; ctx: Ctx; variable: CharmVariable }) {
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
        <p className="mt-1 text-[13px] text-n-ink-3">
          A published probability for one {read.sizeKm.toFixed(1)} km cell, not a toxin measurement
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
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-point">
      <p className="nx-data text-[13px] text-n-ink-3">
        {lat.toFixed(4)}°N {Math.abs(lon).toFixed(4)}°W
      </p>
      <button type="button" onClick={() => ctx.onSelect({ kind: "cell", lat, lon })} className="nx-press nx-focus flex min-h-12 items-center justify-between rounded-xl border border-n-line-strong px-3 text-[15px] font-semibold">
        What C-HARM publishes for this cell <Icon name="chevronRight" />
      </button>
      <Section title="Nearest shore stations">
        {near.near.length || near.nearestBeyond ? (
          <ul>
            {(near.near.length ? near.near.slice(0, 3) : [near.nearestBeyond!]).map((n) => (
              <li key={n.station.station_id}>
                <button type="button" onClick={() => ctx.onSelect({ kind: "station", id: n.station.station_id })} className="nx-focus flex min-h-12 w-full items-center justify-between gap-3 rounded-lg px-1 text-left hover:bg-n-surface-2">
                  <span className="text-[14.5px] font-semibold">{n.station.name}</span>
                  <span className="nx-data text-[13px] text-n-ink-3">
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

// ------------------------------------------------------------------------------------- area

export function areaStats(read: { grid: import("@/generated/schema").ValueGrid; codes: Uint16Array } | null, bbox: [number, number, number, number]) {
  if (!read) return null;
  const g = read.grid;
  const [w, s, e, n] = bbox;
  let domain = 0;
  const vals: number[] = [];
  for (let row = 0; row < g.height; row++) {
    const lat = g.lat_first + row * g.lat_step;
    if (lat < s || lat > n) continue;
    for (let col = 0; col < g.width; col++) {
      const lon = g.lon_first + col * g.lon_step;
      if (lon < w || lon > e) continue;
      domain++;
      const c = read.codes[row * g.width + col];
      if (c !== g.nodata) vals.push(c * g.scale_factor + g.add_offset);
    }
  }
  vals.sort((a, b) => a - b);
  const q = (p: number) => vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
  return { domain, valued: vals.length, min: vals[0] ?? null, median: vals.length ? q(0.5) : null, max: vals[vals.length - 1] ?? null };
}

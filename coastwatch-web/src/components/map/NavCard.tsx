"use client";

import { DEMO } from "@/lib/demo";
import { useState } from "react";
import type { Palette } from "@/generated/schema";
import type { PortIntel } from "@/generated/port_intel";
import { OFFICIAL_STATUS } from "@/content/copy";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { colorAt } from "@/components/ui/ProbabilityLegend";
import { Icon } from "@/components/ui/Icon";
import type { CharmVariable } from "@/lib/layers";

export type RegionDef = { id: string; label: string; bounds: [[number, number], [number, number]] };

type Props = {
  /** show the C-HARM forecast medians next to ports and regions (only while the forecast is on the map) */
  showForecast?: boolean;
  regions: RegionDef[]; // first entry is statewide
  region: string;
  onRegion: (id: string) => void;
  ports: PortIntel[];
  portsError: string | null;
  port: number | null;
  onPort: (code: number) => void;
  variable: CharmVariable;
  lead: number;
  palette: Palette | null | undefined;
  /** official notices that may apply in the selected region (record ids) */
  regionNotices: number | null;
};

const pct = (v: number) => `${Math.round(v * 100)}%`;

/**
 * Navigation (design reset rev. 2, §3.1): the official summary first, then places. Default
 * and region views list places; with a port open the card shrinks to a breadcrumb because
 * the port inspector carries the detail. Never more than one full panel per side.
 */
export function NavCard({ regions, region, onRegion, ports, portsError, port, onPort, variable, lead, palette, regionNotices, showForecast = true }: Props) {
  const { openDrawer, count, verification, available } = useOfficialDrawer();
  const [q, setQ] = useState("");
  const statewide = regions[0];
  const current = regions.find((r) => r.id === region) ?? statewide;
  // C-HARM forecast medians: shown only while the HAB forecast is on the map, so they are never read as chlorophyll or currents
  const median = (p: PortIntel) => (showForecast ? (p.charm?.leads.find((l) => l.lead_days === lead)?.variables[variable]?.median ?? null) : null);
  const inRegion = (id: string) => ports.filter((p) => p.region === id);
  const hits = q.trim() ? ports.filter((p) => p.display_name.toLowerCase().includes(q.trim().toLowerCase())) : [];
  const selected = port != null ? ports.find((p) => p.port_code === port) : null;
  const verWord = verification ? OFFICIAL_STATUS.verification[verification.state] : "Checking…";

  const portRow = (p: PortIntel) => {
    const m = median(p);
    return (
      <li key={p.port_code}>
        <button
          onClick={() => onPort(p.port_code)}
          data-testid={`port-row-${p.port_code}`}
          aria-current={port === p.port_code ? "true" : undefined}
          className={`grid w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[14px] ${showForecast ? "grid-cols-[1fr_72px_40px]" : "grid-cols-[1fr_auto]"} ${port === p.port_code ? "bg-navy-900 text-white" : "text-ink hover:bg-surface-2"}`}
        >
          <span className="truncate">{p.display_name}</span>
          {showForecast ? (
            <>
              <span className="relative h-1.5 overflow-hidden rounded-full bg-surface-3" aria-hidden>
                {m != null && palette && <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${m * 100}%`, background: colorAt(palette, m) }} />}
              </span>
              <span className="text-right font-semibold tabular">{m != null ? pct(m) : "—"}</span>
            </>
          ) : (
            <span aria-hidden className="text-ink-3">›</span>
          )}
        </button>
      </li>
    );
  };

  return (
    <section data-testid="nav-card" aria-label="Official notices and places" className="theme-paper flex max-h-full flex-col overflow-hidden rounded-2xl bg-surface text-ink shadow-[0_1px_2px_rgba(6,17,30,0.12),0_8px_24px_rgba(6,17,30,0.18)]">
      {port == null && (
        <button
          type="button"
          data-testid="official-summary"
          onClick={(e) => openDrawer(e.currentTarget)}
          aria-haspopup="dialog"
          className="grid grid-cols-[auto_1fr_auto] items-center gap-2 border-b border-official-line bg-official-bg px-3.5 py-2.5 text-left text-[14px] hover:brightness-[0.98]"
        >
          <Icon name="shield" className="h-[18px] w-[18px] text-official" />
          <span className="leading-snug">
            {available ? (
              region !== statewide.id && regionNotices != null ? (
                <>
                  <b className="font-semibold text-official-ink">
                    {regionNotices} official notice{regionNotices === 1 ? "" : "s"}
                  </b>{" "}
                  may apply in {current.label}
                </>
              ) : (
                <>
                  <b className="font-semibold text-official-ink">
                    {count} official notice{count === 1 ? "" : "s"}
                  </b>{" "}
                  {DEMO ? "listed here (list incomplete)" : "active in California"}
                </>
              )
            ) : (
              <b className="font-semibold text-official-ink">Official notices unavailable: check CDFW and CDPH</b>
            )}
          </span>
          <span className="flex items-center gap-1 text-[12px] font-medium text-official-ink">
            {available ? verWord : ""}
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
              <path d="m9 6 6 6-6 6" />
            </svg>
          </span>
        </button>
      )}
      <label className="mx-2.5 mb-1 mt-2.5 flex h-9 items-center gap-2 rounded-lg border border-hairline bg-surface-2 px-2.5 text-ink-3 focus-within:border-accent">
        <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
          <circle cx="11" cy="11" r="6.5" />
          <path d="m20 20-4.2-4.2" />
        </svg>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a port" aria-label="Find a port" data-testid="port-search" className="min-w-0 flex-1 bg-transparent text-[14px] text-ink outline-none placeholder:text-ink-3" />
      </label>

      <div className="min-h-0 overflow-y-auto px-1.5 pb-2">
        {q.trim() ? (
          hits.length ? <ul>{hits.map(portRow)}</ul> : <p className="px-3 py-2 text-[13px] text-ink-3">No port named “{q.trim()}”.</p>
        ) : selected ? (
          <button onClick={() => onRegion(selected.region ?? statewide.id)} data-testid="nav-crumb" className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left text-[14px] font-medium text-accent">
            <span aria-hidden>‹</span> {regions.find((r) => r.id === selected.region)?.label ?? statewide.label}
            <span className="ml-auto text-[12px] font-normal text-ink-3">{inRegion(selected.region ?? "").length} ports</span>
          </button>
        ) : region !== statewide.id ? (
          <>
            <button onClick={() => onRegion(statewide.id)} data-testid="nav-all" className="flex items-center gap-1.5 px-2.5 pt-1.5 text-[14px] font-medium text-accent">
              <span aria-hidden>‹</span> {statewide.label}
            </button>
            <h2 data-testid="nav-region-title" className="px-2.5 pb-1.5 pt-0.5 font-display text-[22px] font-medium leading-tight">
              {current.label}
            </h2>
            {portsError ? <p className="px-2.5 text-[13px] text-ink-3">Port summaries unavailable: {portsError}</p> : <ul data-testid="region-ports">{inRegion(region).map(portRow)}</ul>}
            {showForecast && <p className="px-2.5 pt-1.5 text-[12px] leading-snug text-ink-3">C-HARM forecast: median of model cells within 15 km of each port, not conditions at the dock.</p>}
          </>
        ) : (
          <>
            <p className="px-2.5 pb-1 pt-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-3">Coast, north to south</p>
            <ul>
              {regions.slice(1).map((r) => {
                const vals = inRegion(r.id)
                  .map(median)
                  .filter((v): v is number => v != null);
                return (
                  <li key={r.id}>
                    <button
                      onClick={() => onRegion(r.id)}
                      data-testid={`region-${r.id}`}
                      aria-pressed={region === r.id}
                      className="grid w-full grid-cols-[1fr_auto_auto] items-center gap-2.5 rounded-lg px-2.5 py-2 text-left hover:bg-surface-2"
                    >
                      <span className="text-[15px] font-medium">{r.label}</span>
                      <span className="text-[13px] font-medium text-ink-2 tabular">{showForecast ? (vals.length ? `${pct(Math.min(...vals))}–${pct(Math.max(...vals))}` : "—") : ""}</span>
                      <span aria-hidden className="text-ink-3">›</span>
                    </button>
                  </li>
                );
              })}
            </ul>
            {showForecast && <p className="px-2.5 pt-1 text-[12px] text-ink-3">C-HARM forecast: range of port medians for the selected day.</p>}
          </>
        )}
      </div>
    </section>
  );
}

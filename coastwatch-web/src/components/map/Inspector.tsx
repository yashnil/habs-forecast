"use client";

import { useEffect, useState } from "react";
import type { Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { FORECAST_COPY, OFFICIAL_STATUS } from "@/content/copy";
import { ACTION_LABEL, officialAt, type Verification } from "@/lib/official";
import { VerificationBadge } from "@/components/official/Official";
import { loadGrid, sample, type Sample } from "@/lib/grid";
import { CHARM_VARIABLES, artifactUrl, charmLayer, leadLabel } from "@/lib/layers";
import { formatDate } from "@/lib/time";
import { gradientCss } from "@/components/ui/ProbabilityLegend";

export type InspectPoint = { lat: number; lon: number };

type Row = { variable: string; title: string; threshold: string | null; sample: Sample | null; error?: string };

export function Inspector({
  manifest,
  baseUrl,
  point,
  lead,
  onClose,
  official,
  verification,
}: {
  manifest: Manifest;
  baseUrl: string;
  point: InspectPoint;
  lead: number;
  onClose: () => void;
  official: OfficialDataset | null;
  verification: Verification | null;
}) {
  const here = officialAt(official, point.lon, point.lat);
  const [rows, setRows] = useState<Row[] | null>(null);
  const layers = CHARM_VARIABLES.map((v) => charmLayer(manifest, v, lead));
  const first = layers.find(Boolean) ?? null;

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    Promise.all(
      CHARM_VARIABLES.map(async (v): Promise<Row> => {
        const l = charmLayer(manifest, v, lead);
        if (!l || !l.grid) return { variable: v, title: v, threshold: null, sample: null, error: "not issued" };
        try {
          const codes = await loadGrid(artifactUrl(baseUrl, l.grid.url));
          return { variable: v, title: l.short_title, threshold: l.threshold_text ?? null, sample: sample(l.grid, codes, point.lat, point.lon) };
        } catch (e) {
          return { variable: v, title: l.short_title, threshold: l.threshold_text ?? null, sample: null, error: (e as Error).message };
        }
      }),
    ).then((r) => !cancelled && setRows(r));
    return () => {
      cancelled = true;
    };
  }, [manifest, baseUrl, point.lat, point.lon, lead]);

  const palette = first?.palette;

  return (
    <section
      data-testid="inspector"
      aria-live="polite"
      className="w-full"
    >
      <header className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-accent">Point</p>
          <h3 className="text-[14px] font-semibold text-ink tabular">
            {point.lat.toFixed(3)}°N, {Math.abs(point.lon).toFixed(3)}°W
          </h3>
          {first?.time.valid_date && (
            <p className="mt-0.5 text-[11.5px] text-ink-2">
              C-HARM {leadLabel(lead).toLowerCase()} · valid {formatDate(first.time.valid_date, { year: true })}
            </p>
          )}
        </div>
        <button onClick={onClose} className="rounded-md px-2 py-1 text-[12px] text-ink-3 hover:bg-surface-3 hover:text-ink" aria-label="Close point details">
          ✕
        </button>
      </header>

      <div className="mt-3 space-y-1.5 rounded-md border border-[#ffb547]/30 bg-[#ffb547]/[0.04] p-2.5" data-testid="inspect-official">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[12px] font-semibold text-ink">Official notices here</p>
          <VerificationBadge v={verification} />
        </div>
        {!official ? (
          <p className="text-[11.5px] text-ink-2">Official notices unavailable; check CDFW and CDPH.</p>
        ) : (
          <ul className="space-y-1 text-[11.5px] text-ink-2">
            {here.map(({ record: r, reason }) => (
              <li key={r.id} data-testid={`inspect-notice-${r.id}`}>
                <span className="font-medium text-ink">{ACTION_LABEL[r.action]}:</span> {r.title} <span className="text-ink-3">({reason})</span>
              </li>
            ))}
            {here.some((h) => h.reason !== "statewide") && <li className="text-ink-3">{OFFICIAL_STATUS.areaNote}</li>}
          </ul>
        )}
        <p className="text-[11px] text-ink-3">{OFFICIAL_STATUS.missingNotOpen}</p>
      </div>

      {!first ? (
        <p className="mt-3 text-[12.5px] text-ink-2">No forecast is available for this valid day.</p>
      ) : !rows ? (
        <p className="mt-3 text-[12px] text-ink-3">Reading forecast values…</p>
      ) : (
        <ul className="mt-3 space-y-2.5">
          {rows.map((r) => (
            <li key={r.variable} data-testid={`inspect-${r.variable}`}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[12.5px] text-ink-2">{r.title}</span>
                {r.sample?.kind === "value" && r.sample.nearest ? (
                  <span className="text-[12px] text-ink-3 tabular" data-testid={`inspect-value-${r.variable}`} data-nearest="true">
                    {`${(r.sample.value * 100).toFixed(0)}% at ${r.sample.distanceKm.toFixed(1)} km`}
                  </span>
                ) : (
                  <span className="text-[14px] font-semibold text-ink tabular" data-testid={`inspect-value-${r.variable}`}>
                    {r.sample?.kind === "value" ? `${(r.sample.value * 100).toFixed(0)}%` : "—"}
                  </span>
                )}
              </div>
              {r.sample?.kind === "value" && !r.sample.nearest && palette && (
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-3">
                  <div className="h-full" style={{ width: `${Math.max(2, r.sample.value * 100)}%`, background: gradientCss(palette), backgroundSize: `${100 / Math.max(0.02, r.sample.value)}% 100%` }} />
                </div>
              )}
              <p className="mt-0.5 text-[11px] text-ink-3">
                {r.error
                  ? r.error === "not issued"
                    ? "Not issued in this run."
                    : `Could not read values: ${r.error}`
                  : r.sample?.kind === "value" && r.sample.nearest
                    ? `No forecast value at this point (nearshore cell not covered). Value shown is the nearest forecast cell, ${r.sample.distanceKm.toFixed(1)} km away.`
                    : r.sample?.kind === "none"
                      ? FORECAST_COPY.noValue
                      : r.threshold}
              </p>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 border-t border-hairline pt-2 text-[11px] leading-snug text-ink-3">
        {FORECAST_COPY.notA} {FORECAST_COPY.lowNotSafe}
      </p>
    </section>
  );
}


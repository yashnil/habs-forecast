"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { LayerArtifact, Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { AGENCY_LINKS, COAST_COPY } from "@/content/coast";
import { FORECAST_COPY, OFFICIAL_STATUS } from "@/content/copy";
import { ACTION_LABEL, currentRecords, type Verification } from "@/lib/official";
import { AUDIENCE, cdfwRegionsAt, type Association } from "@/lib/applicability";
import { ageDays, ageWords, availability, RECENCY_LABEL, recency, type Audience, type Availability, type CoastPlace, type StationNear } from "@/lib/coast";
import { CHARM_LEADS, artifactUrl, charmLayer, leadLabel } from "@/lib/layers";
import { loadGrid } from "@/lib/grid";
import { classifyTime } from "@/lib/freshness";
import { sourceCsvUrl } from "@/lib/observations";
import { formatDate } from "@/lib/time";
import { VerificationBadge } from "@/components/official/Official";
import { SatelliteNear, type SatelliteResult } from "@/components/map/SatelliteNear";
import { CurrentsNear } from "@/components/map/CurrentsNear";
import { useRegistryDisclosure } from "@/components/shell/OfficialShell";
import { currentsHourly } from "@/lib/currents";
import { Mark } from "@/components/ui/Mark";

// ---------------------------------------------------------------- shared bits

export function SectionHead({ id, kind, title, aside }: { id: string; kind: "official" | "measured" | "model" | "plain"; title: string; aside?: React.ReactNode }) {
  const dot = { official: "bg-official", measured: "bg-measured", model: "bg-model", plain: "bg-ink-3" }[kind];
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
      <h2 id={id} className="flex items-center gap-2.5 font-display text-[24px] font-medium leading-tight text-ink">
        <span aria-hidden className={`h-2.5 w-2.5 rounded-full ${dot}`} />
        {title}
      </h2>
      {aside}
    </div>
  );
}

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;
const linkCls = "text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent";
const fmtPct = (v: number) => `${Math.round(v * 100)}%`;
const dayWord = ageWords;

// ---------------------------------------------------------------- official

function NoticeItem({ a, muted = false }: { a: Association; muted?: boolean }) {
  const r = a.record;
  const src = r.sources[0];
  return (
    <li className={`border-t border-official-line/60 py-3 first:border-t-0 first:pt-0 ${muted ? "opacity-[0.92]" : ""}`} data-testid={`coast-notice-${r.id}`} data-tier={a.tier}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] font-medium">
        <span className="rounded border border-official-line px-1.5 py-px text-official-ink">{r.agency}</span>
        <span className="text-official-ink">{ACTION_LABEL[r.action]}</span>
        <span className="text-ink-2">{AUDIENCE[r.fishery].label}</span>
      </div>
      <p className="mt-1 text-[16px] font-semibold leading-snug text-ink">{r.title}</p>
      <p className="mt-0.5 text-[14px] leading-snug text-ink-2">
        {r.effective_date ? <>Since {formatDate(r.effective_date, { year: true })}</> : "Start date not published"} · {r.area.description}
        {src && (
          <>
            {" · "}
            <a href={src.url} {...ext} className={linkCls}>
              {r.agency} source ↗
            </a>
          </>
        )}
      </p>
      <p className="mt-1.5 text-[14px] leading-snug text-ink-2" data-testid="coast-notice-basis">
        <span className="font-medium text-ink">{a.tier === "geographic" ? "Why it applies here: " : a.tier === "unresolved" ? "Why it is unresolved: " : "Why it is listed: "}</span>
        {a.basis}
      </p>
      {a.caveats.map((c) => (
        <p key={c} className="mt-1 text-[14px] leading-snug text-official-ink">
          {c}
        </p>
      ))}
      <p className="mt-1 text-[13px] text-ink-3" data-testid="coast-notice-review">
        {a.review.state === "not_reviewed" ? "Match to these waters not reviewed" : `Match ${a.review.state} by ${a.review.reviewer} on ${a.review.reviewedAt}`}
      </p>
    </li>
  );
}

function AgencyCall({ phone, tel, label }: { phone: string; tel: string; label: string }) {
  return (
    <li>
      {label}: <a href={`tel:${tel}`} className={`${linkCls} font-medium tabular`}>{phone}</a>
    </li>
  );
}

export function OfficialSection({
  official,
  assocs,
  audience,
  place,
  verification,
}: {
  official: OfficialDataset | null;
  assocs: Association[];
  audience: Audience;
  place: CoastPlace;
  verification: Verification | null;
}) {
  const fits = (a: Association) => audience === "all" || AUDIENCE[a.record.fishery].groups.includes(audience);
  // a notice an agency release appears to lift is set apart, never listed as applying
  const disputed = assocs.filter((a) => a.dispute);
  const geo = assocs.filter((a) => a.tier === "geographic" && !a.dispute);
  const unresolved = assocs.filter((a) => a.tier === "unresolved" && !a.dispute);
  const outside = assocs.filter((a) => a.tier === "outside" && !a.dispute);
  const reg = useRegistryDisclosure();
  const active = currentRecords(official).length;
  const geoFit = geo.filter(fits);
  const geoOther = geo.filter((a) => !fits(a));
  const regions = cdfwRegionsAt(place.lat);
  const statements = official?.registry.statements ?? [];
  return (
    <section aria-labelledby="coast-official-h" className="space-y-4" data-testid="coast-official">
      <SectionHead id="coast-official-h" kind="official" title="Official information" aside={<VerificationBadge v={verification} />} />
      <p className="text-[14px] text-ink-2">{COAST_COPY.unreviewed} The notices themselves are transcribed from the agencies&apos; pages and not verified.</p>
      {reg.changed && (
        <p className="rounded-lg border border-hairline border-l-[3px] border-l-official-line bg-surface px-3 py-2 text-[14px] text-official-ink" role="note" data-testid="coast-registry-changed">
          {reg.reason} A notice may have been issued or lifted since; the agency pages below decide.
        </p>
      )}
      {disputed.length > 0 && (
        <div className="rounded-2xl border-2 border-dashed border-warning/60 bg-surface px-4 py-4 sm:px-5" data-testid="coast-notices-disputed" role="note">
          <h3 className="text-[13.5px] font-semibold text-warning">Reported lifted by the agency · awaiting review</h3>
          <p className="mt-1 text-[14px] text-ink-2">
            The agency has published a release that appears to lift {disputed.length === 1 ? "this notice" : "these notices"}. Until a person reviews it, CoastWatch does not show {disputed.length === 1 ? "it" : "them"} as in effect. Read the release.
          </p>
          <ul className="mt-3">
            {disputed.map((a) => (
              <li key={a.record.id} className="py-1" data-testid={`coast-disputed-${a.record.id}`}>
                <p className="text-[15px] font-semibold text-ink">{a.record.title}</p>
                <p className="mt-0.5 text-[14px] text-ink-2">
                  {a.dispute!.agency} release{" "}
                  <a href={a.dispute!.url} {...ext} className={linkCls}>
                    {a.dispute!.release}: “{a.dispute!.title}” ↗
                  </a>
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
      {!official ? (
        <p className="text-[15px] text-ink-2">The list of official notices is unavailable right now. Check CDFW and CDPH directly (below).</p>
      ) : (
        <div className="rounded-2xl border border-hairline border-l-[3px] border-l-official-line bg-surface px-4 py-4 sm:px-5">
          <h3 className="cw-kind text-[13.5px] font-semibold text-official-ink"><Mark kind="official" />Apply here by their official geography</h3>
          {geoFit.length ? (
            <ul className="mt-3" data-testid="coast-notices-geo">
              {geoFit.map((a) => (
                <NoticeItem key={a.record.id} a={a} />
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-[15px] text-ink-2" data-testid="coast-notices-geo-empty">
              None in CoastWatch&apos;s list{audience !== "all" ? " for this kind of fishing" : ""}. That is not an all-clear: the list may be incomplete, and it has not been verified.
            </p>
          )}
          {geoOther.length > 0 && (
            <details className="mt-3 border-t border-official-line/60 pt-3" data-testid="coast-notices-other">
              <summary className="cursor-pointer text-[14px] font-medium text-ink">
                {geoOther.length} more for other kinds of fishing or harvest
              </summary>
              <ul className="mt-2">
                {geoOther.map((a) => (
                  <NoticeItem key={a.record.id} a={a} />
                ))}
              </ul>
            </details>
          )}
          <p className="mt-3 border-t border-official-line/60 pt-3 text-[14px] text-ink-2">{OFFICIAL_STATUS.missingNotOpen}</p>
        </div>
      )}

      {unresolved.length > 0 && (
        <div className="rounded-2xl border border-official-line/70 bg-surface px-4 py-4 sm:px-5" data-testid="coast-notices-unresolved">
          <h3 className="cw-kind text-[13.5px] font-semibold text-official-ink"><Mark kind="official" />May apply: not settled by their official geography</h3>
          <p className="mt-1 text-[14px] text-ink-2">Their stated area does not show whether they cover these waters. Check each notice&apos;s agency source before fishing or harvesting here.</p>
          <ul className="mt-3">
            {unresolved.map((a) => (
              <NoticeItem key={a.record.id} a={a} muted />
            ))}
          </ul>
        </div>
      )}

      {outside.length > 0 && (
        <details className="rounded-2xl border border-hairline bg-surface px-4 py-3 sm:px-5" data-testid="coast-notices-outside">
          <summary className="cursor-pointer text-[15px] font-medium text-ink">
            {outside.length} nearby notice{outside.length === 1 ? "" : "s"} outside {outside.length === 1 ? "its" : "their"} official area
          </summary>
          <p className="mt-2 text-[14px] text-ink-2">By their stated boundaries these do not cover these waters. They are listed because they are close, for anyone who also fishes or harvests there.</p>
          <ul className="mt-3">
            {outside.map((a) => (
              <NoticeItem key={a.record.id} a={a} muted />
            ))}
          </ul>
        </details>
      )}

      {statements.length > 0 && (
        <details className="rounded-xl border border-hairline bg-surface px-4 py-3 text-[14px] text-ink-2">
          <summary className="cursor-pointer font-medium text-ink">Statewide agency statements (Dungeness crab, spiny lobster)</summary>
          <ul className="mt-2 space-y-1.5">
            {statements.map((s) => (
              <li key={s.id}>
                “{s.statement}” <span className="text-ink-3">— {s.agency},</span>{" "}
                <a href={s.source.url} {...ext} className={linkCls}>
                  source ↗
                </a>
              </li>
            ))}
          </ul>
        </details>
      )}

      <div className="rounded-2xl border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="coast-agencies">
        <h3 className="text-[16px] font-semibold text-ink">Check the rules with the agencies</h3>
        <p className="mt-1 text-[14px] text-ink-2" data-testid="coast-agencies-complete">
          CoastWatch matches notices to places automatically and may miss one. The agencies&apos; own lists are complete; CoastWatch&apos;s list has {active} active notice{active === 1 ? "" : "s"} statewide
          {" "}(
          <button type="button" onClick={(e) => reg.openDrawer(e.currentTarget)} className={`${linkCls} font-medium`} data-testid="coast-all-notices">
            see all
          </button>
          ).
        </p>
        <ul className="mt-2 space-y-1.5 text-[15px] text-ink-2">
          <li>
            <a href={AGENCY_LINKS.cdfwAdvisories.url} {...ext} className={linkCls}>
              {AGENCY_LINKS.cdfwAdvisories.label} ↗
            </a>
          </li>
          {(audience === "all" || audience === "recreational" || audience === "shellfish") &&
            [...regions.at, ...regions.near].map((r) => (
              <li key={r.id} data-testid={`coast-region-${r.id}`}>
                <a href={r.url} {...ext} className={linkCls}>
                  CDFW recreational regulations, {r.name} region ↗
                </a>{" "}
                <span className="text-ink-3">({r.limits}{regions.near.includes(r) ? "; these waters are near its limit" : ""})</span>
              </li>
            ))}
          {(audience === "all" || audience === "commercial") && (
            <li>
              <a href={AGENCY_LINKS.cdfwCommercial.url} {...ext} className={linkCls}>
                {AGENCY_LINKS.cdfwCommercial.label} ↗
              </a>
            </li>
          )}
          <li>
            <a href={AGENCY_LINKS.cdfwMpa.url} {...ext} className={linkCls}>
              {AGENCY_LINKS.cdfwMpa.label} ↗
            </a>
          </li>
          <li>
            <a href={AGENCY_LINKS.cdphAdvisories.url} {...ext} className={linkCls}>
              {AGENCY_LINKS.cdphAdvisories.label} ↗
            </a>{" "}
            ·{" "}
            <a href={AGENCY_LINKS.cdphMap.url} {...ext} className={linkCls}>
              advisory map ↗
            </a>
          </li>
          {(audience === "all" || audience === "shellfish") && (
            <li>
              <a href={AGENCY_LINKS.cdphResults.url} {...ext} className={linkCls}>
                {AGENCY_LINKS.cdphResults.label} ↗
              </a>{" "}
              ·{" "}
              <a href={AGENCY_LINKS.cdphRazorResults.url} {...ext} className={linkCls}>
                razor clams ↗
              </a>
            </li>
          )}
          <li>
            <a href={AGENCY_LINKS.oehha.url} {...ext} className={linkCls}>
              {AGENCY_LINKS.oehha.label} ↗
            </a>
          </li>
          <AgencyCall {...AGENCY_LINKS.cdphLine} />
          <AgencyCall {...AGENCY_LINKS.cdfwClosureLine} />
        </ul>
        <p className="mt-3 text-[14px] text-ink-3">{COAST_COPY.notAuthority}</p>
        {(audience === "all" || audience === "shellfish") && <p className="mt-1 text-[14px] text-ink-3">{COAST_COPY.noPsp}</p>}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- measured

const VAR_ORDER: { id: string; label: string }[] = [
  { id: "pDA", label: "Particulate domoic acid (toxin in seawater)" },
  { id: "pn_seriata", label: "Pseudo-nitzschia, large cells" },
  { id: "pn_delicatissima", label: "Pseudo-nitzschia, small cells" },
];

function fmtObs(value: number | null, units: string, qualifier: string | null) {
  if (value === null) return "—";
  if (qualifier === "reported_zero" || value === 0) return "reported 0 (not quantified)";
  if (units === "cells/L") return `${Math.round(value).toLocaleString("en-US")} cells/L`;
  return `${value < 1 ? value.toPrecision(2) : value.toFixed(2)} ${units === "ng/mL" ? "ng/mL" : units}`;
}

export function MeasuredSection({ near, nearestBeyond, now }: { near: StationNear[]; nearestBeyond: StationNear | null; now: Date | null }) {
  return (
    <section aria-labelledby="coast-measured-h" className="space-y-4" data-testid="coast-measured">
      <SectionHead id="coast-measured-h" kind="measured" title="Measured nearby" aside={<span className="text-[13px] text-ink-3">CalHABMAP shore stations · within 40 km</span>} />
      {near.length === 0 && nearestBeyond && (
        <p className="rounded-xl bg-surface-3 px-4 py-3 text-[15px] text-ink-2" data-testid="coast-measured-none">
          <b className="text-ink">No monitoring station within 40 km.</b> The nearest is {nearestBeyond.station.name}, {Math.round(nearestBeyond.km)} km {nearestBeyond.bearing}
          {nearestBeyond.station.latest.length ? (
            <>, last sampled {formatDate(nearestBeyond.station.latest.map((x) => x.date).sort().at(-1)!, { year: true })}</>
          ) : null}
          . There is no nearby toxin measurement for these waters.
        </p>
      )}
      {near.slice(0, 4).map(({ station: s, km, bearing }) => {
        const csv = sourceCsvUrl({ station_id: s.station_id, request_url: s.request_url ?? undefined });
        const n365 = Math.max(0, ...s.latest.map((x) => x.n_last_365d));
        return (
          <article key={s.station_id} className="rounded-2xl border border-hairline border-l-[3px] border-l-measured bg-surface px-4 py-4 sm:px-5" data-testid={`coast-station-${s.station_id}`}>
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <h3 className="text-[17px] font-semibold text-ink">{s.name}</h3>
              <span className="text-[14px] text-ink-2 tabular">
                {km < 1 ? "under 1" : km.toFixed(km < 10 ? 1 : 0)} km {bearing !== "here" ? bearing : ""}
              </span>
            </div>
            <p className="text-[13px] text-ink-3">Water sample at the pier · {n365} sample{n365 === 1 ? "" : "s"} in the last 12 months</p>
            <dl className="mt-2 divide-y divide-hairline text-[14px]">
              {VAR_ORDER.map((v) => {
                const m = s.latest.find((x) => x.variable === v.id);
                const days = m && now ? ageDays(m.date, now) : null;
                const rec = days != null ? recency(days) : null;
                return (
                  <div key={v.id} className="grid grid-cols-1 gap-x-4 py-2 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]" data-testid={`coast-obs-${s.station_id}-${v.id}`}>
                    <dt className="text-ink-2">{v.label}</dt>
                    <dd>
                      {m ? (
                        <>
                          <span className="font-semibold text-ink tabular">{fmtObs(m.value, m.units, m.qualifier)}</span>
                          <span className="block text-[13px] text-ink-3">
                            sampled {formatDate(m.date, { year: true })}
                            {days != null && <> · {dayWord(days)}</>}
                          </span>
                          {rec && rec !== "recent" && (
                            <span className={`block text-[13px] ${rec === "none_this_year" ? "font-medium text-[#9b2c2c]" : "text-ink-2"}`} data-testid="coast-obs-recency">
                              {RECENCY_LABEL[rec]}
                            </span>
                          )}
                        </>
                      ) : (
                        <span className="text-ink-3">not measured at this station</span>
                      )}
                    </dd>
                  </div>
                );
              })}
            </dl>
            <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14px]">
              <Link href={`/bloom?station=${encodeURIComponent(s.station_id)}`} className={linkCls}>
                History on Bloom Intelligence
              </Link>
              {csv && (
                <a href={csv} {...ext} className={linkCls}>
                  Source data (CSV) ↗
                </a>
              )}
            </p>
          </article>
        );
      })}
      <p className="text-[14px] text-ink-3">{COAST_COPY.pierNotHere} Laboratory results arrive days to weeks after the sample is taken.</p>
    </section>
  );
}

// ---------------------------------------------------------------- model

export type ModelRead = { lead: number; layer: LayerArtifact | null; pda: Availability | null; pn: Availability | null; error: string | null };

/** C-HARM availability around a location for one lead (particulate DA and Pseudo-nitzschia). */
export function useModel(manifest: Manifest, baseUrl: string, lat: number | null, lon: number | null, lead: number): ModelRead | null {
  const [r, setR] = useState<ModelRead | null>(null);
  useEffect(() => {
    let cancelled = false;
    setR(null);
    if (lat == null || lon == null) return;
    const pdaL = charmLayer(manifest, "particulate_domoic", lead);
    const pnL = charmLayer(manifest, "pseudo_nitzschia", lead);
    if (!pdaL?.grid || !pnL?.grid) {
      setR({ lead, layer: pdaL ?? pnL, pda: null, pn: null, error: "not issued" });
      return;
    }
    Promise.all([loadGrid(artifactUrl(baseUrl, pdaL.grid.url)), loadGrid(artifactUrl(baseUrl, pnL.grid.url))])
      .then(([a, b]) => !cancelled && setR({ lead, layer: pdaL, pda: availability(pdaL.grid!, a, lat, lon), pn: availability(pnL.grid!, b, lat, lon), error: null }))
      .catch((e) => !cancelled && setR({ lead, layer: pdaL, pda: null, pn: null, error: (e as Error).message }));
    return () => {
      cancelled = true;
    };
  }, [manifest, baseUrl, lat, lon, lead]);
  return r;
}

const fmtCell = (lat: number, lon: number) => `${lat.toFixed(3)}°N ${Math.abs(lon).toFixed(3)}°W`;

/**
 * One line on what C-HARM says about the chosen waters themselves (the summary and the section's
 * lead). Toxin and bloom grids are masked differently, so a cell can have one and not the other.
 */
export function modelHeadline(model: ModelRead | null): string {
  if (!model) return "Reading…";
  if (model.error) return "No C-HARM run available";
  const tox = !!model.pda?.here;
  const bloom = !!model.pn?.here;
  if (tox && bloom) return "C-HARM has values for these waters";
  if (bloom) return "No C-HARM toxin forecast for these waters; the bloom forecast has a value for their cell";
  if (tox) return "No C-HARM bloom forecast for these waters; the toxin forecast has a value for their cell";
  return "No C-HARM forecast for these waters";
}

function AvailRow({ title, threshold, a, valid }: { title: string; threshold: string | null; a: Availability; valid: string | null }) {
  return (
    <div className="py-3" data-testid={`coast-model-${title.startsWith("Particulate") ? "pda" : "pn"}`}>
      <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
      {threshold && <p className="text-[13px] text-ink-3">{threshold}</p>}
      {a.here ? (
        <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[14px]">
          <dt className="text-ink-3">At these waters</dt>
          <dd className="text-ink" data-testid="coast-model-here">
            <span className="font-semibold tabular">{fmtPct(a.here.value)}</span> <span className="text-ink-3">in the 3 km cell that contains them{valid ? `, valid ${valid}` : ""}</span>
          </dd>
        </dl>
      ) : (
        <>
          <p className="mt-1.5 text-[14px] text-ink" data-testid="coast-model-here">
            <b className="font-semibold">No forecast for these waters.</b> <span className="text-ink-2">The model has no value in the cell that contains them.</span>
          </p>
          {a.nearest && (
            <div className="mt-2 rounded-lg border border-hairline px-3 py-2.5 text-[14px]" data-testid="coast-model-nearest">
              <p className="cw-kind text-model-ink"><Mark kind="model" />Nearest cell with a value: not a forecast for these waters</p>
              <p className="mt-1 text-ink-2">
                <span className="font-semibold text-ink tabular">{fmtPct(a.nearest.value)}</span> at the cell centred {a.nearest.km.toFixed(1)} km {a.nearest.bearing} of here (
                <span className="tabular">{fmtCell(a.nearest.cell.lat, a.nearest.cell.lon)}</span>){valid ? `, valid ${valid}` : ""}.
              </p>
              <p className="mt-1 text-ink-3">It describes that 3 km ocean cell only. Conditions at the shore, in harbors and at piers can differ, and C-HARM does not model them.</p>
            </div>
          )}
        </>
      )}
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[14px]">
        <dt className="text-ink-3">Within {a.region?.radiusKm ?? 15} km</dt>
        <dd className="text-ink-2" data-testid="coast-model-region">
          {a.region ? (
            <>
              median <span className="font-medium text-ink tabular">{fmtPct(a.region.median)}</span>, range {fmtPct(a.region.min)}–{fmtPct(a.region.max)} over {a.region.n} cells{a.here ? "" : ", none of them these waters"}
            </>
          ) : (
            "no cells with a value"
          )}
        </dd>
      </dl>
    </div>
  );
}

const validLabel = (l: LayerArtifact | null | undefined) => (l?.time.valid_date ? formatDate(l.time.valid_date, { year: true }) : null);

export function ModelSection({ manifest, model, lead, onLead, now }: { manifest: Manifest; model: ModelRead | null; lead: number; onLead: (l: number) => void; now: Date | null }) {
  const layers = CHARM_LEADS.map((l) => charmLayer(manifest, "particulate_domoic", l));
  const layer = layers[lead];
  const fresh = layer && now ? classifyTime(layer.freshness, layer.time, now) : null;
  const pnLayer = charmLayer(manifest, "pseudo_nitzschia", lead);
  return (
    <section aria-labelledby="coast-model-h" className="space-y-3" data-testid="coast-model">
      <SectionHead id="coast-model-h" kind="model" title="Forecast availability" aside={<span className="text-[13px] text-ink-3">C-HARM v3.1 · NOAA · 3 km cells</span>} />
      <div role="group" aria-label="Forecast day" className="flex flex-wrap gap-1.5">
        {CHARM_LEADS.map((l) => {
          const ly = layers[l];
          return (
            <button
              key={l}
              type="button"
              disabled={!ly}
              aria-pressed={l === lead}
              onClick={() => onLead(l)}
              className={`min-h-[40px] rounded-lg border px-3 text-left text-[13px] leading-tight disabled:opacity-40 ${l === lead ? "border-model bg-model-bg text-model-ink" : "border-hairline-strong text-ink-2 hover:border-ink-3"}`}
              data-testid={`coast-lead-${l}`}
            >
              <span className="block font-semibold">{ly?.time.valid_date ? formatDate(ly.time.valid_date) : "—"}</span>
              {leadLabel(l)}
            </button>
          );
        })}
      </div>
      {layer && (
        <p className="text-[14px] text-ink-2" data-testid="coast-model-run">
          Run issued {layer.time.issued_date ? formatDate(layer.time.issued_date, { year: true }) : "—"}
          {layer.time.issued_date_derived ? " (date inferred: NOAA does not publish it)" : ""} · valid {layer.time.valid_date ? formatDate(layer.time.valid_date, { year: true }) : "—"}
          {fresh && fresh.state !== "current" && (
            <span className="ml-1 font-medium text-[#9b2c2c]" data-testid="coast-model-stale">
              · {fresh.state === "stale" ? "Stale" : "Historical"}: the latest run is {dayWord(fresh.ageDays ?? 0)}
            </span>
          )}
        </p>
      )}
      <div className="rounded-2xl border border-hairline border-l-[3px] border-l-model bg-surface px-4 sm:px-5" aria-live="polite">
        {!model ? (
          <p className="py-4 text-[14px] text-ink-3">Reading the forecast grid…</p>
        ) : model.error ? (
          <p className="py-4 text-[14px] text-ink-2" data-testid="coast-model-error">
            {model.error === "not issued" ? "No C-HARM run is available for this day." : `The forecast grid could not be read (${model.error}).`}
          </p>
        ) : (
          <div className="divide-y divide-hairline">
            {!(model.pda?.here && model.pn?.here) && (
              <p className="py-3 font-display text-[19px] font-medium leading-snug text-ink" data-testid="coast-model-none">
                {modelHeadline(model)}.
              </p>
            )}
            {model.pda && <AvailRow title="Particulate domoic acid" threshold={layer?.threshold_text ?? null} a={model.pda} valid={validLabel(layer)} />}
            {model.pn && <AvailRow title="Pseudo-nitzschia bloom" threshold={pnLayer?.threshold_text ?? null} a={model.pn} valid={validLabel(pnLayer)} />}
          </div>
        )}
      </div>
      <p className="text-[14px] text-ink-3">
        {COAST_COPY.modelNotHere} {FORECAST_COPY.lowNotSafe} The map shows the cell used, outlined, and a dashed line to it.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- observed

export function ObservedSection({ manifest, baseUrl, place, onSat }: { manifest: Manifest; baseUrl: string; place: CoastPlace; onSat?: (r: SatelliteResult) => void }) {
  const hourly = currentsHourly(manifest);
  const latest = hourly.at(-1) ?? null;
  return (
    <section aria-labelledby="coast-observed-h" className="space-y-3" data-testid="coast-observed">
      <SectionHead id="coast-observed-h" kind="measured" title="Recent observations" aside={<span className="text-[13px] text-ink-3">Satellite · HF radar</span>} />
      <div className="space-y-4 rounded-2xl border border-hairline bg-surface px-4 py-4 sm:px-5 [&_h3]:text-[15px]">
        <SatelliteNear manifest={manifest} baseUrl={baseUrl} lat={place.lat} lon={place.lon} place={place.name} onRead={onSat} />
        <div className="border-t border-hairline pt-4">
          <CurrentsNear layer={latest} baseUrl={baseUrl} lat={place.lat} lon={place.lon} place={place.name} />
          {!latest && <p className="text-[14px] text-ink-3">No observed currents are published in this dataset.</p>}
        </div>
      </div>
      <p className="text-[14px] text-ink-3">
        {COAST_COPY.chlNotToxin} {COAST_COPY.currentsNotForecast}
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- unknowns

export function UnknownsSection({ items }: { items: string[] }) {
  return (
    <section aria-labelledby="coast-unknown-h" className="space-y-3" data-testid="coast-unknowns">
      <SectionHead id="coast-unknown-h" kind="plain" title="What CoastWatch does not know here" />
      <ul className="list-disc space-y-1.5 pl-5 text-[15px] leading-relaxed text-ink-2">
        {items.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </section>
  );
}

export { leadLabel };

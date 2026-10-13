"use client";

import { useEffect, useState } from "react";
import type { LayerArtifact } from "@/generated/schema";
import { SPECIES, recordNamesSpecies, speciesPhrase, type Species, type SpeciesId } from "@/content/species";
import { AUDIENCE, type Association } from "@/lib/applicability";
import { AUDIENCES, type Audience } from "@/lib/coast";
import { compass, marineHeadline, type Conditions } from "@/lib/conditions";
import type { Step } from "@/lib/nextSteps";
import { ACTION_LABEL, type Verification } from "@/lib/official";
import { formatDate, formatDateTimePT } from "@/lib/time";
import { ageWords } from "@/lib/coast";
import { leadLabel } from "@/lib/layers";

/**
 * My Coast's briefing (Coastal mode): the place's official status, next steps and conditions,
 * each short, each sourced and dated. The scientific sections stay one tap away.
 */

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;
const link = "text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent";
export const fmtWhen = (s: string) => (/^\d{4}-\d{2}-\d{2}$/.test(s) ? formatDate(s, { year: true }) : formatDateTimePT(s));

function Eyebrow({ children, tone = "ink" }: { children: React.ReactNode; tone?: "ink" | "official" | "accent" }) {
  const c = { ink: "text-ink-3", official: "text-official-ink", accent: "text-accent" }[tone];
  return <p className={`text-[12px] font-semibold uppercase tracking-[0.09em] ${c}`}>{children}</p>;
}

// ---------------------------------------------------------------- conditions (live, from NOAA)

export function useConditions(lat: number | null, lon: number | null): { data: Conditions | null; error: string | null; loading: boolean } {
  const [s, setS] = useState<{ data: Conditions | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: false });
  useEffect(() => {
    if (lat == null || lon == null) return;
    let cancelled = false;
    setS({ data: null, error: null, loading: true });
    fetch(`/api/conditions?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: Conditions) => !cancelled && setS({ data: d, error: null, loading: false }))
      .catch((e: Error) => !cancelled && setS({ data: null, error: e.message, loading: false }));
    return () => {
      cancelled = true;
    };
  }, [lat, lon]);
  return s;
}

// ---------------------------------------------------------------- plan: activity, species, day

export function PlanBar({
  audience,
  onAudience,
  species,
  onSpecies,
  leads,
  lead,
  onLead,
}: {
  audience: Audience;
  onAudience: (a: Audience) => void;
  species: Species;
  onSpecies: (s: SpeciesId) => void;
  leads: (LayerArtifact | null)[];
  lead: number;
  onLead: (l: number) => void;
}) {
  const options = SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience === "shellfish" ? "shellfish" : audience));
  return (
    <div className="space-y-2.5" data-testid="coast-plan">
      <div role="group" aria-label="Activity" className="-mx-4 flex gap-1.5 overflow-x-auto px-4 pb-0.5 sm:mx-0 sm:flex-wrap sm:px-0" data-testid="coast-audience">
        {AUDIENCES.map((x) => (
          <button
            key={x.id}
            type="button"
            aria-pressed={audience === x.id}
            onClick={() => onAudience(x.id)}
            className={`min-h-[40px] shrink-0 rounded-full border px-3.5 text-[14px] ${audience === x.id ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong bg-surface text-ink-2 hover:border-ink-3"}`}
            data-testid={`coast-for-${x.id}`}
          >
            {{ all: "Any activity", commercial: "Commercial", recreational: "Recreational", shellfish: "Shellfish" }[x.id]}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex min-h-[40px] items-center gap-2 rounded-full border border-hairline-strong bg-surface pl-3.5 pr-2 text-[14px] text-ink-2">
          <span>Species</span>
          <select
            value={species.id}
            onChange={(e) => onSpecies(e.target.value as SpeciesId)}
            className="min-h-[36px] max-w-[52vw] bg-transparent pr-1 font-medium text-ink outline-none"
            data-testid="coast-species"
          >
            {options.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <div role="group" aria-label="Toxin forecast day" className="flex items-center gap-1" data-testid="coast-day">
          <span className="mr-1 text-[13px] text-ink-3 max-[420px]:sr-only">Forecast day</span>
          {leads.map((l, i) =>
            l?.time.valid_date ? (
              <button
                key={i}
                type="button"
                aria-pressed={lead === i}
                onClick={() => onLead(i)}
                title={leadLabel(i)}
                className={`min-h-[40px] rounded-full border px-3 text-[13px] ${lead === i ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong bg-surface text-ink-2 hover:border-ink-3"}`}
                data-testid={`coast-day-${i}`}
              >
                {formatDate(l.time.valid_date).replace(/^\w+, /, "")}
              </button>
            ) : null,
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- official status

const fits = (a: Association, audience: Audience) => audience === "all" || AUDIENCE[a.record.fishery].groups.includes(audience);

export function OfficialBrief({
  assocs,
  audience,
  species,
  verification,
  totalActive,
  onAll,
  onDetails,
}: {
  assocs: Association[];
  audience: Audience;
  species: Species;
  verification: Verification | null;
  totalActive: number;
  onAll: (el: HTMLElement) => void;
  onDetails: () => void;
}) {
  const live = assocs.filter((a) => !a.dispute);
  const applies = live.filter((a) => a.tier === "geographic" && fits(a, audience));
  const mine = applies.filter((a) => recordNamesSpecies(a.record.species, species));
  const others = applies.filter((a) => !mine.includes(a));
  const unresolved = live.filter((a) => a.tier === "unresolved" && fits(a, audience) && recordNamesSpecies(a.record.species, species));
  const outside = live.filter((a) => a.tier === "outside").length;
  const disputed = assocs.filter((a) => a.dispute);
  const forWhat = species.id === "any" ? (audience === "all" ? "" : ` for ${AUDIENCES.find((x) => x.id === audience)!.label.toLowerCase()}`) : ` for ${speciesPhrase(species)}`;
  return (
    <section aria-labelledby="brief-official-h" className="rounded-[18px] border border-official-line bg-official-bg/60 px-4 py-4 sm:px-5" data-testid="brief-official">
      <div className="flex items-center justify-between gap-3">
        <Eyebrow tone="official">Official notices · these waters</Eyebrow>
        <span className="rounded-full border border-official-line px-2 py-0.5 text-[12px] font-medium text-official-ink" data-testid="brief-verification">
          {verification?.state === "verified" ? "Checked by a person" : "Not verified"}
        </span>
      </div>
      <h2 id="brief-official-h" className="mt-2 font-display text-[22px] font-medium leading-snug text-ink sm:text-[24px]" data-testid="brief-official-headline">
        {mine.length
          ? `${mine.length} notice${mine.length === 1 ? " applies" : "s apply"} here${forWhat}`
          : `No notice in CoastWatch's list applies here${forWhat}`}
      </h2>
      {!mine.length && <p className="mt-1 text-[14px] text-ink-2">That is not an all-clear: the list may be incomplete. Check the agencies (next steps).</p>}
      {(mine.length > 0 || disputed.length > 0) && (
        <ul className="mt-3 divide-y divide-official-line/50">
          {disputed.map((a) => (
            <li key={a.record.id} className="py-2.5" data-testid={`brief-disputed-${a.record.id}`}>
              <p className="text-[12px] font-semibold text-warning">Reported lifted by {a.dispute!.agency} · awaiting review</p>
              <p className="text-[15px] font-medium leading-snug text-ink">{a.record.title}</p>
              <a href={a.dispute!.url} {...ext} className={`${link} text-[13px]`}>
                {a.dispute!.agency} {a.dispute!.release} ↗
              </a>
            </li>
          ))}
          {mine.map((a) => (
            <BriefRow key={a.record.id} a={a} />
          ))}
        </ul>
      )}
      {unresolved.length > 0 && (
        <p className="mt-2 text-[14px] text-ink-2" data-testid="brief-unresolved">
          <span className="font-medium text-official-ink">May apply, not settled by its stated area:</span> {unresolved.map((a) => a.record.title).join("; ")}.
        </p>
      )}
      <p className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-official-line/50 pt-2.5 text-[13px] text-ink-2">
        {others.length > 0 && (
          <button type="button" onClick={onDetails} className={link} data-testid="brief-others">
            {others.length} more for other species here
          </button>
        )}
        {outside > 0 && (
          <button type="button" onClick={onDetails} className={link}>
            {outside} nearby, outside {outside === 1 ? "its" : "their"} area
          </button>
        )}
        <button type="button" onClick={(e) => onAll(e.currentTarget)} className={link} data-testid="brief-all">
          All {totalActive} notices in California
        </button>
      </p>
    </section>
  );
}

function BriefRow({ a }: { a: Association }) {
  const r = a.record;
  const src = r.sources.find((s) => s.url.startsWith("https://"));
  return (
    <li className="py-2.5" data-testid={`brief-notice-${r.id}`}>
      <p className="text-[12px] font-semibold text-official-ink">
        {r.agency} · {ACTION_LABEL[r.action]} · {AUDIENCE[r.fishery].label}
      </p>
      <p className="text-[15px] font-medium leading-snug text-ink">{r.title}</p>
      <p className="mt-0.5 text-[13px] text-ink-3">
        {r.effective_date ? `Since ${formatDate(r.effective_date, { year: true })}` : "Start date not published"}
        {src && (
          <>
            {" · "}
            <a href={src.url} {...ext} className={link}>
              {r.agency} source ↗
            </a>
          </>
        )}
      </p>
      {a.caveats[0] && <p className="mt-0.5 text-[13px] text-official-ink">{a.caveats[0]}</p>}
    </li>
  );
}

// ---------------------------------------------------------------- next steps

const KIND: Record<Step["kind"], { dot: string; label: string }> = {
  official: { dot: "bg-official", label: "Official" },
  conditions: { dot: "bg-accent", label: "Conditions" },
  gap: { dot: "bg-ink-3", label: "Not known here" },
};

export function NextSteps({ steps, max = 7 }: { steps: Step[]; max?: number }) {
  const [all, setAll] = useState(false);
  const shown = all ? steps : steps.slice(0, max);
  return (
    <section aria-labelledby="next-h" className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="next-steps">
      <Eyebrow>Before you go</Eyebrow>
      <h2 id="next-h" className="mt-1 font-display text-[22px] font-medium leading-snug text-ink">
        Next steps
      </h2>
      <ol className="mt-3 space-y-3.5">
        {shown.map((s) => (
          <li key={s.id} className="grid grid-cols-[14px_1fr] gap-x-2.5" data-testid={`step-${s.id}`} data-kind={s.kind}>
            <span aria-hidden className={`mt-[7px] h-2 w-2 rounded-full ${KIND[s.kind].dot}`} />
            <div>
              <p className="text-[15px] font-medium leading-snug text-ink">
                <span className="sr-only">{KIND[s.kind].label}: </span>
                {s.title}
              </p>
              <p className="mt-0.5 text-[13px] leading-snug text-ink-2">{s.reason}</p>
              <p className="mt-0.5 text-[13px] text-ink-3">
                {s.when ? `${s.when} · ` : ""}
                {"url" in s.source ? (
                  <a href={s.source.url} {...ext} className={link}>
                    {s.source.label} ↗
                  </a>
                ) : (
                  <a href={`tel:${s.source.tel}`} className={`${link} tabular`}>
                    {s.source.label}
                  </a>
                )}
              </p>
            </div>
          </li>
        ))}
      </ol>
      {steps.length > max && (
        <button type="button" onClick={() => setAll(!all)} className="mt-3 text-[14px] font-medium text-accent hover:underline" data-testid="next-steps-more">
          {all ? "Fewer steps" : `${steps.length - max} more steps`}
        </button>
      )}
      <p className="mt-3 border-t border-hairline pt-2.5 text-[13px] text-ink-3">
        <span className="mr-3 inline-flex items-center gap-1.5"><span aria-hidden className="h-2 w-2 rounded-full bg-official" />Official</span>
        <span className="mr-3 inline-flex items-center gap-1.5"><span aria-hidden className="h-2 w-2 rounded-full bg-accent" />Conditions</span>
        <span className="inline-flex items-center gap-1.5"><span aria-hidden className="h-2 w-2 rounded-full bg-ink-3" />Not known here</span>
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- conditions

/** "Monterey Canyon Outer, CA  (156)" → "Monterey Canyon Outer"; "MONTEREY, MONTEREY BAY" → "Monterey, Monterey Bay". */
export function stationName(n: string): string {
  const s = n.replace(/\s*\(\d+\)\s*$/, "").replace(/,\s*CA\s*$/i, "").replace(/\s*-\s*\d+\s*NM.*$/i, "").trim();
  return s === s.toUpperCase() ? s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()) : s;
}

export type ToxinNear = { station: string; km: number; date: string; ageDays: number; value: string } | null;

function Tile({ label, value, sub, state, testid, href }: { label: string; value: React.ReactNode; sub: React.ReactNode; state: "ok" | "old" | "none"; testid: string; href?: string }) {
  const dot = { ok: "bg-measured", old: "bg-warning", none: "bg-ink-3/50" }[state];
  return (
    <div className="min-w-0" data-testid={testid} data-state={state}>
      <p className="flex items-center gap-1.5 text-[13px] text-ink-3">
        <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        {label}
      </p>
      <p className="mt-0.5 text-[16px] font-medium leading-snug text-ink">{value}</p>
      <p className="mt-0.5 text-[12.5px] leading-snug text-ink-3">
        {sub}
        {href && (
          <>
            {" · "}
            <a href={href} {...ext} className={link}>
              source ↗
            </a>
          </>
        )}
      </p>
    </div>
  );
}

export function ConditionsCard({
  conditions,
  loading,
  error,
  toxin,
  toxinError = null,
  forecastLine,
  forecastSub,
  satelliteLine,
  now,
}: {
  conditions: Conditions | null;
  loading: boolean;
  error: string | null;
  toxin: ToxinNear;
  toxinError?: string | null;
  forecastLine: string;
  forecastSub: string;
  satelliteLine: string;
  now: Date | null;
}) {
  const c = conditions;
  const wait = loading ? "Loading…" : error ? "Unavailable right now" : "—";
  const ageH = (iso: string) => (now ? (now.getTime() - new Date(iso).getTime()) / 3_600_000 : 0);
  const nextTides = (() => {
    if (c?.tides.status !== "ok" || !now) return null;
    const nowLocal = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(now).replace(",", "");
    return c.tides.data.filter((t) => t.at >= nowLocal).slice(0, 2);
  })();
  const tideTime = (t: string) => {
    const [d, hm] = t.split(" ");
    const [h, m] = hm.split(":").map(Number);
    return `${formatDate(d)} ${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
  };
  return (
    <section aria-labelledby="cond-h" className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="conditions">
      <Eyebrow>Context for planning · not a toxin or fishing forecast</Eyebrow>
      <h2 id="cond-h" className="mt-1 font-display text-[22px] font-medium leading-snug text-ink">
        Conditions
      </h2>
      {c?.alerts.status === "ok" && c.alerts.data.length > 0 && (
        <p className="mt-2 rounded-lg bg-warning/10 px-3 py-2 text-[14px] font-medium text-warning" data-testid="cond-alert">
          {c.alerts.data[0].event}
          {c.alerts.data[0].ends ? ` until ${formatDateTimePT(c.alerts.data[0].ends)}` : ""}
          <span className="font-normal text-ink-2"> · NWS {c.marine.status === "ok" ? c.marine.data.zoneName : ""}</span>
        </p>
      )}
      <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-4 min-[480px]:grid-cols-2">
        {c?.marine.status === "ok" ? (
          <Tile
            label={`Marine forecast · ${c.marine.data.periods[0].name}`}
            value={marineHeadline(c.marine.data.periods[0])}
            sub={`NWS ${c.marine.data.zone}${c.marine.data.updated ? `, issued ${formatDateTimePT(c.marine.data.updated)}` : ""}`}
            state="ok"
            href={c.marine.url}
            testid="cond-marine"
          />
        ) : (
          <Tile label="Marine forecast" value={c ? "Not available" : wait} sub={c && "reason" in c.marine ? c.marine.reason : "NWS"} state="none" testid="cond-marine" />
        )}
        {c?.buoy.status === "ok" && c.buoy.station ? (
          <Tile
            label={`Measured · ${stationName(c.buoy.station.name)}`}
            value={[
              c.buoy.data.windKt ? `Wind ${Math.round(c.buoy.data.windKt.value)} kt${c.buoy.data.windDirDeg ? ` ${compass(c.buoy.data.windDirDeg.value)}` : ""}` : null,
              c.buoy.data.waveFt ? `waves ${c.buoy.data.waveFt.value.toFixed(1)} ft` : null,
              c.buoy.data.waterC ? `water ${c.buoy.data.waterC.value.toFixed(1)} °C` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
            sub={`NDBC ${c.buoy.station.id}, ${Math.round(c.buoy.station.km)} km away, ${formatDateTimePT(Object.values(c.buoy.data)[0]!.at)}`}
            state={ageH(Object.values(c.buoy.data)[0]!.at) > 3 ? "old" : "ok"}
            href={c.buoy.url}
            testid="cond-buoy"
          />
        ) : (
          <Tile label="Measured nearby" value={c ? "No recent buoy reading" : wait} sub={c && "reason" in c.buoy ? c.buoy.reason : "NDBC"} state="none" testid="cond-buoy" />
        )}
        {nextTides?.length ? (
          <Tile
            label={`Tides · ${c!.tides.status === "ok" && c!.tides.station ? stationName(c!.tides.station.name) : ""}`}
            value={nextTides.map((t) => `${t.type === "H" ? "High" : "Low"} ${t.ft.toFixed(1)} ft ${tideTime(t.at).split(" ").slice(-2).join(" ")}`).join(" · ")}
            sub={`Predicted, ${tideTime(nextTides[0].at).split(" ").slice(0, 3).join(" ")} · NOAA CO-OPS, feet above MLLW`}
            state="ok"
            href={c!.tides.status === "ok" ? c!.tides.url : undefined}
            testid="cond-tides"
          />
        ) : (
          <Tile label="Tides" value={c ? "Not available" : wait} sub={c && "reason" in c.tides ? c.tides.reason : "NOAA CO-OPS"} state="none" testid="cond-tides" />
        )}
        <Tile
          label="Toxin in seawater · nearest sample"
          value={toxin ? `${toxin.value}, ${ageWords(toxin.ageDays)}` : toxinError ? "Station data unavailable" : "No sample within 40 km"}
          sub={toxin ? `${toxin.station}, ${toxin.km < 10 ? toxin.km.toFixed(1) : Math.round(toxin.km)} km, sampled ${formatDate(toxin.date, { year: true })} · CalHABMAP` : "CalHABMAP shore stations"}
          state={!toxin ? "none" : toxin.ageDays > 30 ? "old" : "ok"}
          testid="cond-toxin"
        />
        <Tile label="Toxin forecast" value={forecastLine} sub={forecastSub} state={/^No /.test(forecastLine) ? "none" : "ok"} testid="cond-forecast" />
        <Tile label="Satellite view of the water" value={satelliteLine} sub="NOAA CoastWatch chlorophyll (algae, not toxin)" state={/^Clear/.test(satelliteLine) ? "ok" : "none"} testid="cond-satellite" />
      </div>
    </section>
  );
}

"use client";

import { useEffect, useState } from "react";
import NextLink from "next/link";
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
import { Mark } from "@/components/ui/Mark";
import { AnimatePresence, motion } from "motion/react";

/**
 * My Coast's briefing (Coastal mode): the place's official status, next steps and conditions,
 * each short, each sourced and dated. The scientific sections stay one tap away.
 */

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;
const link = "text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent";
export const fmtWhen = (s: string) => (/^\d{4}-\d{2}-\d{2}$/.test(s) ? formatDate(s, { year: true }) : formatDateTimePT(s));

/** A kind label: the mark of how it is known, then sentence-case words (signature P5). */
function Eyebrow({ children, tone = "ink" }: { children: React.ReactNode; tone?: "ink" | "official" | "accent" }) {
  const c = {
    ink: "text-ink-3",
    official: "text-official-ink",
    accent: "text-accent",
  }[tone];
  const mark = tone === "official" ? "official" : tone === "accent" ? "conditions" : null;
  return (
    <p className={`cw-kind ${c}`}>
      {mark && <Mark kind={mark} />}
      <span>{children}</span>
    </p>
  );
}

// ---------------------------------------------------------------- conditions (live, from NOAA)

export function useConditions(lat: number | null, lon: number | null): { data: Conditions | null; error: string | null; loading: boolean } {
  const [s, setS] = useState<{
    data: Conditions | null;
    error: string | null;
    loading: boolean;
  }>({ data: null, error: null, loading: false });
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

// ---------------------------------------------------------------- plan: activity and species

/**
 * What the reader is after: two compact selects in one row, after the answer (signature P2: the
 * plan refines the answer; it is not a gate before it). The forecast day lives with the forecast
 * (ConditionsCard), the one place it changes anything.
 */
export function PlanBar({
  audience,
  onAudience,
  species,
  onSpecies,
}: {
  audience: Audience;
  onAudience: (a: Audience) => void;
  species: Species;
  onSpecies: (s: SpeciesId) => void;
}) {
  const options = SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience === "shellfish" ? "shellfish" : audience));
  const box = "flex min-h-[44px] min-w-0 items-center gap-2 rounded-xl border border-hairline-strong bg-surface pl-3 pr-1.5 text-[14px] text-ink-3 focus-within:border-ink-3";
  return (
    <div className="grid grid-cols-2 gap-2" data-testid="coast-plan">
      <label className={box}>
        <span className="shrink-0 max-[520px]:sr-only">Activity</span>
        <select
          value={audience}
          onChange={(e) => onAudience(e.target.value as Audience)}
          className="min-h-[40px] min-w-0 flex-1 bg-transparent font-medium text-ink outline-none"
          data-testid="coast-audience"
        >
          {AUDIENCES.map((x) => (
            <option key={x.id} value={x.id}>
              {
                {
                  all: "Any activity",
                  commercial: "Commercial",
                  recreational: "Recreational",
                  shellfish: "Shellfish",
                }[x.id]
              }
            </option>
          ))}
        </select>
      </label>
      <label className={box}>
        <span className="shrink-0 max-[520px]:sr-only">Species</span>
        <select value={species.id} onChange={(e) => onSpecies(e.target.value as SpeciesId)} className="min-h-[40px] min-w-0 flex-1 bg-transparent font-medium text-ink outline-none" data-testid="coast-species">
          {options.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

/**
 * "What are you after?" (the trip plan's first chapter): the kind of fishing as one row of chips
 * with a sliding pill, then the species that kind of fishing involves, as chips. Both go in the link.
 */
export function TargetPicker({ audience, onAudience, species, onSpecies, compact = false }: { audience: Audience; onAudience: (a: Audience) => void; species: Species; onSpecies: (s: SpeciesId) => void; compact?: boolean }) {
  const options = SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience === "shellfish" ? "shellfish" : audience));
  const label: Record<Audience, string> = {
    all: "Any activity",
    commercial: "Commercial",
    recreational: "Recreational",
    shellfish: "Shellfish",
  };
  return (
    <div className={compact ? "space-y-3" : "space-y-4"} data-testid="coast-plan">
      <div>
        <p className="mb-2 text-[13px] font-medium text-ink-3">Kind of fishing</p>
        <div role="radiogroup" aria-label="Kind of fishing" className="grid grid-cols-2 gap-1 rounded-2xl bg-surface-3 p-1 sm:inline-flex sm:flex-wrap" data-testid="coast-audience">
          {AUDIENCES.map((x) => {
            const on = audience === x.id;
            return (
              <button
                key={x.id}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => onAudience(x.id)}
                className={`cw-press relative min-h-[40px] rounded-xl px-3.5 text-[14px] font-medium transition-colors ${on ? "text-white" : "text-ink-2 hover:text-ink"}`}
                data-testid={`coast-for-${x.id}`}
              >
                {on && <motion.span layoutId="audience-pill" className="absolute inset-0 rounded-xl bg-navy-900" transition={{ type: "spring", stiffness: 420, damping: 34 }} aria-hidden />}
                <span className="relative">{label[x.id]}</span>
              </button>
            );
          })}
        </div>
      </div>
      <div>
        <p className="mb-2 text-[13px] font-medium text-ink-3">Target species</p>
        <motion.div
          layout
          role="radiogroup"
          aria-label="Target species"
          className={`flex gap-1.5 ${compact ? "-mx-4 flex-nowrap overflow-x-auto px-4 pb-1 [mask-image:linear-gradient(to_right,#000_88%,transparent)] [scrollbar-width:none] sm:-mx-5 sm:px-5" : "flex-wrap"}`}
          data-testid="coast-species"
        >
          <AnimatePresence initial={false}>
            {options.map((s) => {
              const on = species.id === s.id;
              return (
                <motion.button
                  layout
                  key={s.id}
                  initial={{ opacity: 0, scale: 0.92 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.92 }}
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  onClick={() => onSpecies(s.id as SpeciesId)}
                  data-testid={`coast-species-${s.id}`}
                  className={`cw-press min-h-[38px] shrink-0 whitespace-nowrap rounded-full border px-3 text-[14px] transition-colors ${on ? "border-navy-900 bg-navy-900 text-white" : "border-hairline-strong bg-surface text-ink-2 hover:border-ink-3 hover:text-ink"}`}
                >
                  {s.label}
                </motion.button>
              );
            })}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- the answer (Level 1)

export type AnswerClause = {
  kind: "official" | "measured" | "model" | "gap";
  text: string;
  aged?: boolean;
  testid: string;
};

/**
 * The answer under the place name, one short line per kind of knowing, each led by its mark
 * (signature §6 Level 1): the first screen answers before it asks anything.
 */
export function AnswerLine({ clauses }: { clauses: AnswerClause[] }) {
  return (
    <ul className="max-w-[62ch] space-y-2 text-[16px] leading-snug text-ink sm:text-[17px]" data-testid="coast-answer">
      {clauses.map((c, i) => (
        <li key={c.testid} className="grid grid-cols-[16px_1fr] items-baseline gap-x-2" data-testid={c.testid} data-kind={c.kind}>
          <Mark kind={c.kind} aged={c.aged} className="translate-y-px" />
          {/* when the answer changes (new waters, species or day) the line shows that it did */}
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={c.text}
              initial={{ opacity: 0, y: 6 }}
              animate={{
                opacity: 1,
                y: 0,
                transition: {
                  delay: i * 0.05,
                  duration: 0.28,
                  ease: [0.2, 0, 0, 1],
                },
              }}
              exit={{ opacity: 0, y: -4, transition: { duration: 0.12 } }}
            >
              {c.text}
            </motion.span>
          </AnimatePresence>
        </li>
      ))}
    </ul>
  );
}

/** The official answer for these waters, shared by the answer line and the official card. */
export function officialAnswer(assocs: Association[], audience: Audience, species: Species) {
  const live = assocs.filter((a) => !a.dispute);
  const applies = live.filter((a) => a.tier === "geographic" && fits(a, audience));
  const mine = applies.filter((a) => recordNamesSpecies(a.record.species, species));
  const forWhat = species.id === "any" ? (audience === "all" ? "" : ` for ${AUDIENCES.find((x) => x.id === audience)!.label.toLowerCase()}`) : ` for ${speciesPhrase(species)}`;
  const disputed = assocs.filter((a) => a.dispute);
  const headline = mine.length ? `${mine.length} notice${mine.length === 1 ? " covers" : "s cover"} these waters${forWhat}` : `No notice in CoastWatch's list covers these waters${forWhat}`;
  return { live, applies, mine, forWhat, disputed, headline };
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
  statements = [],
}: {
  /** CDFW's statewide statements for the chosen species and activity (season status first), quoted */
  statements?: {
    id: string;
    agency: string;
    statement: string;
    topic: string;
    source: { url: string };
  }[];
  assocs: Association[];
  audience: Audience;
  species: Species;
  verification: Verification | null;
  totalActive: number;
  onAll: (el: HTMLElement) => void;
  onDetails: () => void;
}) {
  const { live, applies, mine, disputed, headline } = officialAnswer(assocs, audience, species);
  const others = applies.filter((a) => !mine.includes(a));
  const unresolved = live.filter((a) => a.tier === "unresolved" && fits(a, audience) && recordNamesSpecies(a.record.species, species));
  const outside = live.filter((a) => a.tier === "outside").length;
  return (
    <section aria-labelledby="brief-official-h" className="rounded-[18px] border border-hairline border-l-[3px] border-l-official-line bg-surface px-4 py-4 sm:px-5" data-testid="brief-official">
      <div className="flex items-center justify-between gap-3">
        <Eyebrow tone="official">Official notices for these waters</Eyebrow>
        <span className="shrink-0 whitespace-nowrap rounded-full border border-official-line px-2 py-0.5 text-[12px] font-medium text-official-ink" data-testid="brief-verification">
          {verification?.state === "verified" ? "Checked by a person" : "Not verified"}
        </span>
      </div>
      {statements.length > 0 && (
        // a fishery's status leads: "no notice" below it must never read as "open"
        <ul className="mt-2 space-y-1.5" data-testid="brief-status">
          {statements.map((st) => (
            <li key={st.id} className="rounded-xl bg-official-bg/60 px-3 py-2">
              <p className="text-[12.5px] font-medium text-official-ink">
                {st.agency} · {st.topic}
              </p>
              <p className="text-[15.5px] font-medium leading-snug text-ink">“{st.statement}”</p>
              <a href={st.source.url} {...ext} className={`${link} text-[13px]`}>
                {st.agency} page ↗
              </a>
            </li>
          ))}
        </ul>
      )}
      <h2 id="brief-official-h" className="mt-2 font-display text-[22px] font-medium leading-snug text-ink sm:text-[24px]" data-testid="brief-official-headline">
        {headline}
      </h2>
      {/* an absence states the limit of the claim, never an all-clear (signature P6) */}
      {!mine.length && <p className="mt-1 text-[14px] text-ink-2">That is not the same as safe or open: the list may be incomplete. Check the agencies (next steps).</p>}
      {(mine.length > 0 || disputed.length > 0) && (
        <ul className="mt-3 divide-y divide-hairline">
          {disputed.map((a) => (
            <li key={a.record.id} className="py-2.5" data-testid={`brief-disputed-${a.record.id}`}>
              <p className="text-[13px] font-medium text-warning">Reported lifted by {a.dispute!.agency} · awaiting review</p>
              <p className="text-[15px] font-medium leading-snug text-ink">{a.record.title}</p>
              <p className="mt-0.5 flex flex-wrap gap-x-3 text-[13px]">
                {a.record.sources.find((x) => x.url.startsWith("https://")) && (
                  <a href={a.record.sources.find((x) => x.url.startsWith("https://"))!.url} {...ext} className={link} data-testid={`brief-original-${a.record.id}`}>
                    The original notice ↗
                  </a>
                )}
                <a href={a.dispute!.url} {...ext} className={link}>
                  {a.dispute!.agency} {a.dispute!.release}, the later release ↗
                </a>
              </p>
              <p className="text-[13px] text-ink-3">Confirm with {a.dispute!.agency} which applies; CoastWatch shows it as neither in effect nor lifted.</p>
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
      <p className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-hairline pt-2.5 text-[13px] text-ink-2">
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
      <p className="text-[13px] font-medium text-official-ink">
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

const STEP_MARK: Record<
  Step["kind"],
  {
    mark: "official" | "conditions" | "gap" | "measured";
    label: string;
    aged?: boolean;
  }
> = {
  official: { mark: "official", label: "Official" },
  conditions: { mark: "conditions", label: "Conditions" },
  gap: { mark: "gap", label: "Not known here" },
  hab: { mark: "measured", label: "Measured nearby" },
  history: { mark: "measured", label: "History", aged: true },
};

/**
 * Next steps as rows (signature P8: eight right steps had become a document): each row is the
 * step and its source, led by its mark; why it is suggested opens in place. The first `max` show;
 * on a phone they fit beside the official answer instead of a wall of text. Conditions readings are
 * left to the conditions card when it is on the same page.
 */
export function NextSteps({ steps: given, max = 5, conditionsShown = false, sub }: { steps: Step[]; max?: number; conditionsShown?: boolean; sub?: string }) {
  const [all, setAll] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  // where the conditions card sits right below, its readings are not repeated as steps
  const steps = conditionsShown ? given.filter((s) => s.kind !== "conditions") : given;
  const shown = all ? steps : steps.slice(0, max);
  return (
    <section aria-labelledby="next-h" className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="next-steps">
      <Eyebrow>Next steps</Eyebrow>
      <h2 id="next-h" className="mt-1 font-display text-[22px] font-medium leading-snug text-ink">
        Before you go
      </h2>
      {sub && (
        <p className="mt-0.5 text-[14px] text-ink-2" data-testid="next-steps-for">
          {sub}
        </p>
      )}
      <ol className="mt-2 divide-y divide-hairline">
        {shown.map((s) => {
          const isOpen = open === s.id;
          const rid = `step-why-${s.id}`;
          return (
            <li key={s.id} className="grid grid-cols-[16px_1fr] gap-x-2.5 py-2.5" data-testid={`step-${s.id}`} data-kind={s.kind}>
              <Mark kind={STEP_MARK[s.kind].mark} aged={STEP_MARK[s.kind].aged} className="mt-[5px]" />
              <div className="min-w-0">
                <p className="text-[15px] font-medium leading-snug text-ink">
                  <span className="sr-only">{STEP_MARK[s.kind].label}: </span>
                  {s.title}
                </p>
                {s.scope && (
                  <p className="mt-0.5 text-[13px] leading-snug text-ink-2" data-testid={`scope-${s.id}`}>
                    {s.scope}
                  </p>
                )}
                <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-ink-3">
                  {"url" in s.source ? (
                    s.source.url.startsWith("/") ? (
                      <NextLink href={s.source.url} className={`${link} font-medium`}>
                        {s.source.label} →
                      </NextLink>
                    ) : (
                      <a href={s.source.url} {...ext} className={`${link} font-medium`}>
                        {s.source.label} ↗
                      </a>
                    )
                  ) : (
                    <a href={`tel:${s.source.tel}`} className={`${link} font-medium tabular`}>
                      {s.source.label}
                    </a>
                  )}
                  {s.also && (
                    <a href={s.also.url} {...ext} className={`${link} font-medium`} data-testid={`also-${s.id}`}>
                      {s.also.label} ↗
                    </a>
                  )}
                  {s.when && <span>{s.when}</span>}
                  <button type="button" aria-expanded={isOpen} aria-controls={rid} onClick={() => setOpen(isOpen ? null : s.id)} className="min-h-[28px] text-ink-3 underline decoration-ink-3/40 underline-offset-2 hover:text-ink" data-testid={`why-${s.id}`}>
                    {isOpen ? "Less" : "Why"}
                  </button>
                </p>
                <div id={rid} hidden={!isOpen} className="mt-1 space-y-1 text-[14px] leading-snug text-ink-2">
                  <p>{s.reason}</p>
                  {s.caveat && <p className="text-ink-3">How sure: {s.caveat}</p>}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
      {steps.length > max && (
        <button type="button" onClick={() => setAll(!all)} className="cw-press mt-1 min-h-[36px] text-[14px] font-medium text-accent hover:underline" data-testid="next-steps-more">
          {all ? "Fewer steps" : `${steps.length - max} more step${steps.length - max === 1 ? "" : "s"}`}
        </button>
      )}
    </section>
  );
}

// ---------------------------------------------------------------- conditions

/** "Monterey Canyon Outer, CA  (156)" → "Monterey Canyon Outer"; "MONTEREY, MONTEREY BAY" → "Monterey, Monterey Bay". */
export function stationName(n: string): string {
  const s = n.replace(/\s*\(\d+\)\s*$/, "").replace(/,\s*CA\s*$/i, "").replace(/\s*-\s*\d+\s*NM.*$/i, "").trim();
  return s === s.toUpperCase() ? s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()) : s;
}

/** What a Coastal reader needs from a failed or empty part: plain words; the technical reason goes in `title`. */
function plain(part: { status: string; reason?: string } | undefined, what: string): { text: string; detail?: string } {
  if (!part || !("reason" in part) || !part.reason) return { text: what };
  if (part.status === "unavailable")
    return {
      text: `NOAA's ${what} service did not answer just now.`,
      detail: part.reason,
    };
  return { text: part.reason };
}

export type ToxinNear = {
  station: string;
  km: number;
  date: string;
  ageDays: number;
  value: string;
} | null;

function Tile({ label, value, sub, state, testid, href, detail }: { label: string; value: React.ReactNode; sub: React.ReactNode; state: "ok" | "old" | "none"; testid: string; href?: string; detail?: string }) {
  const dot = { ok: "bg-measured", old: "bg-warning", none: "bg-ink-3/50" }[state];
  return (
    <div className="min-w-0" data-testid={testid} data-state={state} title={detail}>
      <p className="flex items-center gap-1.5 text-[13px] text-ink-3">
        <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        {label}
      </p>
      {value === "Loading…" ? (
        // a skeleton the size of the answer (no shimmer loop); the words stay for screen readers
        <p className="mt-1.5" aria-live="polite">
          <span className="block h-[18px] w-[min(220px,80%)] rounded-md bg-surface-3" aria-hidden />
          <span className="sr-only">Loading</span>
        </p>
      ) : (
        <p className="cw-fade-in mt-0.5 text-[16px] font-medium leading-snug text-ink">{value}</p>
      )}
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
  leads,
  lead,
  onLead,
  habTiles = true,
  day = null,
  tidesFirst = false,
  onMap,
  mapOn = false,
}: {
  /** shows the buoy and tide station on the map (My Coast) */
  onMap?: () => void;
  mapOn?: boolean;
  conditions: Conditions | null;
  loading: boolean;
  error: string | null;
  toxin?: ToxinNear;
  toxinError?: string | null;
  forecastLine?: string;
  forecastSub?: string;
  satelliteLine?: string;
  now: Date | null;
  leads?: (LayerArtifact | null)[];
  lead?: number;
  onLead?: (l: number) => void;
  /** the toxin, model and satellite tiles (off where the HAB briefing shows them) */
  habTiles?: boolean;
  /** the trip day (Pacific date), when one is chosen: its marine period and tides */
  day?: string | null;
  /** shellfish harvest: the tides lead */
  tidesFirst?: boolean;
}) {
  const c = conditions;
  // the trip day's marine forecast period ("THU", "THU NIGHT"), else the first one
  const dayName = day ? new Date(`${day}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" }).toUpperCase() : null;
  const period = c?.marine.status === "ok" ? ((dayName && c.marine.data.periods.find((p) => p.name.toUpperCase().startsWith(dayName.slice(0, 3)) && !/NIGHT/i.test(p.name))) ?? c.marine.data.periods[0]) : null;
  const wait = loading ? "Loading…" : error ? "Unavailable right now" : "—";
  const ageH = (iso: string) => (now ? (now.getTime() - new Date(iso).getTime()) / 3_600_000 : 0);
  const nextTides = (() => {
    if (c?.tides.status !== "ok" || !now) return null;
    const nowLocal = new Intl.DateTimeFormat("en-CA", {
      timeZone: "America/Los_Angeles",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    })
      .format(now)
      .replace(",", "");
    const after = c.tides.data.filter((t) => t.at >= nowLocal);
    const onDay = day ? after.filter((t) => t.at.startsWith(day)) : [];
    return (onDay.length ? onDay : after).slice(0, onDay.length ? 4 : 2);
  })();
  const tideTime = (t: string) => {
    const [d, hm] = t.split(" ");
    const [h, m] = hm.split(":").map(Number);
    return `${formatDate(d)} ${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
  };
  return (
    <section aria-labelledby="cond-h" className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="conditions">
      <Eyebrow tone="accent">Context for planning, from NOAA</Eyebrow>
      <h2 id="cond-h" className="mt-1 font-display text-[22px] font-medium leading-snug text-ink">
        Conditions
      </h2>
      {/* a disclaimer that changes the meaning is body text, not a micro label (signature P5) */}
      <p className="mt-0.5 text-[14px] text-ink-2" data-testid="cond-not-forecast">
        Wind, waves and tides only: not a toxin or fishing forecast.
      </p>
      {c?.alerts.status === "ok" && c.alerts.data.length > 0 && (
        <p className="mt-2 rounded-lg bg-warning/10 px-3 py-2 text-[14px] font-medium text-warning" data-testid="cond-alert">
          {c.alerts.data[0].event}
          {c.alerts.data[0].ends ? ` until ${formatDateTimePT(c.alerts.data[0].ends)}` : ""}
          <span className="font-normal text-ink-2"> · NWS {c.marine.status === "ok" ? c.marine.data.zoneName : ""}</span>
        </p>
      )}
      <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-4 min-[480px]:grid-cols-2">
        {tidesFirst && (
          <>
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
              <Tile label="Tides" value={c ? "Not available" : wait} sub={c ? plain(c.tides, "tide prediction").text : "NOAA CO-OPS"} detail={c ? plain(c.tides, "tide prediction").detail : undefined} state="none" testid="cond-tides" />
            )}
          </>
        )}
        {c?.marine.status === "ok" && period ? (
          <Tile
            label={`Marine forecast · ${period.name}`}
            value={marineHeadline(period)}
            sub={`NWS ${c.marine.data.zone}${c.marine.data.updated ? `, issued ${formatDateTimePT(c.marine.data.updated)}` : ""}`}
            state="ok"
            href={c.marine.url}
            testid="cond-marine"
          />
        ) : (
          c?.marine.status === "none" && c.marine.url ? (
            // CoastWatch does not query NWS's API (its robots.txt): the reader goes to NWS directly
            <Tile
              label="Marine forecast and warnings"
              value={
                <a href={c.marine.url} {...ext} className={link} data-testid="cond-marine-link">
                  Read NWS&apos;s marine forecast ↗
                </a>
              }
              sub="From the National Weather Service; CoastWatch does not copy it here"
              state="none"
              testid="cond-marine"
            />
          ) : (
          <Tile label="Marine forecast" value={c ? "Not available" : wait} sub={c ? plain(c.marine, "forecast").text : "NWS"} detail={c ? plain(c.marine, "forecast").detail : undefined} href={c && "url" in c.marine ? c.marine.url : undefined} state="none" testid="cond-marine" />
          )
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
          <Tile label="Measured nearby" value={c ? "No recent buoy reading" : wait} sub={c ? plain(c.buoy, "buoy data").text : "NDBC"} detail={c ? plain(c.buoy, "buoy data").detail : undefined} state="none" testid="cond-buoy" />
        )}
        {!tidesFirst && (
          <>
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
              <Tile label="Tides" value={c ? "Not available" : wait} sub={c ? plain(c.tides, "tide prediction").text : "NOAA CO-OPS"} detail={c ? plain(c.tides, "tide prediction").detail : undefined} state="none" testid="cond-tides" />
            )}
          </>
        )}
        {habTiles && (
          <>
            <Tile
              label="Toxin in seawater · nearest sample"
              value={toxin ? `${toxin.value}, ${ageWords(toxin.ageDays)}` : toxinError ? "Station data unavailable" : "No sample within 40 km"}
              sub={toxin ? `${toxin.station}, ${toxin.km < 10 ? toxin.km.toFixed(1) : Math.round(toxin.km)} km, sampled ${formatDate(toxin.date, { year: true })} · CalHABMAP` : "CalHABMAP shore stations"}
              state={!toxin ? "none" : toxin.ageDays > 30 ? "old" : "ok"}
              testid="cond-toxin"
            />
            <div className="min-w-0">
              <Tile label="Toxin forecast · model" value={forecastLine ?? ""} sub={forecastSub ?? ""} state={/^No /.test(forecastLine ?? "") ? "none" : "ok"} testid="cond-forecast" />
              {leads && onLead && (
                <label className="mt-1.5 inline-flex min-h-[36px] items-center gap-2 rounded-lg border border-hairline-strong px-2.5 text-[13px] text-ink-3">
                  For
                  <select value={lead} onChange={(e) => onLead(Number(e.target.value))} className="min-h-[32px] bg-transparent font-medium text-ink outline-none" data-testid="coast-day">
                    {leads.map((l, i) =>
                      l?.time.valid_date ? (
                        <option key={i} value={i}>
                          {formatDate(l.time.valid_date)} ({leadLabel(i)})
                        </option>
                      ) : null,
                    )}
                  </select>
                </label>
              )}
            </div>
            <Tile label="Satellite view of the water" value={satelliteLine ?? ""} sub="NOAA CoastWatch chlorophyll (algae, not toxin)" state={/^Clear/.test(satelliteLine ?? "") ? "ok" : "none"} testid="cond-satellite" />
          </>
        )}
      </div>
      {onMap && c && (c.buoy.status === "ok" || c.tides.status === "ok") && (
        <button type="button" onClick={onMap} aria-pressed={mapOn} className="cw-press mt-3 text-[13px] font-medium text-accent hover:underline" data-testid="cond-map">
          {mapOn ? "Shown on the map: the buoy and the tide station" : "Show the buoy and the tide station on the map"}
        </button>
      )}
    </section>
  );
}

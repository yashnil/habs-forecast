import type { LayerArtifact } from "@/generated/schema";
import type { Association } from "@/lib/applicability";
import { AUDIENCE } from "@/lib/applicability";
import type { Audience } from "@/lib/coast";
import { compass, marineHeadline, stationName, type Conditions, type MarinePeriod, type Tide } from "@/lib/conditions";
import { ageWords, fmtMeasured, fmtPct, EVIDENCE_RADIUS_KM, NEIGHBOURHOOD_KM, RECENT_DAYS, type Neighbourhood, type StationEvidence } from "@/lib/hab";
import type { Step } from "@/lib/nextSteps";
import type { Verification } from "@/lib/official";
import { recordNamesSpecies, speciesPhrase, type Species } from "@/content/species";

/**
 * The trip briefing (My Coast, Coastal mode): one trip (waters, activity, species, day) and, for it,
 * what matters right now, the HAB evidence near the waters and what to check before leaving.
 *
 * Pure functions over facts the page already has; they choose and order, they never add a claim:
 * - an official item quotes or names the agency's own words and links them;
 * - a measured item keeps its station, distance and date; a model item says it is a probability for
 *   offshore cells and never the harbour's own value unless the waters' own cell has one;
 * - a gap says once what is not known here, and is never phrased as safe, open or clean.
 */

// ---------------------------------------------------------------- the trip day

export type TripDay = {
  /** Pacific date, YYYY-MM-DD */
  date: string;
  /** "Today", "Tomorrow", "Thu" */
  label: string;
  /** "Oct 14" */
  day: string;
  /** the C-HARM lead index valid that day, or null when no model run covers it */
  lead: number | null;
};

/** Days a trip can be planned for: today and the next three (the span of NOAA's tide predictions the conditions read), each with the model day that covers it, if any. */
export function tripDays(today: string, leads: (LayerArtifact | null)[], span = 4): TripDay[] {
  const t0 = Date.parse(`${today}T12:00:00Z`);
  return Array.from({ length: span }, (_, i) => {
    const date = new Date(t0 + i * 86400000).toISOString().slice(0, 10);
    const d = new Date(`${date}T12:00:00Z`);
    const lead = leads.findIndex((l) => l?.time.valid_date === date);
    return {
      date,
      label: i === 0 ? "Today" : i === 1 ? "Tomorrow" : d.toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" }),
      day: d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" }),
      lead: lead >= 0 ? lead : null,
    };
  });
}

/** A trip date from a link: kept when it is one of the plannable days, else today. */
export function parseTripDate(v: string | null, days: TripDay[]): string | null {
  if (!days.length) return null;
  return v && days.some((d) => d.date === v) ? v : days[0].date;
}

// ---------------------------------------------------------------- what matters right now

export type MatterKind = "official" | "measured" | "model" | "gap" | "conditions" | "satellite";
export type Matter = {
  id: string;
  kind: MatterKind;
  /** alert: the reader must act on it (a fishery closed, a notice disputed); note: worth knowing; gap: not known here */
  tone: "alert" | "note" | "gap";
  headline: string;
  detail?: string;
  /** the agency's own words, quoted */
  quote?: string;
  /** a second statement from the agency about the same fishery, quoted with its own page */
  also?: { label: string; quote: string; url: string };
  source?: { label: string; url: string };
  /** what the map shows when the item is chosen */
  focus?: "cells" | "toxin" | "model" | "satellite" | "conditions" | "official" | null;
  aged?: boolean;
};

export type MatterInput = {
  assocs: Association[];
  audience: Audience;
  species: Species;
  /** CDFW statements for the species and activity, season status first */
  statements: { id: string; agency: string; statement: string; topic: string; source: { url: string } }[];
  verification: Verification | null;
  hab: { cells: StationEvidence | null; toxin: StationEvidence | null } | null;
  habError?: string | null;
  /** the model on the trip day: the waters' own cell and the real cells near them */
  model: { valid: string | null; pda: Neighbourhood | null } | null | "error" | "loading";
  /** the trip day has no model run */
  modelBeyond?: boolean;
  /** the newest C-HARM run is stale or historical by its own policy (what the map labels it) */
  modelStale?: { state: "stale" | "historical"; issued: string | null; ageDays: number } | null;
  conditions: Conditions | null;
  tripDate: string | null;
  /** today (Pacific), to pick the NWS forecast period for the trip day */
  today?: string | null;
  /** formats a date (YYYY-MM-DD) or an ISO time */
  fmt: (s: string) => string;
};

const fits = (a: Association, audience: Audience) => audience === "all" || AUDIENCE[a.record.fishery].groups.includes(audience);
const CLOSED = /\b(closed|closure|delay(ed)?|not open|prohibit|do not (eat|consume|take))\b/i;
const RESTRICTIVE = new Set(["fishery_closure", "take_restriction", "consumption_advisory", "quarantine", "special_advisory"]);
/** "Dungeness crab, commercial — marine toxin closures" → "marine toxin closures" */
export const topicWords = (topic: string) => (topic.includes("—") ? topic.split("—").pop()!.trim() : topic.charAt(0).toLowerCase() + topic.slice(1));
const km = (k: number) => (k < 10 ? k.toFixed(1) : String(Math.round(k)));

/** The notices that bear on this trip: disputed ones, ones that apply by their geography, and ones their geography does not settle. */
export function tripNotices(assocs: Association[], audience: Audience, species: Species) {
  const named = (a: Association) => recordNamesSpecies(a.record.species, species);
  return {
    disputed: assocs.filter((a) => a.dispute && named(a)),
    applying: assocs.filter((a) => !a.dispute && a.tier === "geographic" && fits(a, audience) && named(a)),
    unresolved: assocs.filter((a) => !a.dispute && a.tier === "unresolved" && fits(a, audience) && named(a)),
  };
}

/** Up to four things that matter for this trip, most consequential first. Official words lead; one HAB line; one conditions line. */
export function whatMatters(x: MatterInput): Matter[] {
  const out: Matter[] = [];
  const forWhat = x.species.id === "any" ? (x.audience === "all" ? "" : ` for ${{ commercial: "commercial fishing", recreational: "recreational fishing", shellfish: "shellfish harvest", all: "" }[x.audience]}`) : ` for ${speciesPhrase(x.species)}`;
  const n = tripNotices(x.assocs, x.audience, x.species);

  // 1. a fishery's own status (quoted): "closed" or "delayed" says more than any notice count; the
  // agency's other statement about the species (e.g. on marine toxin closures) is quoted with it
  const season = x.statements.find((s) => /season status/i.test(s.topic));
  const other = x.statements.find((s) => s !== season) ?? null;
  if (season)
    out.push({
      id: "season",
      kind: "official",
      tone: CLOSED.test(season.statement) ? "alert" : "note",
      headline: `${season.agency} season status for ${speciesPhrase(x.species)}`,
      quote: season.statement,
      also: other ? { label: `${other.agency} on ${topicWords(other.topic)}`, quote: other.statement, url: other.source.url } : undefined,
      source: { label: `${season.agency} page`, url: season.source.url },
      focus: null,
    });

  // without a season status, the agency's statement stands on its own
  for (const st of season ? [] : other ? [other] : [])
    out.push({
      id: `statement-${st.id}`,
      kind: "official",
      tone: CLOSED.test(st.statement) && !/\bno (closures?|delay)/i.test(st.statement) ? "alert" : "note",
      headline: `${st.agency} on ${topicWords(st.topic)}`,
      quote: st.statement,
      source: { label: `${st.agency} page`, url: st.source.url },
      focus: null,
    });

  // 2. notices an agency release appears to lift: never in effect, never lifted, until a person reviews them
  for (const a of n.disputed.slice(0, 1)) {
    const src = a.record.sources.find((s) => s.url.startsWith("https://"));
    out.push({
      id: `disputed-${a.record.id}`,
      kind: "official",
      tone: "alert",
      headline: `${a.dispute!.agency} appears to have lifted “${a.record.title}”`,
      detail: `Release ${a.dispute!.release}. Confirm with ${a.dispute!.agency} which applies before relying on either${src ? "; the original notice is linked in the evidence below" : ""}.`,
      source: { label: `${a.dispute!.agency} ${a.dispute!.release}`, url: a.dispute!.url },
      focus: "official",
    });
  }

  // 3. notices that cover these waters (by their own geography), the most restrictive first
  const applying = [...n.applying].sort((a, b) => Number(RESTRICTIVE.has(b.record.action)) - Number(RESTRICTIVE.has(a.record.action)));
  if (applying.length) {
    const first = applying[0];
    const src = first.record.sources.find((s) => s.url.startsWith("https://"));
    out.push({
      id: "notices",
      kind: "official",
      tone: applying.some((a) => RESTRICTIVE.has(a.record.action) || CLOSED.test(a.record.title)) ? "alert" : "note",
      headline: applying.length === 1 ? first.record.title : `${applying.length} notices cover these waters${forWhat}`,
      detail:
        applying.length === 1
          ? `${first.record.agency}${first.record.effective_date ? ` · since ${x.fmt(first.record.effective_date)}` : ""}${first.caveats.find((c) => /limit/.test(c)) ? ` · ${first.caveats.find((c) => /limit/.test(c))}` : ""}`
          : applying.slice(0, 3).map((a) => a.record.title).join(" · "),
      source: src ? { label: `${first.record.agency} notice`, url: src.url } : undefined,
      focus: "official",
    });
  } else if (!season) {
    out.push({
      id: "notices",
      kind: "gap",
      tone: "gap",
      headline: `No notice in Neritic's copy names these waters${forWhat}`,
      detail: n.unresolved.length
        ? `${n.unresolved.length === 1 ? "One more" : `${n.unresolved.length} more`} may apply; ${n.unresolved.length === 1 ? "its" : "their"} stated area does not settle it. Not the same as open or safe.`
        : "Not the same as open or safe: the copy can lag the agencies.",
      focus: "official",
    });
  }

  // 4. HAB: the most meaningful evidence, in one line (the evidence section shows all of it)
  out.push(habMatter(x));

  // 5. conditions on the trip day (context, never a HAB or fishing forecast)
  const c = conditionsMatter(x.conditions, x.tripDate, x.audience, x.fmt, x.today ?? x.tripDate);
  if (c) out.push(c);
  return out;
}

function habMatter(x: MatterInput): Matter {
  if (x.habError) return { id: "hab", kind: "gap", tone: "gap", headline: "Shore measurements could not be loaded", focus: null };
  const near = [x.hab?.cells, x.hab?.toxin].filter((e): e is StationEvidence => !!e && e.near && e.ageDays <= RECENT_DAYS).sort((a, b) => b.date.localeCompare(a.date) || a.km - b.km);
  const m = x.model !== "error" && x.model !== "loading" ? x.model : null;
  const stale = m && x.modelStale ? modelStaleWords(x.modelStale, x.fmt) : "";
  const modelText = (x.modelBeyond
    ? "NOAA's model does not reach this day"
    : m?.pda?.here != null
      ? `model ${fmtPct(m.pda.here)} chance of toxin above threshold in these waters' own cell`
      : m?.pda && m.pda.n > 0
        ? `model: no value at these nearshore waters (offshore cells within ${NEIGHBOURHOOD_KM} km: median ${fmtPct(m.pda.median)})`
        : m
          ? "outside NOAA's toxin model"
          : null) as string | null;
  const modelLine = modelText && stale && !x.modelBeyond && m ? `${modelText} (${stale})` : modelText;
  if (near.length) {
    const e = near[0];
    const v = e.kind === "cells" ? `Pseudo-nitzschia ${fmtMeasured("pn_seriata", e.values.reduce((a, y) => a + y.value, 0))}` : `Domoic acid in seawater ${fmtMeasured("pDA", e.values[0].value)}`;
    return {
      id: "hab",
      kind: "measured",
      tone: "note",
      headline: `${v} at ${e.station.name}, ${ageWords(e.ageDays)}`,
      detail: `${km(e.km)} km ${e.bearing} of these waters${modelLine ? ` · ${modelLine}` : ""}. Seawater, not seafood.`,
      focus: e.kind === "cells" ? "cells" : "toxin",
      aged: e.ageDays > 14,
    };
  }
  if (m?.pda?.here != null)
    return {
      id: "hab",
      kind: "model",
      tone: "note",
      headline: `NOAA model: ${fmtPct(m.pda.here)} chance toxin in the water exceeds its threshold here${stale ? ` (${stale})` : ""}`,
      detail: `A probability for this 3 km cell${m.valid ? ` on ${x.fmt(m.valid)}` : ""}, not a measurement.${stale ? " NOAA has published no newer run that Neritic could read." : ""} No shore sample within ${EVIDENCE_RADIUS_KM} km in the last ${RECENT_DAYS} days.`,
      focus: "model",
      aged: !!stale,
    };
  return {
    id: "hab",
    kind: "gap",
    tone: "gap",
    headline: `No recent HAB sample within ${EVIDENCE_RADIUS_KM} km`,
    detail: `${modelLine ? `${modelLine.charAt(0).toUpperCase()}${modelLine.slice(1)}. ` : ""}No sample is not a clean result.`,
    focus: m?.pda && m.pda.n > 0 ? "model" : null,
  };
}

/** "stale run, issued Oct 12, 2 days ago": the same state the map and the Science tab show. */
export function modelStaleWords(s: NonNullable<MatterInput["modelStale"]>, fmt: (d: string) => string): string {
  const age = s.ageDays === 1 ? "1 day ago" : `${s.ageDays} days ago`;
  return `${s.state === "stale" ? "stale" : "old"} run${s.issued ? `, issued ${fmt(s.issued)}` : ""}, ${age}`;
}

/** The trip day's tides, low tides first for shellfish harvest. */
export function tidesOn(tides: Tide[], date: string | null): Tide[] {
  return date ? tides.filter((t) => t.at.startsWith(date)) : [];
}

/** The NWS forecast period for the trip day: "Tonight"/"Today" for today, else the day's name ("Thu"), never a night period first. */
export function periodFor(periods: MarinePeriod[], date: string | null, today: string | null): MarinePeriod | null {
  if (!periods.length) return null;
  if (!date || !today || date === today) return periods[0];
  const day = new Date(`${date}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" }).toLowerCase();
  return periods.find((p) => p.name.toLowerCase() === day || p.name.toLowerCase() === day.slice(0, 3)) ?? null;
}

function conditionsMatter(c: Conditions | null, date: string | null, audience: Audience, fmt: (s: string) => string, today: string | null): Matter | null {
  if (!c) return null;
  const parts: string[] = [];
  let source: Matter["source"];
  // NWS's forecast for the trip day comes first, in its own words; then the nearest buoy now
  let nws: string | null = null;
  if (c.marine.status === "ok") {
    const p = periodFor(c.marine.data.periods, date, today);
    if (p) {
      nws = `NWS ${/tonight|today|this afternoon/i.test(p.name) ? p.name.toLowerCase() : `for ${p.name}`}: ${marineHeadline(p)}`;
      source = { label: `NWS forecast, ${c.marine.data.zoneName}`, url: c.marine.url };
    }
  }
  if (c.buoy.status === "ok" && c.buoy.station) {
    const b = c.buoy.data;
    const w = [b.windKt ? `wind ${Math.round(b.windKt.value)} kt${b.windDirDeg ? ` ${compass(b.windDirDeg.value)}` : ""}` : null, b.waveFt ? `waves ${b.waveFt.value.toFixed(1)} ft` : null].filter(Boolean).join(", ");
    if (w) {
      parts.push(`${w.charAt(0).toUpperCase()}${w.slice(1)} at ${stationName(c.buoy.station.name)} buoy now`);
      source ??= { label: `NDBC ${c.buoy.station.id}`, url: c.buoy.url };
    }
  }
  if (c.tides.status === "ok") {
    const day = tidesOn(c.tides.data, date);
    const lows = day.filter((t) => t.type === "L").sort((a, b) => a.ft - b.ft);
    if (audience === "shellfish" && lows.length) parts.unshift(`Lowest tide ${lows[0].ft.toFixed(1)} ft at ${fmt(lows[0].at).replace(/^.*?,\s*/, "").replace(/^\w+ \d+,\s*/, "")}`);
    else if (day.length) parts.push(`tides ${day.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft`).slice(0, 2).join(", ")}`);
    if (!source) source = { label: "NOAA tide predictions", url: c.tides.url };
  }
  const marine = c.marine.status === "ok" ? null : (c.marine.url ?? null);
  const advisory = c.alerts.status === "ok" ? c.alerts.data[0] : null;
  if (!parts.length && !marine && !nws) return null;
  const head = nws ?? (parts.length ? parts.join(" · ") : "No buoy or tide station reports near these waters");
  const rest = nws ? parts : [];
  return {
    id: "conditions",
    kind: "conditions",
    tone: advisory ? "alert" : "note",
    headline: advisory ? `${advisory.event} · ${head}` : head,
    detail: [rest.length ? `${rest.join(" · ")}.` : "", nws ? (advisory ? `NWS: “${advisory.headline}”.` : "Wind and seas, not a HAB or fishing forecast.") : "Marine forecast and warnings: read NWS for your zone."].filter(Boolean).join(" "),
    source: source ?? (marine ? { label: "NWS marine forecast", url: marine } : undefined),
    focus: "conditions",
  };
}

// ---------------------------------------------------------------- before you leave

const PRIORITY: [RegExp, number][] = [
  [/^dispute-/, 0],
  [/^statement-/, 1],
  [/^read-notices$/, 2],
  [/^cdph-map$/, 3],
  [/^confirm-/, 4],
  [/^closure-line$/, 5],
  [/^whale-safe$/, 6],
  [/^marine-alert$/, 7],
  [/^(marine|marine-link|marine-gap)$/, 8],
  [/^hab-sample$/, 9],
  [/^(species-rules|commercial|region-)/, 10],
  [/^toxin-gap$/, 11],
  [/^mpa$/, 12],
  [/^history$/, 13],
];
/** Steps that state a fact become the action that settles it. */
const ACTION_TITLE: Record<string, string> = {
  "toxin-gap": "Look at CDPH's latest toxin test results",
  "marine-gap": "Read the NWS marine forecast for your zone",
  marine: "Read NWS's full forecast for your zone",
};
const rank = (s: Step) => PRIORITY.find(([re]) => re.test(s.id))?.[1] ?? 20;

/**
 * "Before you leave": the steps as actions, most consequential first (the agency's own word on the
 * fishery, then its notices, then the safety lines and weather), at most `max`, the rest kept. A
 * quoted statement becomes the action of reading it at its source; buoy readings stay with the
 * conditions (they are facts, not actions) and the unverified-list reminder is the source line itself.
 */
export function checklist(steps: Step[], max = 5): { shown: Step[]; more: Step[] } {
  const seen = new Set<string>();
  const acts = steps
    .filter((s) => s.id !== "buoy" && s.id !== "verify")
    .map((s) =>
      s.id.startsWith("statement-")
        ? {
            ...s,
            title: /season status/i.test(s.reason) ? `Confirm the season status on ${s.source.label.replace(/ page$/, "")}'s page` : `Check ${s.source.label.replace(/ page$/, "")}'s page on ${topicWords(s.reason)}`,
            reason: `Neritic quotes it as “${s.title.replace(/^[^“]*“|”$/g, "")}” Agency pages can change between Neritic's checks.`,
          }
        : s,
    )
    .map((s) => (ACTION_TITLE[s.id] ? { ...s, title: ACTION_TITLE[s.id] } : s))
    .sort((a, b) => rank(a) - rank(b))
    .reduce<Step[]>((acc, s) => {
      // one step per source page: a second item on the same page is folded into the first
      const k = "url" in s.source ? s.source.url : s.source.tel;
      const kept = s.id.startsWith("dispute-") ? undefined : acc.find((y) => ("url" in y.source ? y.source.url : y.source.tel) === k);
      if (kept) {
        kept.reason = `${kept.reason} Also on this page: ${s.title.replace(/^(Check|Read|Confirm) /, (m) => m.toLowerCase())}.`;
        return acc;
      }
      if (seen.has(k)) return acc;
      seen.add(k);
      acc.push({ ...s });
      return acc;
    }, []);
  return { shown: acts.slice(0, max), more: acts.slice(max) };
}

/** A stable key for the checklist's ticks: one trip (waters, activity, species, day). */
export const checklistKey = (place: string, audience: Audience, species: string, date: string | null) => `cw-trip-checks-v1:${place}:${audience}:${species}:${date ?? "today"}`;

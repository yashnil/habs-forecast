import type { PortFact, PortToday } from "../contracts";
import { addDays, isStale, pacificDate } from "./age";
import { dayIn, FISHERY_LABEL, FISHERY_WORDS, fisheryIn } from "./intents";
import type { PilotPort } from "./places";
import { nwsZone, REF } from "./templates";
import type { AnswerLine, Fishery, Intent, Outcome, Referral, SafetyKind, SeenSnapshot } from "./types";

/**
 * The deterministic grounded answerer (readiness "demo"). It composes answers ONLY from the
 * PortFacts of one port: every line quotes a fact's own wording and cites that fact's id. It never
 * writes a number of its own; ages and dates are added by `age.ts` outside the guarded lines.
 */

export type Composed = { outcome: Outcome; lines: AnswerLine[]; caveats: string[]; referrals: Referral[] };

const SEV_RANK: Record<PortFact["severity"], number> = { red: 0, amber: 1, neutral: 2, info: 3 };
const bySeverity = (a: PortFact, b: PortFact) => SEV_RANK[a.severity] - SEV_RANK[b.severity];
const text = (f: PortFact) => `${f.id} ${f.headline} ${f.detail ?? ""}`;
const DOMOIC_WORDS = /domoic|toxin|biotoxin|psp|quarantine|advis/i;
/** toxin-related official records ("…not domoic acid" is the opposite) */
const DOMOIC = { test: (s: string) => DOMOIC_WORDS.test(s.replace(/\bnot (?:because of )?domoic acid\b/gi, "")) };
const WHALE = /whale|entangle|ramp/i;

const sentence = (s: string) => (/[.!?]$/.test(s.trim()) ? s.trim() : `${s.trim()}.`);
const day = (f: PortFact) => f.as_of?.slice(0, 10) ?? "an unknown date";

function officialLine(f: PortFact): AnswerLine {
  const checked = f.as_of ? ` (${f.source.name}, checked ${day(f)})` : ` (${f.source.name})`;
  return { text: `${sentence(f.headline)}${f.detail ? ` ${sentence(f.detail)}` : ""}${checked}`, cites: [f.id] };
}

/** One closing line for any unverified official records (the hierarchy's honesty rule). */
function unverifiedLine(facts: PortFact[]): AnswerLine[] {
  const un = facts.filter((f) => f.basis === "official" && f.verified === false);
  if (!un.length) return [];
  const names = [...new Set(un.map((f) => f.source.name))].join(" and ");
  return [{ text: `These official records were transcribed by Neritic from ${names} and are not yet verified by a person; the agency page is the authority.`, cites: un.map((f) => f.id) }];
}

/** Measurements always carry their date in the line itself (the guard's stale_undated rule). */
const dated = (f: PortFact, verb: string) => (f.as_of && !f.headline.includes(day(f)) ? ` ${verb} ${day(f)}.` : "");

function testLine(f: PortFact): AnswerLine {
  return { primary: true, text: `${sentence(f.headline)}${dated(f, "Sampled")}${f.value ? ` Result: ${f.value}.` : ""} This is a past lab sample, not a forecast.`, cites: [f.id] };
}

function plainLine(f: PortFact): AnswerLine {
  return { text: `${sentence(f.headline)}${f.detail ? ` ${sentence(f.detail)}` : ""}`, cites: [f.id] };
}

function observationLine(f: PortFact): AnswerLine {
  return { text: `${sentence(f.headline)}${dated(f, "Measured")} That is a measurement (an observation, not a forecast).`, cites: [f.id] };
}

const matchesFishery = (f: PortFact, fi: Fishery | null) => !fi || FISHERY_WORDS[fi].test(text(f)) || (fi === "dungeness" && /\bcrab\b/i.test(f.headline) && !/rock crab/i.test(f.headline));
const statusFacts = (t: PortToday) => t.facts.filter((f) => (f.kind === "status" || f.kind === "notice") && f.basis === "official").sort(bySeverity);

function noRecordLine(what: string, port: PilotPort): AnswerLine {
  return { text: `Neritic has no record of ${what} for ${port.label}. That does not mean there is none; check the agency page.`, cites: [] };
}

/* ------------------------------------------------------------------------------ intents */

function status(t: PortToday, port: PilotPort, q: string): Composed {
  const fi = fisheryIn(q);
  const facts = statusFacts(t).filter((f) => matchesFishery(f, fi));
  if (!facts.length)
    return {
      outcome: "no_data",
      lines: [noRecordLine(`the official status of ${fi ? FISHERY_LABEL[fi] : "any fishery"}`, port)],
      caveats: ["No record is not the same as open. CDFW sets seasons and closures."],
      referrals: [{ ...REF.cdfwInseason, why: "Season and in-season changes" }, { ...REF.cdfwAdvisories, why: "Closures and advisories" }],
    };
  const shown = fi ? facts : facts.slice(0, 5);
  return {
    outcome: "answered",
    lines: [...shown.map(officialLine), ...unverifiedLine(shown)],
    caveats: [],
    referrals: [{ ...REF.cdfwInseason, why: "Season and in-season changes" }, { ...REF.cdfwAdvisories, why: "Closures and advisories" }],
  };
}

function toxin(t: PortToday, port: PilotPort, q: string): Composed {
  const fi = fisheryIn(q);
  const official = statusFacts(t).filter((f) => matchesFishery(f, fi) && (fi ? true : DOMOIC.test(text(f))) && f.severity !== "info");
  const tests = t.facts.filter((f) => f.kind === "test" && matchesFishery(f, fi)).sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""));
  const lines = [...official.slice(0, 2).map(officialLine), ...tests.slice(0, 3).map(testLine)];
  if (!tests.length) lines.push(noRecordLine(`a CDPH toxin test${fi ? ` for ${FISHERY_LABEL[fi]}` : ""}`, port));
  return {
    outcome: tests.length || official.length ? "answered" : "no_data",
    lines: [...lines, ...unverifiedLine(official.slice(0, 2))],
    caveats: ["Toxin is measured only by CDPH lab tests of seafood. A result describes the sample on its date."],
    referrals: [{ ...REF.cdphDomoic, why: "All CDPH domoic acid results" }, { ...REF.cdphShellfish, why: "Seafood advisories" }],
  };
}

function sea(t: PortToday, port: PilotPort, q: string, now: Date): Composed {
  const today = pacificDate(now);
  const want = dayIn(q);
  const target = want === "today" ? today : addDays(today, 1);
  const forecasts = t.facts
    .filter((f) => (f.kind === "sea" || f.kind === "weather" || f.kind === "tide") && f.basis === "forecast")
    .filter((f) => (f.valid_for ? f.valid_for.from <= target && target <= f.valid_for.to : false));
  const wantsTide = /\btides?\b/i.test(q) && !/\b(wind|sea|swell|wave|weather)/i.test(q);
  const chosen = wantsTide ? forecasts.filter((f) => f.kind === "tide") : forecasts;
  const obs = want === "tomorrow" ? [] : t.facts.filter((f) => f.kind === "observation" && /buoy|wind|seas?\b/i.test(f.headline)).slice(0, 1);
  const lines = [...chosen.map(plainLine), ...obs.map(observationLine)];
  if (!chosen.length) lines.unshift({ text: `Neritic has no ${wantsTide ? "tide prediction" : "marine forecast"} on record for ${port.label} for ${want === "today" ? "today" : "tomorrow"}.`, cites: [] });
  return {
    outcome: chosen.length ? "answered" : "no_data",
    lines,
    caveats: ["Information only, not a go/no-go. Check the official NWS marine forecast before you leave."],
    referrals: [nwsZone(port)],
  };
}

function bloom(t: PortToday, port: PilotPort, q: string, now: Date): Composed {
  const fi = fisheryIn(q);
  const official = statusFacts(t).filter((f) => DOMOIC.test(text(f)) && matchesFishery(f, fi) && f.severity !== "info").slice(0, 2);
  const lines: AnswerLine[] = official.map(officialLine);
  const charm = t.facts.find((f) => f.kind === "bloom_forecast");
  if (charm) {
    if (isStale(charm, now) && charm.valid_for)
      lines.push({ primary: true, text: `The latest ${charm.source.name} run (${day(charm)}) covered ${charm.valid_for.from} to ${charm.valid_for.to} and is out of date, so Neritic has no current short-range bloom forecast for ${port.label}.`, cites: [charm.id] });
    else lines.push({ ...plainLine(charm), primary: true });
  } else lines.push({ text: `Neritic has no C-HARM bloom forecast on record for ${port.label}.`, cites: [] });
  const outlook = t.facts.find((f) => f.kind === "bloom_outlook");
  if (outlook && !isStale(outlook, now)) lines.push({ ...plainLine(outlook), primary: true });
  return {
    outcome: "answered",
    lines: [...lines, ...unverifiedLine(official)],
    caveats: ["Chlorophyll measures algae, not toxin. Only CDPH lab tests measure toxin in seafood.", "Bloom forecasts describe the water, not the seafood."],
    referrals: [{ ...REF.cdphDomoic, why: "Measured toxin in seafood" }],
  };
}

function whale(t: PortToday, port: PilotPort): Composed {
  const facts = [...t.facts.filter((f) => f.kind === "whale"), ...statusFacts(t).filter((f) => WHALE.test(text(f)))].sort(bySeverity);
  if (!facts.length)
    return { outcome: "no_data", lines: [noRecordLine("a whale-entanglement (RAMP) action", port)], caveats: [], referrals: [{ ...REF.cdfwWhale, why: "Current RAMP actions" }] };
  return { outcome: "answered", lines: [...facts.slice(0, 4).map(officialLine), ...unverifiedLine(facts.slice(0, 4))], caveats: [], referrals: [{ ...REF.cdfwWhale, why: "Current RAMP actions and zone maps" }] };
}

export function signature(f: PortFact): string {
  return `${f.severity}|${f.headline}|${f.value ?? ""}|${f.as_of ?? ""}`;
}

export function snapshotOf(t: PortToday): SeenSnapshot {
  return Object.fromEntries(t.facts.map((f) => [f.id, signature(f)]));
}

function changed(t: PortToday, port: PilotPort, since: SeenSnapshot | null | undefined): Composed {
  const lineFor = (f: PortFact) => (f.basis === "official" ? officialLine(f) : f.basis === "measured" ? (f.kind === "test" ? testLine(f) : observationLine(f)) : plainLine(f));
  if (!since || !Object.keys(since).length) {
    const newest = t.facts
      .filter((f) => f.kind !== "data_age" && f.as_of)
      .sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""))
      .slice(0, 3);
    return {
      outcome: "answered",
      lines: [{ text: `There is no earlier visit to ${port.label} on this device to compare with, so here are the newest records.`, cites: [] }, ...newest.map(lineFor), ...unverifiedLine(newest)],
      caveats: [],
      referrals: [],
    };
  }
  const changedFacts = t.facts.filter((f) => f.kind !== "data_age" && since[f.id] !== signature(f)).sort(bySeverity);
  const gone = Object.keys(since).filter((id) => !t.facts.some((f) => f.id === id)).length;
  const lines: AnswerLine[] = changedFacts.slice(0, 5).map((f) => ({ ...lineFor(f), text: `${since[f.id] ? "Updated" : "New"}: ${lineFor(f).text}` }));
  const feed = t.facts.find((f) => f.kind === "data_age");
  if (!lines.length) lines.push({ text: `Nothing in Neritic's records for ${port.label} has changed since your last check.`, cites: feed ? [feed.id] : [] });
  return {
    outcome: "answered",
    lines: [...lines, ...unverifiedLine(changedFacts.slice(0, 5))],
    caveats: gone ? ["Some records you saw before are no longer in Neritic's data."] : [],
    referrals: [],
  };
}

const BASIS_LABEL: Record<PortFact["basis"], string> = {
  official: "Official status (transcribed by Neritic, not yet verified by a person)",
  measured: "Measurements",
  forecast: "Forecasts",
  outlook: "Outlooks",
  analysis: "Analysis",
  derived: "Neritic",
};

function sources(t: PortToday): Composed {
  const order: PortFact["basis"][] = ["official", "measured", "forecast", "outlook", "analysis", "derived"];
  const lines: AnswerLine[] = [];
  for (const b of order) {
    const fs = t.facts.filter((f) => f.basis === b);
    if (!fs.length) continue;
    const label = b === "official" && fs.every((f) => f.verified) ? "Official status (verified by a person)" : BASIS_LABEL[b];
    lines.push({ text: `${label}: ${[...new Set(fs.map((f) => f.source.name))].join(", ")}.`, cites: fs.map((f) => f.id) });
  }
  return { outcome: "answered", lines, caveats: ["Neritic is not an official source. Official records outrank measurements, which outrank forecasts."], referrals: [] };
}

function dataAgeAnswer(t: PortToday): Composed {
  const lines: AnswerLine[] = [];
  const official = t.facts.filter((f) => f.basis === "official").sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""))[0];
  if (official?.as_of) lines.push({ text: `Official status last checked: ${day(official)} (${official.source.name}).`, cites: [official.id] });
  const feed = t.facts.find((f) => f.kind === "data_age");
  if (feed) lines.push(plainLine(feed));
  const groups: [string, (f: PortFact) => boolean][] = [
    ["Latest toxin test sampled", (f) => f.kind === "test"],
    ["Marine forecast issued", (f) => f.kind === "sea" || f.kind === "weather"],
    ["Bloom forecast run", (f) => f.kind === "bloom_forecast"],
  ];
  for (const [label, pick] of groups) {
    const f = t.facts.filter(pick).sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""))[0];
    if (f?.as_of) lines.push({ text: `${label}: ${day(f)} (${f.source.name}).`, cites: [f.id] });
  }
  return { outcome: lines.length ? "answered" : "no_data", lines: [...lines, ...unverifiedLine(official ? [official] : [])], caveats: [], referrals: [] };
}

/* --------------------------------------------------------------------- safety (info only) */

export function relatedForSafety(kind: SafetyKind, t: PortToday, q: string, now: Date): AnswerLine[] {
  const fi = fisheryIn(q);
  if (kind === "eat") {
    const official = statusFacts(t).filter((f) => matchesFishery(f, fi) && (fi ? true : DOMOIC.test(text(f))) && f.severity !== "info").slice(0, 2);
    const tests = t.facts.filter((f) => f.kind === "test" && matchesFishery(f, fi)).slice(0, 2);
    return [...official.map(officialLine), ...tests.map(testLine), ...unverifiedLine(official)];
  }
  if (kind === "fish_legal") {
    const official = statusFacts(t).filter((f) => matchesFishery(f, fi)).slice(0, 3);
    return [...official.map(officialLine), ...unverifiedLine(official)];
  }
  const target = dayIn(q) === "today" ? pacificDate(now) : addDays(pacificDate(now), 1);
  const fc = t.facts.filter((f) => (f.kind === "sea" || f.kind === "weather") && f.valid_for && f.valid_for.from <= target && target <= f.valid_for.to);
  const obs = t.facts.filter((f) => f.kind === "observation" && /buoy/i.test(f.headline)).slice(0, 1);
  return [...fc.map(plainLine), ...obs.map(observationLine)];
}

export function composeIntent(intent: Intent, t: PortToday, port: PilotPort, q: string, now: Date, since?: SeenSnapshot | null): Composed {
  switch (intent) {
    case "status":
      return status(t, port, q);
    case "toxin":
      return toxin(t, port, q);
    case "sea":
      return sea(t, port, q, now);
    case "bloom":
      return bloom(t, port, q, now);
    case "whale":
      return whale(t, port);
    case "changed":
      return changed(t, port, since);
    case "sources":
      return sources(t);
    case "data_age":
      return dataAgeAnswer(t);
    default:
      return { outcome: "abstained", lines: [], caveats: [], referrals: [] };
  }
}

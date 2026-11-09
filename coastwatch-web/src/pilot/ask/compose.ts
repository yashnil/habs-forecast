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
const DOMOIC_WORDS = /domoic|toxin|biotoxin|psp|quarantine|advis|do not eat|human consumption/i;
/** what a record is about: its id and headline (a long detail may mention another area's toxin closure in passing) */
const about = (f: PortFact) => `${f.id} ${f.headline}`;
/** toxin-related official records ("…not domoic acid" is the opposite; a RAMP "Fleet Advisory" is about whales) */
const DOMOIC = { test: (s: string) => DOMOIC_WORDS.test(s.replace(/\bnot (?:a |because of )?domoic acid\b/gi, "").replace(/\bnot toxin\b/gi, "").replace(/\bfleet advisory\b/gi, "")) };
const WHALE = /whale|entangle|ramp/i;

const sentence = (s: string) => (/[.!?]$/.test(s.trim()) ? s.trim() : `${s.trim()}.`);
const day = (f: PortFact) => f.as_of?.slice(0, 10) ?? "an unknown date";

/**
 * A long record is quoted by its first sentence and its "Next review" sentence (the Today card, one tap
 * away on the citation, has the rest). Only whole sentences of the fact's own words: nothing is rewritten.
 */
const BRIEF_OVER = 260;
export function brief(detail: string): string {
  const d = detail.trim();
  if (d.length <= BRIEF_OVER) return sentence(d);
  // a sentence ends at ". " before a capital; "37 11.000' N" and "Jan. 1" stay whole
  const parts = d.split(/(?<=[a-z0-9)"”'’]\.)\s+(?=[A-Z"“])/);
  const next = parts.slice(1).filter((x) => /^(Next (review|CDFW)|Sources differ)/.test(x));
  return [parts[0], ...next].map(sentence).join(" ");
}

function officialLine(f: PortFact): AnswerLine {
  // a curated record is "checked" on its date; a registry notice carries the date it took effect
  const when = f.as_of ? `, ${f.kind === "notice" ? "effective" : f.kind === "whale" ? "last assessed" : "checked"} ${day(f)}` : "";
  return { text: `${sentence(f.headline)}${f.detail ? ` ${brief(f.detail)}` : ""} (${f.source.name}${when})`, cites: [f.id] };
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
  // the result is repeated only when the headline does not already state its number
  const n = f.value?.match(/\d[\d,.]*/)?.[0];
  const result = f.value && !(n && f.headline.includes(n)) ? ` Result: ${f.value}.` : "";
  return { primary: true, text: `${sentence(f.headline)}${dated(f, "Sampled")}${result} This is a past lab sample, not a forecast.`, cites: [f.id] };
}

function plainLine(f: PortFact): AnswerLine {
  return { text: `${sentence(f.headline)}${f.detail ? ` ${sentence(f.detail)}` : ""}`, cites: [f.id] };
}

function observationLine(f: PortFact): AnswerLine {
  return { text: `${sentence(f.headline)}${dated(f, "Measured")} That is a measurement (an observation, not a forecast).`, cites: [f.id] };
}

const matchesFishery = (f: PortFact, fi: Fishery | null) => !fi || FISHERY_WORDS[fi].test(about(f)) || (fi === "dungeness" && /\bcrab\b/i.test(f.headline) && !/rock crab/i.test(f.headline));
/** A notice that a status row already quotes as its headline (the row was built from that notice). */
const quotedBy = (n: PortFact, rows: PortFact[]) => n.kind === "notice" && rows.some((r) => r.kind === "status" && sentence(r.headline).endsWith(sentence(n.headline.replace(/^[^:]{1,40}:\s*/, ""))));
const statusFacts = (t: PortToday) => {
  const all = t.facts.filter((f) => (f.kind === "status" || f.kind === "notice") && f.basis === "official");
  return all.filter((f) => !quotedBy(f, all)).sort(bySeverity);
};
const restrictive = (f: PortFact) => f.severity === "red" || f.severity === "amber";

function noRecordLine(what: string, port: PilotPort): AnswerLine {
  return { text: `Neritic has no record of ${what} for ${port.label}. That does not mean there is none; check the agency page.`, cites: [] };
}

/* ------------------------------------------------------------------------------ intents */

function status(t: PortToday, port: PilotPort, q: string): Composed {
  const fi = fisheryIn(q);
  // "recreational crab" / "commercial crab": the sector named leads (both are still shown)
  const sector = /\b(recreational|sport)\b/i.test(q) ? /recreational|sport/i : /\bcommercial\b/i.test(q) ? /commercial/i : null;
  const facts = statusFacts(t)
    .filter((f) => matchesFishery(f, fi))
    .sort((a, b) => (sector ? Number(sector.test(about(b))) - Number(sector.test(about(a))) : 0));
  if (!facts.length)
    return {
      outcome: "no_data",
      lines: [noRecordLine(`the official status of ${fi ? FISHERY_LABEL[fi] : "any fishery"}`, port)],
      caveats: ["No record is not the same as open. CDFW sets seasons and closures."],
      referrals: [{ ...REF.cdfwInseason, why: "Season and in-season changes" }, { ...REF.cdfwAdvisories, why: "Closures and advisories" }],
    };
  // no fishery named: what is closed, delayed or restricted comes first; "nothing on record" rows only when that is all there is
  const flagged = facts.filter(restrictive);
  const shown = fi ? facts.slice(0, 4) : (flagged.length ? flagged : facts).slice(0, 5);
  const more = (fi ? facts : flagged.length ? flagged : facts).length - shown.length;
  return {
    outcome: "answered",
    lines: [...shown.map(officialLine), ...unverifiedLine(shown)],
    caveats: more > 0 ? [`${more} more official record${more === 1 ? "" : "s"} for ${port.label} ${more === 1 ? "is" : "are"} on the Today page.`] : [],
    referrals: [{ ...REF.cdfwInseason, why: "Season and in-season changes" }, { ...REF.cdfwAdvisories, why: "Closures and advisories" }],
  };
}

function toxin(t: PortToday, port: PilotPort, q: string): Composed {
  const fi = fisheryIn(q);
  const official = statusFacts(t).filter((f) => matchesFishery(f, fi) && (fi ? true : DOMOIC.test(about(f))) && f.severity !== "info");
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

/**
 * Does a forecast speak to `target`? With valid_for, by its dates. Without one (the current NWS
 * product, the next tide), only while it is fresh: its own headline names the period, so quoting it
 * is honest, and the stale note says when it is old.
 */
function covers(f: PortFact, target: string, now: Date): boolean {
  if (f.valid_for) return f.valid_for.from <= target && target <= f.valid_for.to;
  return !isStale(f, now);
}

const isForecast = (f: PortFact, kinds: PortFact["kind"][]) => f.basis === "forecast" && kinds.includes(f.kind);
const nwsAlerts = (t: PortToday) => t.facts.filter((f) => f.id.startsWith("weather:alert:"));

/**
 * The forecasts that speak to the day asked about. A dated forecast (valid_for) answers for its day;
 * the current period (no valid_for: the NWS period in force, the next tide) answers "now" and "today",
 * and stands in for tomorrow only when nothing dated reaches it (the answer then says so).
 */
function forecastsFor(t: PortToday, kinds: PortFact["kind"][], want: "today" | "tomorrow" | null, now: Date): PortFact[] {
  const today = pacificDate(now);
  const days = want === "today" ? [today] : want === "tomorrow" ? [addDays(today, 1)] : [today, addDays(today, 1)];
  const pool = t.facts.filter((f) => isForecast(f, kinds));
  const dated = pool.filter((f) => f.valid_for && days.some((d) => covers(f, d, now)));
  const current = pool.filter((f) => !f.valid_for && covers(f, today, now));
  return want === "tomorrow" ? (dated.length ? dated : current) : [...current, ...dated];
}

function sea(t: PortToday, port: PilotPort, q: string, now: Date): Composed {
  const want = dayIn(q);
  const wantsTide = /\btides?\b/i.test(q) && !/\b(wind|sea|swell|wave|weather)/i.test(q);
  const weather = wantsTide ? [] : forecastsFor(t, ["sea", "weather"], want, now).slice(0, 4);
  let tides = forecastsFor(t, ["tide"], want, now);
  // a general question gets the next tide, not two days of tables
  if (!wantsTide && want === null) tides = tides.filter((f) => !f.valid_for).length ? tides.filter((f) => !f.valid_for) : tides.slice(0, 1);
  if (!wantsTide && want === "today") tides = tides.slice(0, 1);
  const chosen = [...weather, ...tides];
  // an NWS advisory or warning in force leads: it is official, the forecast is not
  const alerts = wantsTide ? [] : nwsAlerts(t);
  const wantsWater = /\bwater temp|\bsst\b|\btemperature/i.test(q);
  const observed = t.facts.filter((f) => f.kind === "observation" && /buoy|wind|seas?\b|water/i.test(f.headline));
  const isWater = (f: PortFact) => /water temp/i.test(f.headline);
  // the buoy's waves and, where a second buoy supplies it, the wind; the water only when asked for
  const obs = want === "tomorrow" || wantsTide ? [] : wantsWater ? observed.filter(isWater).slice(0, 1) : observed.filter((f) => !isWater(f)).slice(0, 2);
  // "what is the water temperature?" is answered by the reading itself
  if (wantsWater && obs.length)
    return { outcome: "answered", lines: obs.map(observationLine), caveats: ["One reading at one place; the water a few miles away can differ."], referrals: [] };
  // "the buoy reading now": the measurement is what was asked for (a phone shows it first)
  const obsFirst = /\bbuoy|\breading|\bmeasur/i.test(q);
  const lines = [...alerts.map(plainLine), ...chosen.map(plainLine), ...obs.map((f) => ({ ...observationLine(f), ...(obsFirst ? { primary: true } : {}) }))];
  if (!chosen.length) lines.unshift({ text: `Neritic has no ${wantsTide ? "tide prediction" : "marine forecast"} on record for ${port.label} for ${want === "today" ? "today" : "tomorrow"}.`, cites: [] });
  else if (want !== "today" && chosen.every((f) => !f.valid_for))
    lines.unshift({ text: "Neritic holds only the current NWS forecast period, which may not reach tomorrow; the full NWS forecast is linked below.", cites: [] });
  return {
    outcome: chosen.length ? "answered" : "no_data",
    lines,
    caveats: ["Information only, not a go/no-go. Check the official NWS marine forecast before you leave."],
    referrals: [nwsZone(port)],
  };
}

function bloom(t: PortToday, port: PilotPort, q: string, now: Date): Composed {
  const fi = fisheryIn(q);
  const official = statusFacts(t).filter((f) => DOMOIC.test(about(f)) && matchesFishery(f, fi) && restrictive(f)).slice(0, 2);
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

/** The fact that says when Neritic's own data were last refreshed ("age:data"; the eval records call it "data_age:feed"). */
const feedFact = (t: PortToday) => t.facts.find((f) => f.kind === "data_age" && /(^|:)(data|feed)$/.test(f.id)) ?? null;

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
  const feed = feedFact(t);
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
    // official records a person transcribed (verified or not) are one line; an agency's own feed (NWS alerts) is another
    const all = t.facts.filter((f) => f.basis === b);
    const fs = b === "official" ? all.filter((f) => f.verified !== null) : all;
    const direct = b === "official" ? all.filter((f) => f.verified === null && f.kind !== "status") : [];
    if (fs.length) {
      const label = b === "official" && fs.every((f) => f.verified) ? "Official status (verified by a person)" : b === "derived" ? "Data-age notes" : BASIS_LABEL[b];
      lines.push({ text: `${label}: ${[...new Set(fs.map((f) => f.source.name))].join(", ")}.`, cites: fs.map((f) => f.id) });
    }
    if (direct.length) lines.push({ text: `Official alerts, as issued: ${[...new Set(direct.map((f) => f.source.name))].join(", ")}.`, cites: direct.map((f) => f.id) });
  }
  return { outcome: "answered", lines, caveats: ["Neritic is not an official source. Official records outrank measurements, which outrank forecasts."], referrals: [] };
}

function dataAgeAnswer(t: PortToday): Composed {
  const lines: AnswerLine[] = [];
  const official = t.facts.filter((f) => f.basis === "official").sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""))[0];
  if (official?.as_of) lines.push({ text: `Official status last checked: ${day(official)} (${official.source.name}).`, cites: [official.id] });
  const feed = feedFact(t);
  if (feed) lines.push(plainLine(feed));
  // anything Neritic itself has flagged as old or missing, in its own words
  for (const f of t.facts.filter((x) => x.kind === "data_age" && x !== feed).slice(0, 3)) lines.push(plainLine(f));
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
    const official = statusFacts(t).filter((f) => matchesFishery(f, fi) && (fi ? true : DOMOIC.test(about(f))) && f.severity !== "info").slice(0, 2);
    const tests = t.facts.filter((f) => f.kind === "test" && matchesFishery(f, fi)).slice(0, 2);
    return [...official.map(officialLine), ...tests.map(testLine), ...unverifiedLine(official)];
  }
  if (kind === "fish_legal") {
    const official = statusFacts(t).filter((f) => matchesFishery(f, fi)).slice(0, 3);
    return [...official.map(officialLine), ...unverifiedLine(official)];
  }
  const fc = forecastsFor(t, ["sea", "weather"], dayIn(q) === "today" ? "today" : "tomorrow", now).slice(0, 3);
  const obs = t.facts.filter((f) => f.kind === "observation" && /buoy/i.test(f.headline) && !/water temp/i.test(f.headline)).slice(0, 2);
  return [...nwsAlerts(t).map(plainLine), ...fc.map(plainLine), ...obs.map(observationLine)];
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

import type { FisheryId, PortFact, PortToday } from "../contracts";
import { dayWords, fisheryLabel, ledeParts } from "../facts";
import { addDays, isStale, pacificDate, spanWords } from "./age";
import { dayIn, FISHERY_LABEL, FISHERY_WORDS, fisheryIn } from "./intents";
import type { PilotPort } from "./places";
import { nwsZone, REF } from "./templates";
import type { AnswerLine, Fishery, Intent, Outcome, Referral, SafetyKind, SeenSnapshot } from "./types";

/**
 * The deterministic grounded answerer (readiness "demo"). It composes answers ONLY from the
 * PortFacts of one port: every line quotes a fact's own wording and cites that fact's id. It never
 * writes a number of its own; ages and dates are added by `age.ts` outside the guarded lines.
 *
 * An answer reads the way a deckhand would say it: the first line answers the question in one
 * sentence (the lead record's own headline; "No —" or "Yes —" only when an official closure or delay
 * settles a yes/no question about one fishery), then at most three short supporting lines (what is
 * next, a disagreement between sources, the related rule), each with its citation. A record's long
 * detail is not repeated: it is one tap away on the citation ("See it on Today"). Dates are worded
 * as Today words them ("checked Oct 12", "tested Sep 28": src/pilot/facts.ts `dayWords`).
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
/** A fact's date in the reader's terms, as Today words it: "Oct 12", "Jan 1, 2027". */
const day = (f: PortFact, now: Date) => (f.as_of ? dayWords(f.as_of, now.toISOString()) : "an unknown date");
const iso = (f: PortFact) => f.as_of?.slice(0, 10) ?? null;
/** a sentence ends at ". " before a capital; "37 11.000' N" and "Jan. 1" stay whole */
const sentences = (detail: string) => detail.trim().split(/(?<=[a-z0-9)"”'’]\.)\s+(?=[A-Z"“])/).map(sentence);
/** The sentences of a record's detail that say what happens next, where sources disagree, or how far the record can be trusted. */
const KEY_SENTENCE = /^(Next |Sources differ|That is not a guarantee|Check the agenc|This is a rule for)/;
const SHORT_DETAIL = 160;

/**
 * What a record adds to its own headline, as short lines. Its key sentences when it has any (one
 * "Next …" at most: the agency's own schedule over Neritic's bare review date); otherwise the whole
 * detail when it is short. A long detail without them is left to the Today card: nothing is rewritten
 * and no sentence of the headline is said twice.
 */
export function support(f: PortFact): string[] {
  const d = (f.detail ?? "").trim();
  if (!d) return [];
  const all = sentences(d);
  const key = all.filter((x) => KEY_SENTENCE.test(x));
  if (!key.length) return d.length <= SHORT_DETAIL ? [all.join(" ")] : [];
  const next = key.filter((x) => x.startsWith("Next "));
  const keepNext = next.find((x) => !/^Next review\b/.test(x)) ?? next[0];
  const out: string[] = [];
  // "That is not a guarantee: notices can lag. Check the agencies before you go." is one thought, said as one line
  for (const x of key.filter((k) => !k.startsWith("Next ") || k === keepNext)) {
    if (/^Check the agenc/.test(x) && out.length && /^That is not a guarantee/.test(out[out.length - 1])) out[out.length - 1] += ` ${x}`;
    else out.push(x);
  }
  return out;
}

/**
 * A record in one sentence: its headline. A status row's headline is "Fishery (sector): Sector season …";
 * it is said the way Today's hero says it ("Dungeness crab: commercial season delayed …"), the sector once.
 */
export function say(f: PortFact): string {
  if (f.kind === "status" && f.id.startsWith("status:")) {
    const label = fisheryLabel(f.id.slice(7) as FisheryId);
    if (f.headline.startsWith(`${label}: `)) {
      const p = ledeParts(label, f.headline.slice(label.length + 2));
      return sentence(`${p.label}: ${p.text}`);
    }
  }
  return sentence(f.headline);
}

/** "CDFW, checked Oct 12": who says it and when Neritic read it (a notice carries the date it took effect). System text, shown beside the line. */
const sourceOf = (f: PortFact, now: Date) => `${f.source.name}${f.as_of ? `, ${f.kind === "notice" ? "effective" : f.kind === "whale" ? "last assessed" : "checked"} ${day(f, now)}` : ""}`;

/** An official record as one line: its headline, with its source and date beside it. */
function officialLine(f: PortFact, now: Date): AnswerLine {
  return { text: say(f), cites: [f.id], source: sourceOf(f, now) };
}

/** The lead record's supporting lines, then each related record in one line: at most `max` in all, with room kept for one related record. */
function supporting(lead: PortFact, related: PortFact[], now: Date, max = 3): AnswerLine[] {
  const own = support(lead);
  const room = Math.max(1, max - Math.min(1, related.length));
  // when there is more than fits, a disagreement between sources is never the line that is cut
  const keep = own.length <= room ? own : [...own.filter((x) => x.startsWith("Sources differ")), ...own.filter((x) => !x.startsWith("Sources differ"))].slice(0, room);
  const lines = own.filter((x) => keep.includes(x)).map((t): AnswerLine => ({ text: t, cites: [lead.id] }));
  return [...lines, ...related.slice(0, Math.max(0, max - lines.length)).map((f) => officialLine(f, now))];
}

/** One closing line for any unverified official records (the hierarchy's honesty rule). */
function unverifiedLine(facts: PortFact[]): AnswerLine[] {
  const un = facts.filter((f) => f.basis === "official" && f.verified === false);
  if (!un.length) return [];
  const names = [...new Set(un.map((f) => f.source.name))].join(" and ");
  return [{ text: `Transcribed by Neritic from ${names} and not yet verified by a person; the agency page is the authority.`, cites: un.map((f) => f.id) }];
}

/** Measurements always carry their date in the line itself (the guard's stale_undated rule): added when the headline does not already say it. */
const dated = (f: PortFact, verb: string, now: Date) => (f.as_of && !f.headline.includes(iso(f)!) && !f.headline.includes(day(f, now)) ? ` ${verb} ${day(f, now)}.` : "");

function testLine(f: PortFact, now: Date): AnswerLine {
  // the result is repeated only when the headline does not already state its number
  const n = f.value?.match(/\d[\d,.]*/)?.[0];
  const result = f.value && !(n && f.headline.includes(n)) ? ` Result: ${f.value}.` : "";
  return { primary: true, text: `${sentence(f.headline)}${dated(f, "Sampled", now)}${result} This is a past lab sample, not a forecast.`, cites: [f.id] };
}

/** A forecast, outlook or data-age record: its headline, then what its detail adds (never its long tail of numbers). */
function plainLine(f: PortFact): AnswerLine {
  const d = (f.detail ?? "").trim();
  // an outlook's detail ends with its mg/m³ values and held-out skill: those are on its Today card
  const all = sentences(d);
  const cut = all.findIndex((x) => /^(Held-out |About [\d.]+ mg)/.test(x));
  const lean = f.kind === "bloom_outlook" && d ? [...(cut < 0 ? all : all.slice(0, cut)), ...all.filter((x) => /window has passed/.test(x))].join(" ") : d;
  return { text: `${sentence(f.headline)}${lean ? ` ${sentence(lean)}` : ""}`, cites: [f.id] };
}

function observationLine(f: PortFact, now: Date): AnswerLine {
  return { text: `${sentence(f.headline)}${dated(f, "Measured", now)} That is a measurement (an observation, not a forecast).`, cites: [f.id] };
}

const matchesFishery = (f: PortFact, fi: Fishery | null) => !fi || FISHERY_WORDS[fi].test(about(f)) || (fi === "dungeness" && /\bcrab\b/i.test(f.headline) && !/rock crab/i.test(f.headline));
/** A notice that a status row already quotes as its headline (the row was built from that notice). */
const quotedBy = (n: PortFact, rows: PortFact[]) =>
  n.kind === "notice" && rows.some((r) => r.kind === "status" && sentence(r.headline.replace(/\s*\((?:sport harvest|recreational) only\)\.?$/, "")).endsWith(sentence(n.headline.replace(/^[^:]{1,40}:\s*/, ""))));
const statusFacts = (t: PortToday) => {
  const all = t.facts.filter((f) => (f.kind === "status" || f.kind === "notice") && f.basis === "official");
  return all.filter((f) => !quotedBy(f, all)).sort(bySeverity);
};
const restrictive = (f: PortFact) => f.severity === "red" || f.severity === "amber";

function noRecordLine(what: string, port: PilotPort): AnswerLine {
  return { text: `Neritic has no record of ${what} for ${port.label}. That does not mean there is none; check the agency page.`, cites: [] };
}

/* ------------------------------------------------------------------------------ intents */

/** A yes/no question ("Is crab open?", "Did salmon close?"), not a "when", "why" or "what" one. */
const yesNo = (q: string) => /\b(is|are|was|did|has|have|does|do|will)\b/i.test(q) && !/\b(when|why|how|what|where|which|who)\b/i.test(q);
/** Is it a yes/no about being open, or about being closed? */
const asksOpen = (q: string) => yesNo(q) && /\b(open(ed|s|ing)?|opener)\b/i.test(q) && !/\bclos|\bdelay/i.test(q);
const asksClosed = (q: string) => yesNo(q) && /\b(closed?|closures?|delayed|shut)\b/i.test(q) && !/\bopen/i.test(q);
/**
 * "No — " / "Yes — " before the lead record, only where the record settles the question: one fishery
 * was asked about, its row is an official closure or delay of that fishery itself (red; not a rule on
 * sport or recreational harvest only), and Neritic's sources do not disagree about it. Never "yes, it
 * is open": the calmest record is "none on record", and that is said in its own words.
 */
function direct(q: string, lead: PortFact): string {
  const settles = lead.kind === "status" && lead.severity === "red" && !/Sources differ/.test(lead.detail ?? "") && !/^(sport harvest|recreational)\b/i.test(lead.value ?? "");
  if (!settles) return say(lead);
  const word = asksOpen(q) ? "No" : asksClosed(q) ? "Yes" : null;
  if (!word) return say(lead);
  const s = say(lead);
  const first = s.match(/^[A-Za-z]+/)?.[0] ?? "";
  // "No — commercial Dungeness crab is delayed …": a common word loses its capital, a name keeps it
  return `${word} — ${/^(The|A|An|Commercial|Recreational|Season|No|All)$/.test(first) ? s.charAt(0).toLowerCase() + s.slice(1) : s}`;
}

function status(t: PortToday, port: PilotPort, q: string, now: Date): Composed {
  const fi = fisheryIn(q);
  // "recreational crab" / "commercial crab": the sector named leads, and the other sector is left out
  const sector = /\b(recreational|sport)\b/i.test(q) ? /recreational|sport/i : /\bcommercial\b/i.test(q) ? /commercial/i : null;
  const other = sector ? (sector.test("commercial") ? /status:\w+_recreational\b/ : /status:\w+_commercial\b/) : null;
  const facts = statusFacts(t)
    .filter((f) => matchesFishery(f, fi) && !(fi && other && other.test(f.id) && statusFacts(t).some((x) => x !== f && matchesFishery(x, fi) && sector!.test(about(x)))))
    .sort((a, b) => (sector ? Number(sector.test(about(b))) - Number(sector.test(about(a))) : 0));
  const referrals = [{ ...REF.cdfwInseason, why: "Season and in-season changes" }, { ...REF.cdfwAdvisories, why: "Closures and advisories" }];
  if (!facts.length)
    return {
      outcome: "no_data",
      lines: [noRecordLine(`the official status of ${fi ? FISHERY_LABEL[fi] : "any fishery"}`, port)],
      caveats: ["No record is not the same as open. CDFW sets seasons and closures."],
      referrals,
    };
  if (fi) {
    // one fishery: the answer in one sentence, then what is next and the related rule
    const [lead, ...rest] = facts;
    const shown = [lead, ...rest.slice(0, 2)];
    const sup = supporting(lead, rest.slice(0, 2), now);
    const cited = shown.filter((f) => f === lead || sup.some((l) => l.cites.includes(f.id)));
    const more = facts.length - cited.length;
    return {
      outcome: "answered",
      lines: [{ ...officialLine(lead, now), text: direct(q, lead), primary: true }, ...sup, ...unverifiedLine(cited)],
      caveats: more > 0 ? [`${more} more official record${more === 1 ? "" : "s"} for ${port.label} ${more === 1 ? "is" : "are"} on the Today page.`] : [],
      referrals,
    };
  }
  // no fishery named: what is closed, delayed or restricted, one line each; "nothing on record" rows only when that is all there is
  const flagged = facts.filter(restrictive);
  const pool = flagged.length ? flagged : facts;
  const shown = pool.slice(0, 5);
  const more = pool.length - shown.length;
  return {
    outcome: "answered",
    lines: [...shown.map((f) => ({ ...officialLine(f, now), text: `${say(f)}${/Sources differ/.test(f.detail ?? "") ? " Neritic's sources differ on this one; its Today card shows both." : ""}` })), ...unverifiedLine(shown)],
    caveats: more > 0 ? [`${more} more official record${more === 1 ? "" : "s"} for ${port.label} ${more === 1 ? "is" : "are"} on the Today page.`] : [],
    referrals,
  };
}

function toxin(t: PortToday, port: PilotPort, q: string, now: Date): Composed {
  const fi = fisheryIn(q);
  const official = statusFacts(t).filter((f) => matchesFishery(f, fi) && (fi ? true : DOMOIC.test(about(f))) && f.severity !== "info");
  const tests = t.facts.filter((f) => f.kind === "test" && matchesFishery(f, fi)).sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""));
  // the official status of the fishery leads (it outranks any test), in one line; then the tests themselves
  const lines = [...official.slice(0, 2).map((f) => officialLine(f, now)), ...tests.slice(0, 3).map((f) => testLine(f, now))];
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
    return { outcome: "answered", lines: obs.map((f) => observationLine(f, now)), caveats: ["One reading at one place; the water a few miles away can differ."], referrals: [] };
  // "the buoy reading now": the measurement is what was asked for (a phone shows it first)
  const obsFirst = /\bbuoy|\breading|\bmeasur/i.test(q);
  const lines = [...alerts.map(plainLine), ...chosen.map(plainLine), ...obs.map((f) => ({ ...observationLine(f, now), ...(obsFirst ? { primary: true } : {}) }))];
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
  const lines: AnswerLine[] = official.map((f) => officialLine(f, now));
  const charm = t.facts.find((f) => f.kind === "bloom_forecast");
  if (charm) {
    if (isStale(charm, now) && charm.valid_for)
      lines.push({ primary: true, text: `The latest ${charm.source.name} run (${day(charm, now)}) covered ${spanWords(charm.valid_for.from, charm.valid_for.to, now)} and is out of date, so Neritic has no current short-range bloom forecast for ${port.label}.`, cites: [charm.id] });
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

function whale(t: PortToday, port: PilotPort, now: Date): Composed {
  // the RAMP record for the port's zone leads; the fisheries it delays or restricts follow, one line each
  const facts = [...t.facts.filter((f) => f.kind === "whale"), ...statusFacts(t).filter((f) => WHALE.test(text(f))).sort(bySeverity)];
  if (!facts.length)
    return { outcome: "no_data", lines: [noRecordLine("a whale-entanglement (RAMP) action", port)], caveats: [], referrals: [{ ...REF.cdfwWhale, why: "Current RAMP actions" }] };
  const [lead, ...rest] = facts;
  const sup = supporting(lead, rest.slice(0, 2), now);
  const cited = facts.filter((f) => f === lead || sup.some((l) => l.cites.includes(f.id)));
  return { outcome: "answered", lines: [{ ...officialLine(lead, now), primary: true }, ...sup, ...unverifiedLine(cited)], caveats: [], referrals: [{ ...REF.cdfwWhale, why: "Current RAMP actions and zone maps" }] };
}

export function signature(f: PortFact): string {
  return `${f.severity}|${f.headline}|${f.value ?? ""}|${f.as_of ?? ""}`;
}

export function snapshotOf(t: PortToday): SeenSnapshot {
  return Object.fromEntries(t.facts.map((f) => [f.id, signature(f)]));
}

/** The fact that says when Neritic's own data were last refreshed ("age:data"; the eval records call it "data_age:feed"). */
const feedFact = (t: PortToday) => t.facts.find((f) => f.kind === "data_age" && /(^|:)(data|feed)$/.test(f.id)) ?? null;

function changed(t: PortToday, port: PilotPort, since: SeenSnapshot | null | undefined, now: Date): Composed {
  const lineFor = (f: PortFact) => (f.basis === "official" ? officialLine(f, now) : f.basis === "measured" ? (f.kind === "test" ? testLine(f, now) : observationLine(f, now)) : plainLine(f));
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

function dataAgeAnswer(t: PortToday, now: Date): Composed {
  const lines: AnswerLine[] = [];
  const official = t.facts.filter((f) => f.basis === "official").sort((a, b) => (b.as_of ?? "").localeCompare(a.as_of ?? ""))[0];
  if (official?.as_of) lines.push({ text: `Official status last checked: ${day(official, now)} (${official.source.name}).`, cites: [official.id] });
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
    if (f?.as_of) lines.push({ text: `${label}: ${day(f, now)} (${f.source.name}).`, cites: [f.id] });
  }
  return { outcome: lines.length ? "answered" : "no_data", lines: [...lines, ...unverifiedLine(official ? [official] : [])], caveats: [], referrals: [] };
}

/* --------------------------------------------------------------------- safety (info only) */

export function relatedForSafety(kind: SafetyKind, t: PortToday, q: string, now: Date): AnswerLine[] {
  const fi = fisheryIn(q);
  if (kind === "eat") {
    const official = statusFacts(t).filter((f) => matchesFishery(f, fi) && (fi ? true : DOMOIC.test(about(f))) && f.severity !== "info").slice(0, 2);
    const tests = t.facts.filter((f) => f.kind === "test" && matchesFishery(f, fi)).slice(0, 2);
    return [...official.map((f) => officialLine(f, now)), ...tests.map((f) => testLine(f, now)), ...unverifiedLine(official)];
  }
  if (kind === "fish_legal") {
    // the records as they stand, one line each: never a "yes" or a "no" (the template above them says who decides)
    const official = statusFacts(t).filter((f) => matchesFishery(f, fi)).slice(0, 3);
    return [...official.map((f) => officialLine(f, now)), ...unverifiedLine(official)];
  }
  const fc = forecastsFor(t, ["sea", "weather"], dayIn(q) === "today" ? "today" : "tomorrow", now).slice(0, 3);
  const obs = t.facts.filter((f) => f.kind === "observation" && /buoy/i.test(f.headline) && !/water temp/i.test(f.headline)).slice(0, 2);
  return [...nwsAlerts(t).map(plainLine), ...fc.map(plainLine), ...obs.map((f) => observationLine(f, now))];
}

export function composeIntent(intent: Intent, t: PortToday, port: PilotPort, q: string, now: Date, since?: SeenSnapshot | null): Composed {
  switch (intent) {
    case "status":
      return status(t, port, q, now);
    case "toxin":
      return toxin(t, port, q, now);
    case "sea":
      return sea(t, port, q, now);
    case "bloom":
      return bloom(t, port, q, now);
    case "whale":
      return whale(t, port, now);
    case "changed":
      return changed(t, port, since, now);
    case "sources":
      return sources(t);
    case "data_age":
      return dataAgeAnswer(t, now);
    default:
      return { outcome: "abstained", lines: [], caveats: [], referrals: [] };
  }
}

import { EVIDENCE_RADIUS_KM, NEIGHBOURHOOD_KM, RECENT_DAYS } from "@/lib/hab";
import { pageClaims, type Claim, type EvidenceGraph, type EvidenceNode } from "@/lib/harborEvidence";

/**
 * Evidence consistency: rules every claim of an evidence graph must keep against the nodes it
 * cites, checked mechanically so that a wording change, a new source or a new claim cannot quietly
 * say more than the evidence does. Pure; used by the unit and end-to-end tests and by the review
 * tooling, never to change what the page shows.
 *
 * The rules:
 *  1. cites       a claim that states a reading (measured, model, satellite, conditions) cites at
 *                 least one node, and every id it cites is a node of the graph.
 *  2. numbers     every number in a claim's headline is in the nodes it cites (their text, value,
 *                 distance, count or radius) or in the agency's quoted words; every number in its
 *                 detail, history or plain text is somewhere in the graph (nodes or gaps) or is one
 *                 of the stated rules (60 km, 30 days, 15 km, the 3 km cell, 80 km).
 *  3. dates       every date in a claim's headline is a date of a node it cites.
 *  4. kind words  satellite chlorophyll is said with "not toxin"; a model probability is called a
 *                 probability and not a measurement; a measurement is never called a forecast or a
 *                 prediction; seawater is never seafood; cell counts, chlorophyll and readings in
 *                 general are never said to be "about" domoic acid or toxin (#66 review 3.4); an
 *                 agency-page line says it is Neritic's unreviewed copy (3.2) and carries no calendar
 *                 end date that could read as "over" (3.1).
 *  5. safety      no claim calls waters or seafood safe or unsafe, or speaks of risk scores or
 *                 catches, except to say that a reading does *not* mean safe.
 *  6. reach       a claim resting only on context or far nodes says so (context, not the harbour,
 *                 too far, or its distance).
 *  7. once        no sentence is said twice among the claims the page shows.
 */
export type Violation = { rule: "cites" | "numbers" | "dates" | "kind-words" | "safety" | "reach" | "once"; claim: string; message: string };

const READING_KINDS = new Set<Claim["kind"]>(["measured", "model", "satellite", "conditions"]);
const MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec";
const DATE_RE = new RegExp(`\\b(${MONTHS})\\.? (\\d{1,2})(?:, (\\d{4}))?\\b`, "g");
const MONTH_RANGE_RE = new RegExp(`\\b(${MONTHS})–(${MONTHS})\\b`, "g");
const CLOCK_RE = /\b\d{1,2}:\d{2}\s?(AM|PM)\b/gi;
const YEAR_RANGE_RE = /\b(19|20)\d{2}–(19|20)\d{2}\b/g;
const YEAR_RE = /\b(19|20)\d{2}\b/g;
const NUM_RE = /(?<![\w.])\d[\d,]*(?:\.\d+)?/g;
/** distances, windows and grid sizes that are rules of the briefing, not readings */
const RULE_NUMBERS = new Set([EVIDENCE_RADIUS_KM, RECENT_DAYS, NEIGHBOURHOOD_KM, 3, 80, 500, 7].map(String));

const norm = (n: string) => String(Number(n.replace(/,/g, "")));

/** counts of agency notices ("3 notices cover…", "2 more may apply") are counts, not readings */
const COUNT_RE = /\b\d+ (more|notices?|agency notices?)\b/g;

/** the numbers in a string, with dates, clock times, years, month ranges and notice counts taken out first */
export function numbersIn(s: string): string[] {
  const t = s.replace(COUNT_RE, " ").replace(DATE_RE, " ").replace(MONTH_RANGE_RE, " ").replace(CLOCK_RE, " ").replace(YEAR_RANGE_RE, " ").replace(YEAR_RE, " ");
  return (t.match(NUM_RE) ?? []).map(norm);
}

/** "Oct 15" / "October 15, 2025" → "10-15" (month-day, the year checked separately when given) */
export function datesIn(s: string): string[] {
  const out: string[] = [];
  for (const m of s.matchAll(DATE_RE)) {
    const i = MONTHS.split("|").findIndex((x) => x === m[1]);
    const month = i < 12 ? i + 1 : ({ Jan: 1, Feb: 2, Mar: 3, Apr: 4, Jun: 6, Jul: 7, Aug: 8, Sep: 9, Sept: 9, Oct: 10, Nov: 11, Dec: 12 } as Record<string, number>)[m[1]];
    out.push(`${String(month).padStart(2, "0")}-${m[2].padStart(2, "0")}`);
  }
  return out;
}

function nodeNumbers(n: EvidenceNode): Set<string> {
  const out = new Set<string>(numbersIn(`${n.text} ${n.where.name ?? ""} ${n.source.name}`));
  const add = (v: number | null | undefined, digits = [0, 1, 2]) => {
    if (v == null || !Number.isFinite(v)) return;
    out.add(norm(String(v)));
    for (const d of digits) out.add(norm(v.toFixed(d)));
    if (v > 0 && v <= 1) out.add(norm(String(Math.round(v * 100)))); // a probability said as a percent
    out.add(norm(String(Number(v.toPrecision(2)))));
  };
  add(n.value);
  add(n.where.km);
  add(n.support.n);
  add(n.support.radiusKm);
  add(n.time.ageDays);
  // the age as ageWords says it ("3 weeks ago", "14 months ago", "2 years ago")
  const d = n.time.ageDays;
  if (d != null) for (const v of [Math.round(d / 7), Math.round(d / 30.4), Math.floor(d / 365)]) out.add(String(v));
  return out;
}

function nodeDates(n: EvidenceNode): Set<string> {
  return new Set([n.time.observed, n.time.valid, n.time.issued].filter((d): d is string => !!d).map((d) => d.slice(5, 10)));
}

const SAFE_RE = /\b(un)?safe(ty)?\b/gi;
/** "shore readings … are about domoic acid": cells and chlorophyll are not toxin */
const ABOUT_TOXIN_RE = /\b(readings?|samples?|cells|chlorophyll|fluorescence)\b[^.]{0,60}\b(are|is) (all )?about (domoic acid|toxins?|biotoxins?)\b/i;
/** "until October 31", "to October 31 every year": an end date in a standing line */
const END_DATE_RE = new RegExp(`\\b(until|through|to|till) (${MONTHS}) \\d{1,2}\\b`, "i");
const NEGATION_RE = /\b(not|n't|never|no|nor|neither)\b[^.]{0,60}$/i;

export function checkGraph(g: EvidenceGraph): Violation[] {
  const out: Violation[] = [];
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const graphNumbers = new Set<string>(RULE_NUMBERS);
  for (const n of g.nodes) for (const x of nodeNumbers(n)) graphNumbers.add(x);
  for (const gap of g.gaps) {
    for (const x of numbersIn(gap.reason)) graphNumbers.add(x);
    if (gap.nearest) for (const d of [0, 1]) graphNumbers.add(norm(gap.nearest.km.toFixed(d)));
  }

  for (const c of g.claims) {
    const cited = c.supports.map((id) => byId.get(id));
    const all = [c.headline, c.detail, c.history, c.provenance, c.text].filter(Boolean).join(" ");

    // 1. cites
    for (const [i, n] of cited.entries()) if (!n) out.push({ rule: "cites", claim: c.id, message: `cites ${c.supports[i]}, which is not a node` });
    if (READING_KINDS.has(c.kind) && !c.supports.length) out.push({ rule: "cites", claim: c.id, message: `a ${c.kind} claim with no supporting node` });
    const nodes = cited.filter((n): n is EvidenceNode => !!n);

    // 2. numbers
    const own = new Set<string>();
    for (const n of nodes) for (const x of nodeNumbers(n)) own.add(x);
    for (const x of numbersIn(c.quote ?? "")) own.add(x);
    for (const x of numbersIn(c.also?.quote ?? "")) own.add(x);
    for (const w of c.agencyWords ?? []) for (const x of numbersIn(w.quote)) own.add(x);
    if (c.kind !== "official" && c.kind !== "agency_page") {
      for (const x of numbersIn(c.headline)) if (!own.has(x) && !RULE_NUMBERS.has(x)) out.push({ rule: "numbers", claim: c.id, message: `headline number ${x} is not in the nodes it cites` });
    }
    for (const field of [c.detail, c.history, c.text]) {
      for (const x of numbersIn(field ?? "")) if (!graphNumbers.has(x) && !own.has(x)) out.push({ rule: "numbers", claim: c.id, message: `number ${x} appears in no node or gap` });
    }

    // 3. dates
    if (c.kind !== "official" && c.kind !== "agency_page" && nodes.length) {
      const ds = new Set<string>();
      for (const n of nodes) for (const d of nodeDates(n)) ds.add(d);
      for (const d of datesIn(c.headline)) if (!ds.has(d)) out.push({ rule: "dates", claim: c.id, message: `headline date ${d} is not a date of the nodes it cites` });
    }

    // 4. kind words
    // each surface on its own: what the page shows (headline, detail, history) and the plain statement (text)
    const shown = [c.headline, c.detail, c.history, c.provenance].filter(Boolean).join(" ");
    for (const [where, t] of [["page", shown], ["text", c.text]] as const) {
      if (/chlorophyll/i.test(t) && !/not toxin/i.test(t)) out.push({ rule: "kind-words", claim: c.id, message: `chlorophyll without “not toxin” (${where})` });
      if (c.kind === "model" && !(/probability|chance/i.test(t) && /not a measurement/i.test(t))) out.push({ rule: "kind-words", claim: c.id, message: `a model claim not called a probability and not a measurement (${where})` });
    }
    if (c.kind === "measured" && /\b(forecast|predict(ed|ion)?|will be)\b/i.test(`${c.headline} ${c.history ?? ""}`)) out.push({ rule: "kind-words", claim: c.id, message: "a measurement called a forecast" });
    if (nodes.some((n) => n.id.startsWith("pda:") || n.id.startsWith("pn_cells:")) && !/seawater, not seafood|not a test of seafood|seawater measurement/i.test(shown)) out.push({ rule: "kind-words", claim: c.id, message: "a seawater measurement without “seawater, not seafood”" });
    if (ABOUT_TOXIN_RE.test(all)) out.push({ rule: "kind-words", claim: c.id, message: "readings, cells or chlorophyll said to be about toxin" });
    if (c.kind === "agency_page") {
      if (!c.provenance || !/Neritic's copy/.test(c.provenance) || !/not reviewed by a person/.test(c.provenance))
        out.push({ rule: "kind-words", claim: c.id, message: "an agency-page line without “Neritic's copy … not reviewed by a person”" });
      if (END_DATE_RE.test([c.headline, c.detail].filter(Boolean).join(" "))) out.push({ rule: "kind-words", claim: c.id, message: "an agency-page line with a calendar end date (reads as over once it passes)" });
    }

    // 5. safety
    for (const m of all.matchAll(SAFE_RE)) {
      const before = all.slice(0, m.index);
      if (!NEGATION_RE.test(before)) out.push({ rule: "safety", claim: c.id, message: `“${m[0]}” not negated` });
    }
    if (/\b(risk score|risk level|catch(es)? (will|is expected))\b/i.test(all)) out.push({ rule: "safety", claim: c.id, message: "speaks of risk scores or catches" });

    // 6. reach
    if (nodes.length && nodes.every((n) => n.where.representative === "context" || n.where.representative === "far") && !/context|not the harbo(u)?r|too far|\d+(\.\d)? km/i.test(all))
      out.push({ rule: "reach", claim: c.id, message: "rests only on context or far nodes without saying so" });
  }

  // 7. once
  const seen = new Map<string, string>();
  for (const c of pageClaims(g)) {
    for (const s of [c.headline, c.detail, c.history].filter(Boolean).flatMap((x) => x!.split(/(?<=[.!?])\s+/))) {
      const k = s.trim().toLowerCase();
      if (k.length < 24) continue; // "Seawater, not seafood." and other short tags may recur
      const prev = seen.get(k);
      if (prev && prev !== c.id) out.push({ rule: "once", claim: c.id, message: `“${s.trim()}” also said by ${prev}` });
      else seen.set(k, c.id);
    }
  }
  return out;
}

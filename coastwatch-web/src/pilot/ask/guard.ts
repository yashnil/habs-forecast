import type { PortFact } from "../contracts";
import { dayWords, pacificDay } from "../facts";
import type { AnswerLine } from "./types";

/**
 * Deterministic answer verifier — a TypeScript port of the #49 prototype's guard
 * (research/intelligence/cwi/guard.py), re-targeted from its evidence registry to PortFacts.
 *
 * The guard trusts no answerer, including Neritic's own deterministic one. It checks every line
 * against the facts that line cites, and the whole answer against the Neritic science rules
 * (docs/coastwatch/05-science-and-safety.md):
 *
 *   unknown_citation         a line cites a fact id that was not provided
 *   uncited_claim            a factual line (numbers, closures, toxin, …) cites nothing
 *   ungrounded_number        a number or date is not in the cited facts
 *   safety_claim             calling seafood, a trip or a crossing safe, clear or risk-free (unless negated)
 *   chlorophyll_as_toxin     chlorophyll presented as evidence of toxin
 *   observation_as_forecast  a measurement-only line makes a future claim
 *   research_overclaim       the PINN presented as beating the ConvLSTM
 *   hierarchy                a restrictive official record was provided and the answer leads with something else
 *   lifted_claim             an advisory/closure described as lifted/ended without an official record saying so
 *   unverified_omitted       unverified official records cited without saying so
 *   stale_undated            an old measurement cited without its date or age (its date as ISO, or in words exactly as Today
 *                            writes it for that day, year included when it is not this year's: "Sep 28", "Sep 28, 2025")
 */

export type ViolationCode =
  | "unknown_citation"
  | "uncited_claim"
  | "ungrounded_number"
  | "safety_claim"
  | "chlorophyll_as_toxin"
  | "observation_as_forecast"
  | "research_overclaim"
  | "hierarchy"
  | "lifted_claim"
  | "unverified_omitted"
  | "stale_undated";

export type Violation = { code: ViolationCode; line: number; detail: string };
export type GuardReport = { passed: boolean; violations: Violation[]; codes: ViolationCode[]; badLines: number[] };

const ISO_DATE = /\b\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?\b/g;
const NEG_BEFORE = /\b(not|n't|never|no|cannot|can't|doesn't|does not|isn't|aren't|without|neither|nor)\b[^.]{0,48}$/i;
const SAFETY =
  /\b(safe to (?:eat|harvest|consume|fish|collect|dig|go|head|cross|launch|set)|(?:is|are|it's|looks|seems|considered|deemed) (?:perfectly |completely |totally )?safe|safe (?:seas|conditions|day|for (?:you|boats|fishing|eating))|no (?:health )?risk|risk[- ]free|toxin[- ]free|all[- ]clear|nothing to worry|good to go|go ahead and)\b/gi;
const CHL_TOXIN =
  /chlorophyll[^.]{0,80}\b(indicat\w*|mean\w*|show\w*|signal\w*|sign of|evidence of|confirm\w*|proves?|so)\b[^.]{0,40}\b(toxi\w*|domoic|poison\w*)/i;
const FUTURE = /\b(will|expected to|forecast\w*|predict\w*|tomorrow|next (?:week|few days)|coming days|likely to|going to)\b/i;
const NOT_FORECAST = /\bnot (?:a |an )?(?:forecast|prediction)/i;
const OVERCLAIM = /\b(outperform\w*|beat\w*|better than|improv\w* (?:on|over)|more accurate than|superior)\b/i;
const RESEARCH = /\b(pinn|convlstm|physics-informed)\b/i;
const LIFTED = /\b(lifted|has ended|have ended|is over|no longer in effect|rescinded|reopened for eating|all[- ]clear)\b/i;
const UNVERIFIED = /\b(not (?:yet )?(?:been )?(?:human[- ])?verified|unverified|pending (?:human )?review|not yet (?:been )?reviewed|not (?:yet )?checked by a person)\b/i;
const FACTUAL = /\d|\b(advisor(y|ies)|closures?|closed|delayed|quarantine|domoic|toxin|ppm|probabilit\w*|chlorophyll|measured|restricted|banned)\b/i;
const ABSTAIN = /\b(couldn't|could not|can't|cannot|don't have|do not have|no (?:record|data|information|current)|unavailable|not available|isn't covered|not covered)\b/i;

/** An un-negated "lifted / has ended / is over" in a text ("no notice confirming it has ended" is negated). */
function assertsLifted(text: string): string | null {
  for (const m of text.matchAll(new RegExp(LIFTED.source, "gi"))) if (!NEG_BEFORE.test(text.slice(0, m.index))) return m[0];
  return null;
}

/**
 * Every numeric token in a text, normalised ("1,200" → 1200). Dates and times are split into parts.
 * A hyphen between digits is a range ("Zones 1-5", "December 15-18", "2026-27"), not a minus sign:
 * both ends count as stated, exactly as the line check reads them.
 */
function numbersIn(text: string): number[] {
  const out: number[] = [];
  for (const m of text.matchAll(/(?<![\w.])-?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?/g)) {
    const v = Number(m[0].replace(/,/g, ""));
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

function decimals(tok: string): number {
  const i = tok.indexOf(".");
  return i < 0 ? 0 : tok.length - i - 1;
}

/** The text a fact grounds: everything the answerer was shown about it. */
export function factText(f: PortFact): string {
  // a timestamp also grounds its own Pacific calendar day (a buoy reading of 2026-10-13T05:30Z was "measured Oct 12" on this coast)
  return [f.headline, f.value ?? "", f.detail ?? "", f.as_of ?? "", f.as_of && f.as_of.length > 10 ? pacificDay(f.as_of) : "", f.valid_for ? `${f.valid_for.from} ${f.valid_for.to}` : "", f.source.name, f.source.url ?? ""].join(" ");
}

function numberGrounded(tok: string, facts: number[]): boolean {
  const x = Number(tok.replace(/,/g, ""));
  const nd = decimals(tok);
  return facts.some((f) => f === x || Number(f.toFixed(nd)) === x || (0 <= f && f <= 1 && Number((f * 100).toFixed(nd)) === x));
}

const STALE_MEASURED_DAYS = 14;

function ageDays(asOf: string | null, now: Date): number | null {
  if (!asOf) return null;
  const t = Date.parse(asOf.length === 10 ? `${asOf}T12:00:00Z` : asOf);
  return Number.isFinite(t) ? (now.getTime() - t) / 86_400_000 : null;
}

export function check(lines: AnswerLine[], provided: PortFact[], now: Date): GuardReport {
  const byId = new Map(provided.map((f) => [f.id, f]));
  const violations: Violation[] = [];
  const add = (code: ViolationCode, line: number, detail = "") => violations.push({ code, line, detail });
  let firstCitedOfficial: boolean | null = null;
  let anyUnverified = false;
  let topical = false;

  lines.forEach((ln, i) => {
    const body = ln.text;
    const known = ln.cites.map((c) => byId.get(c)).filter((f): f is PortFact => !!f);
    for (const c of ln.cites) {
      if (!byId.has(c)) add("unknown_citation", i, c);
    }
    if (known.length && firstCitedOfficial === null) firstCitedOfficial = known.some((f) => f.basis === "official");
    if (known.some((f) => f.kind === "test" || f.kind === "bloom_forecast" || f.kind === "bloom_outlook")) topical = true;

    if (!ln.cites.length && FACTUAL.test(body) && !(ABSTAIN.test(body) && !/\d/.test(body))) add("uncited_claim", i);

    if (known.length) {
      const grounding = known.map(factText).join(" ");
      const nums = numbersIn(grounding);
      for (const d of body.match(ISO_DATE) ?? []) if (!grounding.includes(d.slice(0, 10))) add("ungrounded_number", i, d);
      const plain = body.replace(ISO_DATE, " ");
      for (const m of plain.matchAll(/(?<![\w.])-?\d[\d,]*(?:\.\d+)?/g)) if (!numberGrounded(m[0], nums)) add("ungrounded_number", i, m[0]);
    }

    for (const m of body.matchAll(SAFETY)) if (!NEG_BEFORE.test(body.slice(0, m.index))) add("safety_claim", i, m[0]);
    const chl = body.match(CHL_TOXIN);
    if (chl && !/\bnot\b|\bisn't\b|\bdoes not\b|\bdoesn't\b|\bnever\b/i.test(chl[0])) add("chlorophyll_as_toxin", i);
    if (known.length && known.every((f) => f.basis === "measured") && FUTURE.test(body) && !NOT_FORECAST.test(body)) add("observation_as_forecast", i);
    if (RESEARCH.test(body) && OVERCLAIM.test(body) && !/\b(not|does not|doesn't|no|matches)\b/i.test(body)) add("research_overclaim", i);
    const lifted = assertsLifted(body);
    if (lifted && !known.some((f) => f.basis === "official" && assertsLifted(factText(f)))) add("lifted_claim", i, lifted);

    for (const f of known) {
      if (f.basis === "official" && f.verified === false) anyUnverified = true;
      const age = ageDays(f.as_of, now);
      // only when the line states this measurement's value (a list of source names does not)
      const own = numbersIn(`${f.headline} ${f.value ?? ""}`.replace(ISO_DATE, " "));
      const states = numbersIn(body.replace(ISO_DATE, " ")).some((n) => own.includes(n));
      const saysDate = !!f.as_of && (body.includes(f.as_of.slice(0, 10)) || new RegExp(`\\b${dayWords(f.as_of, now.toISOString()).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?!\\d|\\s*,\\s*\\d{4})`).test(body));
      if (f.basis === "measured" && states && age !== null && age > STALE_MEASURED_DAYS && f.as_of && !saysDate && !/\b(days?|weeks?|months?) (old|ago)\b/i.test(body))
        add("stale_undated", i, f.id);
    }
  });

  const restrictive = provided.filter((f) => f.basis === "official" && (f.severity === "red" || f.severity === "amber"));
  if (topical && restrictive.length && firstCitedOfficial === false) add("hierarchy", -1, "answer leads with a test or forecast while a restrictive official record was provided");
  const all = lines.map((l) => l.text).join(" ");
  if (anyUnverified && !UNVERIFIED.test(all)) add("unverified_omitted", -1, "official records are pending human review");

  const codes = [...new Set(violations.map((v) => v.code))];
  const badLines = [...new Set(violations.filter((v) => v.line >= 0).map((v) => v.line))];
  return { passed: violations.length === 0, violations, codes, badLines };
}

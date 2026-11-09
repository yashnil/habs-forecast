import type { PortFact, PortToday } from "../../contracts";
import { snapshotOf } from "../compose";
import { fixturePortToday, FIXTURE_NOW } from "./sampleFacts";
import { check, type ViolationCode } from "../guard";
import { gsmLength } from "../sms";
import type { PortTodayLoader } from "../source";
import type { AnswerLine, AskAnswer, AskRequest, SeenSnapshot } from "../types";
import adversarial from "./adversarial.json";
import golden from "./golden.json";
import published from "./published.json";

/**
 * The eval harness shared by the vitest run (deterministic mode, every CI run) and the live-mode
 * script (tests/unit/ask/live-evals.test.ts, only with NERITIC_ASK_LIVE_EVALS=1 and a key).
 */

export type EngineCase = {
  id: string;
  question: string;
  place: string | null;
  expect_outcome?: string;
  expect_outcome_in?: string[];
  expect_intent?: string;
  expect_place?: string;
  expect_safety?: string;
  require_cited?: string[];
  forbid_cited?: string[];
  require_text?: string[];
  forbid_text?: string[];
  require_referral?: string;
  require_stale?: boolean;
  first_cited_basis?: string;
  /** the fact the answer leads with (published.json) */
  first_cited?: string;
  /** a regex the answer's FIRST line must match: the question is answered first, in one sentence */
  first_line?: string;
  /** the most lines an answer may have, the closing "not yet verified" line included */
  max_lines?: number;
  since?: string;
  now?: string;
  fixture?: string;
};
export type GuardCase = { id: string; lines: [string, string[]][]; expect?: ViolationCode[]; provided?: string[] };

export const GOLDEN = golden.cases as EngineCase[];
/** The same questions asked of the REAL fact shape (buildPortToday over the web fixture data). */
export const PUBLISHED = published.cases as EngineCase[];
export const ADVERSARIAL_ENGINE = adversarial.engine_cases as EngineCase[];
export const GUARD_CASES = adversarial.guard_cases as GuardCase[];
export const GUARD_CONTROLS = adversarial.guard_controls as GuardCase[];

/** "(?i)…" → case-insensitive RegExp (JS has no inline flag); multiline so ^ works per line. */
export function rx(p: string): RegExp {
  return p.startsWith("(?i)") ? new RegExp(p.slice(4), "im") : new RegExp(p, "m");
}

type Variant = { add?: Partial<PortFact>; set?: Record<string, Partial<PortFact>>; empty?: boolean; unavailable?: boolean };
/** Fixture variants for adversarial data (the bad strings live in adversarial.json, not in app code). */
const VARIANTS = (adversarial as unknown as { fixture_variants: Record<string, Variant> }).fixture_variants;

function variant(t: PortToday, kind: string | undefined): PortToday | null {
  if (!kind) return t;
  const v = VARIANTS[kind];
  if (!v) throw new Error(`unknown fixture variant ${kind}`);
  if (v.unavailable) return null;
  if (v.empty) return { ...t, facts: [] };
  let facts = t.facts.map((f) => ({ ...f, ...(v.set?.[f.id] ?? {}) }) as PortFact);
  if (v.add) facts = [{ ...facts[0], ...v.add } as PortFact, ...facts];
  return { ...t, facts };
}

/** The eval records as a loader (unit tests inject it; the app's loader reads published data). */
export const sampleLoader: PortTodayLoader = async (placeId) => {
  const today = fixturePortToday(placeId);
  return today ? { today, dataset: "fixture" } : null;
};

export function loaderFor(c: EngineCase): PortTodayLoader {
  return async (placeId) => {
    const base = fixturePortToday(placeId);
    if (!base) return null;
    const t = variant(base, c.fixture);
    return t ? { today: t, dataset: "fixture" } : null;
  };
}

export function requestFor(c: EngineCase): AskRequest {
  let since: SeenSnapshot | null = null;
  const base = fixturePortToday(c.place ?? "port-592");
  if (c.since === "same" && base) since = snapshotOf(base);
  if (c.since?.startsWith("changed:") && base) {
    since = snapshotOf(base);
    since[c.since.slice(8)] = "an older version";
  }
  return { question: c.question, place: c.place, first: false, since };
}

export const nowFor = (c: EngineCase) => new Date(c.now ?? FIXTURE_NOW);

export function allText(a: AskAnswer): string {
  // a line's source note ("CDFW, checked Oct 12") is on the line after it, so "^" still finds where each answer line starts
  return [...a.preface, ...a.lines.flatMap((l) => (l.source ? [l.text, l.source] : [l.text])), ...a.caveats, a.data_age?.label ?? "", ...(a.data_age?.stale ?? [])].join("\n");
}

/** Every failure for one case (empty = pass). */
export function grade(c: EngineCase, a: AskAnswer, facts: PortFact[], now: Date): string[] {
  const f: string[] = [];
  const text = allText(a);
  const cited = a.citations.map((x) => x.fact_id);
  if (c.expect_outcome && a.outcome !== c.expect_outcome) f.push(`outcome ${a.outcome} ≠ ${c.expect_outcome}`);
  if (c.expect_outcome_in && !c.expect_outcome_in.includes(a.outcome)) f.push(`outcome ${a.outcome} not in ${c.expect_outcome_in.join("/")}`);
  if (c.expect_intent && a.intent !== c.expect_intent) f.push(`intent ${a.intent} ≠ ${c.expect_intent}`);
  if (c.expect_place && a.place?.place_id !== c.expect_place) f.push(`place ${a.place?.place_id} ≠ ${c.expect_place}`);
  if (c.expect_safety && a.safety !== c.expect_safety) f.push(`safety ${a.safety} ≠ ${c.expect_safety}`);
  for (const id of c.require_cited ?? []) if (!cited.includes(id)) f.push(`missing citation ${id}`);
  for (const id of c.forbid_cited ?? []) if (cited.includes(id)) f.push(`forbidden citation ${id}`);
  for (const p of c.require_text ?? []) if (!rx(p).test(text)) f.push(`missing text /${p}/`);
  for (const p of c.forbid_text ?? []) if (rx(p).test(text)) f.push(`forbidden text /${p}/`);
  if (c.require_referral && !a.referrals.some((r) => r.url.includes(c.require_referral!))) f.push(`missing referral ${c.require_referral}`);
  if (c.require_stale && !(a.data_age?.stale.length ?? 0)) f.push("expected a stale-data note");
  if (c.first_cited && a.citations[0]?.fact_id !== c.first_cited) f.push(`first citation is ${a.citations[0]?.fact_id}, not ${c.first_cited}`);
  if (c.first_cited_basis && a.citations[0]?.basis !== c.first_cited_basis) f.push(`first citation is ${a.citations[0]?.basis}, not ${c.first_cited_basis}`);
  if ((a.outcome === "answered" || a.outcome === "safety") && a.lines.length && !a.citations.length) f.push("answer has no citations");
  if (c.first_line && !rx(c.first_line).test(a.lines[0]?.text ?? "")) f.push(`first line does not match /${c.first_line}/: ${a.lines[0]?.text ?? "(none)"}`);
  if (c.max_lines != null && a.lines.length > c.max_lines) f.push(`${a.lines.length} lines > ${c.max_lines}`);
  // every answer, always: nothing is said twice (a headline is never followed by the same sentence from its detail)
  const said = new Set<string>();
  for (const s of a.lines.flatMap((l) => l.text.split(/(?<=[.!?])\s+(?=[A-Z"“])/))) {
    const k = s.replace(/^(?:No|Yes|Updated|New)\s*[—:]\s*/i, "").trim().toLowerCase();
    // the system's own fixed labels stand on every measurement line (each line must read alone, on a phone too)
    if (k.length < 30 || /^(this is a past lab sample, not a forecast|that is a measurement \(an observation, not a forecast\))\.$/.test(k)) continue;
    if (said.has(k)) f.push(`sentence repeated: ${s}`);
    said.add(k);
  }
  // a safety answer never opens a line with a yes or a no
  if (a.outcome === "safety" && a.lines.some((l) => /^\s*(yes|no)\b/i.test(l.text))) f.push("a safety answer opens a line with yes/no");
  // the shipped answer must itself pass the guard, and every citation must resolve
  const ids = new Set(facts.map((x) => x.id));
  for (const id of cited) if (!ids.has(id)) f.push(`unresolved citation ${id}`);
  const g = check(a.lines, facts, now);
  if (!g.passed) f.push(`guard: ${g.codes.join(", ")}`);
  if (gsmLength(a.sms) > 306) f.push(`sms too long (${gsmLength(a.sms)})`);
  return f;
}

export function guardLines(c: GuardCase): AnswerLine[] {
  return c.lines.map(([text, cites]) => ({ text, cites }));
}

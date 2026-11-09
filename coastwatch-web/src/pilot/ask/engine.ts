import type { PortFact, PortToday } from "../contracts";
import { dataAge } from "./age";
import { composeIntent, relatedForSafety, snapshotOf } from "./compose";
import { check } from "./guard";
import { routeIntent, safetyKind } from "./intents";
import { PILOT_PORT_NAMES, resolvePlace, type PilotPort } from "./places";
import { smsFromAnswer } from "./sms";
import { loadPortToday, type PortTodayLoader } from "./source";
import { DISCLAIMER, HELP_LINES, safetyTemplate, STANDING_REFERRALS } from "./templates";
import type { AnswerLine, AskAnswer, AskMode, AskRequest, Citation, Dataset, Intent, Outcome, Referral, SafetyKind } from "./types";

/**
 * Ask Neritic's orchestration: the deterministic safety pre-pass, place and intent routing,
 * retrieval of ONE port's PortFacts, the answerer (deterministic here; live.ts plugs in a model),
 * and the deterministic post-pass guard that every answer goes through before anyone sees it.
 */

export const MAX_QUESTION = 400;

/** Text in a data field that reads like an instruction is data we refuse to repeat or forward. */
const SUSPECT = /\b(ignore|disregard|forget)\b[^.]{0,30}\b(previous|prior|above|earlier|all|your)\b[^.]{0,20}\b(instructions?|rules?|prompts?)\b|\bsystem prompt\b|\byou are now\b|\bnew instructions?\b|<\/?(system|instructions?|assistant)>|\bassistant:\s/i;

export function quarantine(facts: PortFact[]): { kept: PortFact[]; withheld: PortFact[] } {
  const kept: PortFact[] = [];
  const withheld: PortFact[] = [];
  for (const f of facts) (SUSPECT.test(`${f.headline} ${f.detail ?? ""} ${f.value ?? ""} ${f.source.name}`) ? withheld : kept).push(f);
  return { kept, withheld };
}

export type AskContext = {
  now: Date;
  mode: AskMode;
  loader?: PortTodayLoader;
};

export type Retrieved = { port: PilotPort; today: PortToday; dataset: Dataset; withheld: number };

export function citationsFor(lines: AnswerLine[], facts: PortFact[]): Citation[] {
  const byId = new Map(facts.map((f) => [f.id, f]));
  const ids = [...new Set(lines.flatMap((l) => l.cites))].filter((id) => byId.has(id));
  return ids.map((id) => {
    const f = byId.get(id)!;
    return {
      fact_id: f.id,
      kind: f.kind,
      basis: f.basis,
      headline: f.headline,
      source_name: f.source.name,
      source_url: f.source.url,
      as_of: f.as_of,
      verified: f.verified,
      today_href: `/today?place=${encodeURIComponent(f.place_id)}#fact-${encodeURIComponent(f.id)}`,
    };
  });
}

/**
 * The post-pass. Lines that break a rule are removed (never repaired with new words); an answer
 * that cites unverified official records without saying so gets the fixed unverified line.
 */
export function enforce(lines: AnswerLine[], facts: PortFact[], now: Date): { lines: AnswerLine[]; removed: number; codes: string[]; passed: boolean } {
  const first = check(lines, facts, now);
  if (first.passed) return { lines, removed: 0, codes: [], passed: true };
  let kept = lines.filter((_, i) => !first.badLines.includes(i));
  if (first.codes.includes("hierarchy")) {
    // lead with official records: move official-citing lines first, keep order otherwise
    const byId = new Map(facts.map((f) => [f.id, f]));
    const isOfficial = (l: AnswerLine) => l.cites.some((c) => byId.get(c)?.basis === "official");
    kept = [...kept.filter(isOfficial), ...kept.filter((l) => !isOfficial(l))];
  }
  const again = check(kept, facts, now);
  if (again.codes.includes("unverified_omitted")) {
    const un = [...new Set(kept.flatMap((l) => l.cites))].filter((id) => facts.find((f) => f.id === id && f.basis === "official" && f.verified === false));
    kept.push({ text: "These official records were transcribed by Neritic and are not yet verified by a person; the agency page is the authority.", cites: un });
  }
  // a residual global failure (e.g. hierarchy with no official line left) drops the topical lines
  const last = check(kept, facts, now);
  if (!last.passed) kept = kept.filter((_, i) => !last.badLines.includes(i));
  const final = check(kept, facts, now);
  if (!final.passed && final.codes.includes("hierarchy")) {
    const byId = new Map(facts.map((f) => [f.id, f]));
    kept = kept.filter((l) => l.cites.every((c) => byId.get(c)?.basis === "official") || !l.cites.length);
  }
  return { lines: kept, removed: lines.filter((l) => !kept.includes(l)).length, codes: first.codes, passed: false };
}

type Draft = {
  outcome: Outcome;
  intent: Intent;
  safety: SafetyKind | null;
  port: PilotPort | null;
  preface: string[];
  lines: AnswerLine[];
  caveats: string[];
  referrals: Referral[];
  safetySms?: string | null;
};

/** Everything that is not the answerer: the guard, citations, data age, SMS text, disclaimer. */
export function finalize(d: Draft, r: Retrieved | null, ctx: AskContext, req: AskRequest, model: string | null = null): AskAnswer {
  const facts = r?.today.facts ?? [];
  const g = enforce(d.lines, facts, ctx.now);
  const caveats = [...d.caveats];
  if (g.removed) caveats.push(`${g.removed} line${g.removed === 1 ? " was" : "s were"} withheld because ${g.removed === 1 ? "it" : "they"} failed Neritic's safety check; see the sources.`);
  if (r?.withheld) caveats.push(`${r.withheld} record${r.withheld === 1 ? " was" : "s were"} withheld because its text looked like an instruction rather than data.`);
  const citations = citationsFor(g.lines, facts);
  const cited = facts.filter((f) => citations.some((c) => c.fact_id === f.id));
  const base = {
    outcome: d.outcome,
    intent: d.intent,
    safety: d.safety,
    place: d.port ? { place_id: d.port.place_id, label: d.port.label } : null,
    preface: d.preface,
    lines: g.lines,
    caveats,
    citations,
    data_age: dataAge(cited, ctx.now),
    referrals: d.referrals,
    notice: req.first ? DISCLAIMER : null,
    mode: ctx.mode,
    model,
    dataset: r?.dataset ?? "fixture",
    guard: { passed: g.passed, removed: g.removed, codes: g.codes },
    snapshot: r ? snapshotOf(r.today) : null,
  };
  return { ...base, sms: smsFromAnswer(base, d.safetySms) };
}

export type Routed =
  | { kind: "final"; draft: Draft; retrieved: Retrieved | null }
  | { kind: "answer"; intent: Intent; safety: SafetyKind | null; retrieved: Retrieved; question: string };

/** Pre-pass + routing + retrieval. Returns either a finished draft or what an answerer needs. */
export async function route(req: AskRequest, ctx: AskContext, intentOverride?: Intent): Promise<Routed> {
  const q = (req.question ?? "").slice(0, MAX_QUESTION).trim();
  const safety = safetyKind(q);
  const intent = intentOverride ?? routeIntent(q);
  const place = resolvePlace(q, req.place);
  const base = { intent, safety, preface: [] as string[], lines: [] as AnswerLine[], caveats: [] as string[], referrals: [] as Referral[] };

  if (!q || (intent === "help" && !safety)) return { kind: "final", retrieved: null, draft: { ...base, outcome: "help", port: place.kind === "port" ? place.port : null, preface: HELP_LINES } };

  if (place.kind === "out_of_area") {
    const t = safety ? safetyTemplate(safety, null) : null;
    return {
      kind: "final",
      retrieved: null,
      draft: {
        ...base,
        outcome: "out_of_area",
        port: null,
        preface: [...(t?.lines.slice(0, 2) ?? []), `Ask Neritic's pilot covers only ${PILOT_PORT_NAMES}. It has no records for ${place.named}, so it can't answer that.`],
        referrals: t?.referrals ?? STANDING_REFERRALS,
        safetySms: t ? t.sms : null,
      },
    };
  }
  if (place.kind === "none") {
    const t = safety ? safetyTemplate(safety, null) : null;
    return {
      kind: "final",
      retrieved: null,
      draft: { ...base, outcome: "need_place", port: null, preface: [...(t?.lines.slice(0, 2) ?? []), `Which port? Ask Neritic's pilot covers ${PILOT_PORT_NAMES}.`], referrals: t?.referrals ?? [], safetySms: t?.sms ?? null },
    };
  }

  const port = place.port;
  const loaded = await (ctx.loader ?? loadPortToday)(port.place_id, ctx.now);
  if (!loaded) return { kind: "final", retrieved: null, draft: { ...base, outcome: "no_data", port, preface: [`Neritic's data for ${port.label} is unavailable right now. Check CDFW and CDPH directly.`], referrals: STANDING_REFERRALS } };
  const { kept, withheld } = quarantine(loaded.today.facts);
  const retrieved: Retrieved = { port, today: { ...loaded.today, facts: kept }, dataset: loaded.dataset, withheld: withheld.length };

  if (safety) {
    const t = safetyTemplate(safety, port);
    return {
      kind: "final",
      retrieved,
      draft: { ...base, outcome: "safety", port, preface: t.lines, lines: relatedForSafety(safety, retrieved.today, q, ctx.now), caveats: [], referrals: t.referrals, safetySms: t.sms },
    };
  }
  if (intent === "unknown")
    return {
      kind: "final",
      retrieved,
      draft: { ...base, outcome: "abstained", port, preface: [`I can only answer questions about ${port.label} from Neritic's records.`, ...HELP_LINES] },
    };
  return { kind: "answer", intent, safety, retrieved, question: q };
}

/** readiness "demo": the deterministic grounded answerer. */
export async function askDeterministic(req: AskRequest, ctx: AskContext): Promise<AskAnswer> {
  const r = await route(req, ctx);
  if (r.kind === "final") return finalize(r.draft, r.retrieved, ctx, req);
  const c = composeIntent(r.intent, r.retrieved.today, r.retrieved.port, r.question, ctx.now, req.since);
  return finalize({ ...c, intent: r.intent, safety: null, port: r.retrieved.port, preface: [] }, r.retrieved, ctx, req);
}


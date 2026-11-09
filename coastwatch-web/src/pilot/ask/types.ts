import type { PortFact, Readiness } from "../contracts";

/**
 * Ask Neritic (pilot track Q): the shapes shared by the deterministic answerer, the live (Claude)
 * answerer, the safety guard, the web chat and the SMS webhook.
 *
 * An answer is a list of lines. Each line carries the ids of the PortFacts it rests on; the guard
 * checks every line against exactly those facts (numbers, dates, safety words, hierarchy). Text the
 * system writes itself (data age, the disclaimer, safety referrals) lives in separate fields and is
 * never mixed into the guarded lines.
 */

export type Intent =
  | "status" // what is open, restricted, delayed for a fishery
  | "toxin" // latest CDPH toxin tests
  | "sea" // sea, wind, weather, tide (forecast or observation)
  | "bloom" // C-HARM 0–3 day risk + 8-day chlorophyll outlook
  | "whale" // RAMP whale-entanglement rules
  | "changed" // what changed since the last visit
  | "sources" // where does this come from
  | "data_age" // how old is the data
  | "help"
  | "unknown";

/** Questions the guard answers only with a fixed template (official referrals first). */
export type SafetyKind = "eat" | "go_out" | "bar" | "fish_legal";

export type Fishery =
  | "dungeness"
  | "rock_crab"
  | "salmon"
  | "groundfish"
  | "anchovy_sardine"
  | "squid"
  | "shellfish"
  | "lobster";

export type Outcome =
  | "answered"
  | "safety" // fixed template + related facts as information only
  | "no_data" // the data does not cover the question (said plainly)
  | "out_of_area"
  | "need_place"
  | "abstained" // off-topic, or the question could not be understood
  | "help"
  | "rate_limited";

/** `primary`: the line that answers the question itself (SMS shows primary lines first). */
export type AnswerLine = { text: string; cites: string[]; primary?: boolean };

export type Citation = {
  fact_id: string;
  kind: PortFact["kind"];
  basis: PortFact["basis"];
  headline: string;
  source_name: string;
  source_url: string | null;
  as_of: string | null;
  verified: boolean | null;
  /** the Today card for this fact (Today track renders `id="fact-<id>"` on each card) */
  today_href: string;
};

export type Referral = { label: string; url: string; why: string };

export type DataAge = {
  /** e.g. "Oldest record in this answer: CDPH test of 2026-09-28 (41 days old)." */
  label: string;
  oldest_as_of: string | null;
  newest_as_of: string | null;
  /** one plain note per stale record ("C-HARM run of 2026-11-02 is past its 0–3 day window") */
  stale: string[];
};

export type AskMode = "demo" | "live" | "live_fallback";
export type Dataset = "fixture" | "published";

export type AskAnswer = {
  outcome: Outcome;
  intent: Intent;
  safety: SafetyKind | null;
  place: { place_id: string; label: string } | null;
  /** fixed template text (safety answers, help), never built from data and not cited */
  preface: string[];
  /** the answer built from facts; every line is checked by the guard */
  lines: AnswerLine[];
  /** fixed caveats shown after the answer ("Chlorophyll is not toxin", "1 record withheld", …) */
  caveats: string[];
  citations: Citation[];
  data_age: DataAge | null;
  referrals: Referral[];
  /** the standing disclaimer, sent with the first exchange */
  notice: string | null;
  mode: AskMode;
  model: string | null;
  dataset: Dataset;
  guard: { passed: boolean; removed: number; codes: string[] };
  /** signatures of every fact at this port now; the client keeps it for the next "what changed" */
  snapshot: SeenSnapshot | null;
  /** the exact text a phone would receive for this answer (≤ 2 SMS segments) */
  sms: string;
};

/** What the client remembers between visits for "what changed" (fact id → signature). */
export type SeenSnapshot = Record<string, string>;

export type AskRequest = {
  question: string;
  place?: string | null;
  first?: boolean;
  since?: SeenSnapshot | null;
  channel?: "web" | "sms";
};

export type AskMeta = {
  readiness: { ask: Readiness; sms: Readiness };
  logging: boolean;
  dataset: Dataset;
};

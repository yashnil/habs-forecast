import type { FactKind, PortFact, PortToday, Severity } from "@/pilot/contracts";

/**
 * Alerts: the differences between the PortToday facts a person (or a subscription) last saw and the
 * current ones, for the triggers they chose. The same rules run on the device (in-app inbox, no
 * account) and on the server (alerts-fanout), so both say the same thing.
 *
 * What an alert may say: what changed, the source, the date that applies, and what the fact is
 * epistemically (official / measured / forecast). Never "safe"; a bloom-risk alert is a model forecast
 * (C-HARM), never a toxin result; chlorophyll outlooks never raise an alert.
 */
export const TRIGGERS = ["status_change", "new_toxin_result", "ramp_action", "bloom_risk_jump", "data_stale"] as const;
export type AlertTrigger = (typeof TRIGGERS)[number];

export const TRIGGER_LABELS: Record<AlertTrigger, { label: string; hint: string }> = {
  status_change: { label: "Official status changes", hint: "A closure, delay, restriction or advisory is added, changed or no longer listed" },
  new_toxin_result: { label: "New toxin test results", hint: "CDPH posts a new domoic acid result for a site near the port" },
  ramp_action: { label: "Whale-entanglement (RAMP) actions", hint: "CDFW changes a RAMP action or assessment date" },
  bloom_risk_jump: { label: "Bloom risk jumps", hint: "NOAA C-HARM's forecast probability rises sharply (a model forecast, not a toxin test)" },
  data_stale: { label: "Data goes stale", hint: "Neritic's information for the port gets too old to rely on" },
};

/** a C-HARM probability rise of at least this many percentage points is a "jump" */
export const BLOOM_JUMP_PTS = 20;

const KIND_TRIGGER: Partial<Record<FactKind, AlertTrigger>> = {
  status: "status_change",
  notice: "status_change",
  test: "new_toxin_result",
  whale: "ramp_action",
  bloom_forecast: "bloom_risk_jump",
  data_age: "data_stale",
};

export function triggerFor(fact: Pick<PortFact, "kind">): AlertTrigger | null {
  return KIND_TRIGGER[fact.kind] ?? null;
}

const RANK: Record<Severity, number> = { info: 0, neutral: 1, amber: 2, red: 3 };

/** The part of a fact that makes it "different": a re-check date alone is not a change. */
export type FactSig = Pick<PortFact, "kind" | "severity" | "headline"> & { value: string | null; as_of: string | null };

export function significant(f: PortFact): FactSig {
  const base = { kind: f.kind, severity: f.severity, headline: f.headline, value: f.value ?? null };
  // a new toxin result is a new sample date; everything else changes by what it says, not when it was re-read
  return { ...base, as_of: f.kind === "test" ? (f.as_of ?? null) : null };
}

/** FNV-1a 32-bit over a canonical string: stable in the browser and on the server. */
export function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function factVersion(f: PortFact): string {
  const s = significant(f);
  return hash([s.kind, s.severity, s.headline, s.value ?? "", s.as_of ?? ""].join("␟"));
}

/** What was remembered about one fact: its version and significant fields (enough to describe a change). */
export type SeenFact = { v: string; sig: FactSig };
/** What was remembered about one place: fact id → seen fact. */
export type PlaceSnapshot = { place_id: string; seen_at: string; facts: Record<string, SeenFact> };

export function snapshotOf(today: PortToday, seenAt: string): PlaceSnapshot {
  const facts: Record<string, SeenFact> = {};
  for (const f of today.facts) if (triggerFor(f)) facts[f.id] = { v: factVersion(f), sig: significant(f) };
  return { place_id: today.place_id, seen_at: seenAt, facts };
}

export type AlertItem = {
  /** idempotency key: place · fact · version (one alert per version, however often the check runs) */
  key: string;
  place_id: string;
  place_label: string;
  fact_id: string;
  fact_version: string;
  trigger: AlertTrigger;
  severity: Severity;
  title: string;
  body: string;
  basis: PortFact["basis"] | null;
  verified: boolean | null;
  source: { name: string; url: string | null } | null;
  as_of: string | null;
  created_at: string;
};

const pct = (v: string | null | undefined): number | null => {
  const m = v?.match(/(\d+(?:\.\d+)?)\s*%/);
  return m ? Number(m[1]) : null;
};

function basisNote(f: PortFact): string {
  if (f.basis === "official" && f.verified === false) return " From an official source; Neritic's copy is not yet verified by a person — check the source before acting.";
  if (f.kind === "bloom_forecast") return " This is NOAA C-HARM's model forecast, not a toxin test.";
  return "";
}

/** Is the change from `prev` to `next` worth an alert for this trigger? (`prev` null = newly listed) */
export function worthAlert(trigger: AlertTrigger, prev: FactSig | null, next: PortFact): boolean {
  switch (trigger) {
    case "status_change":
    case "ramp_action":
      return !prev || prev.severity !== next.severity || prev.headline !== next.headline || prev.value !== (next.value ?? null);
    case "new_toxin_result":
      return !prev || prev.as_of !== (next.as_of ?? null) || prev.value !== (next.value ?? null);
    case "bloom_risk_jump": {
      if (!prev) return RANK[next.severity] >= RANK.amber;
      if (RANK[next.severity] > RANK[prev.severity] && RANK[next.severity] >= RANK.amber) return true;
      const a = pct(prev.value);
      const b = pct(next.value);
      return a != null && b != null && b - a >= BLOOM_JUMP_PTS;
    }
    case "data_stale":
      return RANK[next.severity] >= RANK.amber && (!prev || RANK[next.severity] > RANK[prev.severity]);
  }
}

function describe(trigger: AlertTrigger, prev: FactSig | null, f: PortFact, since: string | null): string {
  const when = since ? ` since ${since.slice(0, 10)}` : "";
  const src = f.source?.name ? ` Source: ${f.source.name}${f.as_of ? `, ${f.as_of.slice(0, 10)}` : ""}.` : "";
  let lead: string;
  if (!prev) lead = trigger === "new_toxin_result" ? "New result listed." : "Newly listed for this port.";
  else if (trigger === "new_toxin_result") lead = `New result${when}. Previous: ${prev.value ?? prev.headline}.`;
  else if (trigger === "bloom_risk_jump") lead = `Bloom risk rose${when}. Was: ${prev.value ?? prev.headline}.`;
  else if (trigger === "data_stale") lead = "Neritic's information here is getting old. Check the official sources directly.";
  else lead = `Changed${when}. Was: “${prev.headline}”.`;
  return `${lead}${f.detail ? ` ${f.detail}` : ""}${src}${basisNote(f)}`.trim();
}

/**
 * Compare a place's remembered snapshot with today's facts. A first look (no snapshot) produces no
 * alerts: alerts begin from what the person has already seen. Facts that disappear raise a
 * status_change alert ("no longer listed; reason unknown") for status/notice facts only.
 */
export function diffFacts(prev: PlaceSnapshot | null, today: PortToday, triggers: readonly AlertTrigger[], now: string): AlertItem[] {
  if (!prev) return [];
  const want = new Set(triggers);
  const out: AlertItem[] = [];
  const present = new Set<string>();
  for (const f of today.facts) {
    const t = triggerFor(f);
    if (!t) continue;
    present.add(f.id);
    const v = factVersion(f);
    const was = prev.facts[f.id] ?? null;
    if (was?.v === v || !want.has(t) || !worthAlert(t, was?.sig ?? null, f)) continue;
    out.push({
      key: `${today.place_id}|${f.id}|${v}`,
      place_id: today.place_id,
      place_label: today.label,
      fact_id: f.id,
      fact_version: v,
      trigger: t,
      severity: f.severity,
      title: `${today.label}: ${f.headline}`,
      body: describe(t, was?.sig ?? null, f, prev.seen_at),
      basis: f.basis,
      verified: f.verified,
      source: f.source ?? null,
      as_of: f.as_of ?? null,
      created_at: now,
    });
  }
  if (want.has("status_change")) {
    for (const [id, was] of Object.entries(prev.facts)) {
      if (present.has(id) || (was.sig.kind !== "status" && was.sig.kind !== "notice")) continue;
      const v = `gone-${was.v}`;
      out.push({
        key: `${today.place_id}|${id}|${v}`,
        place_id: today.place_id,
        place_label: today.label,
        fact_id: id,
        fact_version: v,
        trigger: "status_change",
        severity: "info",
        title: `${today.label}: no longer listed — ${was.sig.headline}`,
        body: "Neritic no longer lists this for the port. The reason is unknown: check the official source before acting.",
        basis: null,
        verified: null,
        source: null,
        as_of: null,
        created_at: now,
      });
    }
  }
  // official status first, then the rest, most severe first
  return out.sort((a, b) => Number(b.trigger === "status_change") - Number(a.trigger === "status_change") || RANK[b.severity] - RANK[a.severity]);
}

import { day } from "@/nx/ds/format";
import { forecastChanges } from "./forecastDiff";
import { norm, readOk, SOURCE_KEYS, SOURCE_LABEL, SOURCE_OF, VERSION_FIELDS } from "./snapshot";
import type { Change, ChangeCategory, EvidenceItem, Snapshot, SourceKey } from "./types";

/**
 * What changed between two snapshots of one trip (v15 §8). Pure.
 *
 * The rules, carried over from the alert check (src/nx/alerts/check.ts):
 *  - identity is the evidence key, change is a new version; a refetch of the same records finds nothing;
 *  - text that differs only in whitespace or case is the same text (the version is hashed from normalized words, and
 *    the field comparison here normalizes again), so a parsing difference is never a change;
 *  - what a check could not see is never "new" and never "removed": a record missing because its source failed is
 *    "couldn't check", and a record first seen once a source answers again is told with that source's return;
 *  - a copy older than the one compared against (a stale cache) is not read as news;
 *  - a disagreement between sources is said every time it is still there, never resolved here.
 * Every title is a description with its source; none is advice.
 *
 * Official records, CDPH tests and source availability are compared here; the forecast, NWS advisories and buoy
 * readings by Agent B's forecastDiff.ts (same-zone, same-valid-time rules), whose changes are merged and ranked here.
 */

const CATEGORY: Record<EvidenceItem["kind"], ChangeCategory> = { season: "official", whale: "official", cdphTest: "toxin", advisory: "advisory", observation: "observation" };
const PRIORITY: Record<ChangeCategory, 1 | 2 | 3> = { official: 1, toxin: 1, forecast: 2, advisory: 2, observation: 3, source: 3 };
const ORDER: ChangeCategory[] = ["official", "toxin", "advisory", "forecast", "source", "observation"];
const OFFICIAL_SOURCES: SourceKey[] = ["season", "whale", "cdphTest"];
/** compared by forecastDiff.ts */
const DELEGATED = new Set<EvidenceItem["kind"]>(["advisory", "observation"]);

const FIELD_WORDS: Record<string, string> = {
  level: "status",
  text: "wording",
  conflict: "agreement between sources",
  differ: "what the sources say",
  verified: "review by a person",
  seasonal: "season calendar",
  measures: "measures",
  next: "next assessment",
  date: "sample date",
  values: "measured values",
  tissue: "tissue",
  action: "action level",
  event: "advisory type",
  headline: "advisory wording",
  ends: "end time",
};

const show = (v: unknown) => (v == null || v === "" ? "none" : typeof v === "boolean" ? (v ? "yes" : "no") : String(v));
const same = (a: unknown, b: unknown) => (typeof a === "string" || typeof b === "string" ? norm(a as string) === norm(b as string) : (a ?? null) === (b ?? null));

/** The version fields that really differ (normalized), in their reader words. */
export function changedFields(a: EvidenceItem, b: EvidenceItem): string[] {
  return VERSION_FIELDS[b.kind].filter((k) => !same(a.fields[k], b.fields[k]));
}

const agency = (e: EvidenceItem) => e.source;

function revised(a: EvidenceItem, b: EvidenceItem, fields: string[]): Change {
  const words = fields.map((f) => FIELD_WORDS[f] ?? f);
  const changes = fields.filter((f) => f !== "text" && f !== "headline" && f !== "measures").map((f) => `${FIELD_WORDS[f] ?? f}: ${show(a.fields[f])} → ${show(b.fields[f])}`);
  const base = {
    id: `revised:${b.key}:${b.version}`,
    priority: PRIORITY[CATEGORY[b.kind]],
    category: CATEGORY[b.kind],
    kind: "revised" as const,
    detail: `Changed: ${words.join(", ")}.${changes.length ? ` ${changes.join("; ")}.` : ""}`,
    before: a.text,
    after: b.text,
    where: b.scope,
    when: b.asOf,
    url: b.url ?? a.url,
    periodKey: null,
    evidenceKey: b.key,
  };
  switch (b.kind) {
    case "season":
      return { ...base, title: `Agency information changed; review the original ${agency(b)} page.` };
    case "whale":
      return { ...base, title: `${agency(b)} whale measures changed in ${b.scope ?? "your zone"}.` };
    case "cdphTest":
      return { ...base, title: "CDPH changed the published test record; inspect measured values and dates." };
    case "advisory":
      return { ...base, title: `NWS revised the ${b.title}.` };
    case "observation":
      return { ...base, title: `Newer reading at ${b.title}.` };
  }
}

function added(b: EvidenceItem): Change {
  const base = {
    id: `new:${b.key}:${b.version}`,
    priority: PRIORITY[CATEGORY[b.kind]],
    category: CATEGORY[b.kind],
    kind: "new" as const,
    detail: b.text,
    before: null,
    after: b.text,
    where: b.scope,
    when: b.asOf,
    url: b.url,
    periodKey: null,
    evidenceKey: b.key,
  };
  switch (b.kind) {
    case "season":
      return { ...base, title: `New ${agency(b)} record for ${b.scope ?? "this fishery"} since you planned; review the original page.` };
    case "whale":
      return { ...base, title: `New ${agency(b)} whale measure in ${b.scope ?? "your zone"}.` };
    case "cdphTest":
      return { ...base, title: "New CDPH test published; inspect measured values and dates." };
    case "advisory":
      return { ...base, title: `New NWS ${b.title} since you planned.` };
    case "observation":
      return { ...base, title: `A reading is now available at ${b.title}.` };
  }
}

function gone(a: EvidenceItem, now: string): Change {
  const ended = a.kind === "advisory" && typeof a.fields.ends === "string" && Date.parse(a.fields.ends) <= Date.parse(now);
  const title =
    a.kind === "season"
      ? `The ${agency(a)} record shown when you planned is no longer listed; review the original page.`
      : a.kind === "whale"
        ? `The ${agency(a)} whale measure shown for ${a.scope ?? "your zone"} is no longer listed.`
        : a.kind === "cdphTest"
          ? "The CDPH test shown when you planned is no longer listed."
          : a.kind === "advisory"
            ? ended
              ? `The NWS ${a.title} has ended.`
              : `The NWS ${a.title} is no longer listed.`
            : `${a.title} is no longer reporting in the answer.`;
  return {
    id: `removed:${a.key}:${a.version}`,
    priority: PRIORITY[CATEGORY[a.kind]],
    category: CATEGORY[a.kind],
    kind: ended ? "expired" : "removed",
    title,
    detail: `Shown when you planned: ${a.text}`,
    before: a.text,
    after: null,
    where: a.scope,
    when: ended ? (a.fields.ends as string) : null,
    url: a.url,
    periodKey: null,
    evidenceKey: a.key,
  };
}

/** A copy of a CDPH test older than the one compared against: a stale cache, never news. */
const olderTest = (a: EvidenceItem, b: EvidenceItem) => typeof a.fields.date === "string" && typeof b.fields.date === "string" && b.fields.date < a.fields.date;
const olderReading = (a: EvidenceItem, b: EvidenceItem) => typeof a.fields.at === "string" && typeof b.fields.at === "string" && Date.parse(b.fields.at) <= Date.parse(a.fields.at);

export function diffSnapshots(base: Snapshot, next: Snapshot, now: string): Change[] {
  const out: Change[] = [];
  const A = new Map(base.evidence.map((e) => [e.key, e]));
  const B = new Map(next.evidence.map((e) => [e.key, e]));

  for (const [key, b] of B) {
    if (DELEGATED.has(b.kind)) continue;
    const a = A.get(key);
    const src = SOURCE_OF[b.kind];
    // conflicts are preserved: said while they last
    if (b.conflict) {
      const persists = !!a?.conflict;
      out.push({
        id: `conflict:${key}`,
        priority: 1,
        category: CATEGORY[b.kind],
        kind: "conflict",
        title: persists ? `Sources still disagree about ${b.scope ?? "this fishery"}; awaiting a person's review.` : `Sources now disagree about ${b.scope ?? "this fishery"}; awaiting a person's review.`,
        detail: b.conflict,
        before: a?.text ?? null,
        after: b.text,
        where: b.scope,
        when: b.asOf,
        url: b.url,
        periodKey: null,
        evidenceKey: key,
      });
    }
    if (!a) {
      // a record first seen because its source answers again is not new; the source's return says it
      if (!readOk(base.sources[src])) continue;
      out.push(added(b));
      continue;
    }
    if (a.version === b.version) {
      if (b.stale && !a.stale && b.kind === "cdphTest") out.push(staleTest(b, now));
      continue;
    }
    if (b.kind === "cdphTest" && olderTest(a, b)) {
      out.push(olderCopy(b, a));
      continue;
    }
    if (b.kind === "observation" && olderReading(a, b)) continue;
    const fields = changedFields(a, b);
    if (!fields.length) continue; // the hash moved, the meaning did not (a parsing difference)
    if (b.kind === "cdphTest" && fields.includes("date")) {
      out.push({ ...added(b), id: `new:${key}:${b.version}`, before: a.text, detail: `Sampled ${day(b.fields.date as string, { ref: now })}; the test shown when you planned was sampled ${day(a.fields.date as string, { ref: now })}.` });
      continue;
    }
    // a conflict appearing is already said above; its other fields may still have moved
    const rest = fields.filter((f) => !(b.conflict && (f === "conflict" || f === "differ" || f === "text")));
    if (rest.length) out.push(revised(a, b, rest));
  }

  for (const [key, a] of A) {
    if (B.has(key) || DELEGATED.has(a.kind)) continue;
    const src = SOURCE_OF[a.kind];
    // gone only when the source answered; a failed read is "couldn't check" (the source change below says it)
    if (!readOk(next.sources[src])) continue;
    if (a.kind === "observation") continue; // a buoy dropping out of the answer is told by the buoy source itself
    out.push(gone(a, now));
  }

  for (const k of SOURCE_KEYS) {
    const was = base.sources[k];
    const is = next.sources[k];
    if (!was || !is) continue;
    if (readOk(was) && !readOk(is)) {
      const hidden = base.evidence.filter((e) => SOURCE_OF[e.kind] === k);
      const official = OFFICIAL_SOURCES.includes(k) && hidden.length > 0;
      out.push({
        id: `source:${k}:unavailable:${next.mode}`,
        priority: official ? 1 : hidden.length && k === "nwsAlerts" ? 2 : 3,
        category: "source",
        kind: "unavailable",
        title: next.mode === "saved" ? `${SOURCE_LABEL[k]}: not checked; compared against the copy saved on this device.` : `${SOURCE_LABEL[k]} could not be checked.`,
        detail: [is.detail, hidden.length ? `Shown when you planned: ${hidden.map((e) => e.title).join("; ")}. Not compared.` : null].filter(Boolean).join(" ") || null,
        before: null,
        after: null,
        where: null,
        when: next.takenAt,
        url: hidden[0]?.url ?? null,
        periodKey: null,
        evidenceKey: null,
      });
    } else if (!readOk(was) && readOk(is)) {
      const seen = next.evidence.filter((e) => SOURCE_OF[e.kind] === k);
      const official = OFFICIAL_SOURCES.includes(k) && seen.length > 0;
      out.push({
        id: `source:${k}:available:${seen.map((e) => e.version).join(",")}`,
        priority: official ? 1 : 3,
        category: "source",
        kind: "available",
        title: `${SOURCE_LABEL[k]} answered again; it could not be read when you planned.`,
        detail: seen.length ? `Now on record: ${seen.map((e) => `${e.title}: ${e.text}`).join(" ")}` : is.detail,
        before: null,
        after: seen[0]?.text ?? null,
        where: seen[0]?.scope ?? null,
        when: is.at,
        url: seen[0]?.url ?? null,
        periodKey: null,
        evidenceKey: seen[0]?.key ?? null,
      });
    }
  }

  // a reading first seen because the buoy answers again is not new (the source's return says it)
  out.push(...forecastChanges(base, next, now).filter((c) => !(c.category === "observation" && c.kind === "new" && !readOk(base.sources.buoy))));
  return rank(out);
}

function staleTest(b: EvidenceItem, now: string): Change {
  return {
    id: `stale:${b.key}:${b.version}`,
    priority: 1,
    category: "toxin",
    kind: "stale",
    title: `CDPH's latest test is now old: sampled ${day(b.fields.date as string, { ref: now })}. It says nothing about today.`,
    detail: null,
    before: null,
    after: b.text,
    where: b.scope,
    when: b.asOf,
    url: b.url,
    periodKey: null,
    evidenceKey: b.key,
  };
}

function olderCopy(b: EvidenceItem, a: EvidenceItem): Change {
  return {
    id: `older:${b.key}:${b.version}`,
    priority: 3,
    category: "source",
    kind: "stale",
    title: "The CDPH copy read now is older than the one shown when you planned; not compared.",
    detail: `Shown when you planned: ${a.text} Read now: ${b.text}`,
    before: a.text,
    after: b.text,
    where: b.scope,
    when: b.asOf,
    url: b.url,
    periodKey: null,
    evidenceKey: b.key,
  };
}

/** Dedupe by id, most important first, then by category, then the order found. */
export function rank(changes: Change[]): Change[] {
  const seen = new Map<string, Change>();
  for (const c of changes) if (!seen.has(c.id)) seen.set(c.id, c);
  return [...seen.values()]
    .map((c, i) => ({ c, i }))
    .sort((x, y) => x.c.priority - y.c.priority || ORDER.indexOf(x.c.category) - ORDER.indexOf(y.c.category) || x.i - y.i)
    .map(({ c }) => c);
}

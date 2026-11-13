import type { Change, EvidenceItem, ForecastPeriod, Snapshot } from "./types";
import { directionSequence, keyLabel, NWS_TZ, seasRangeWords, windRangeWords, zonedIso } from "./forecast";
import type { AdvisoryIn } from "./windows";
import { dayTime } from "@/nx/ds/format";

/**
 * What changed in the marine forecast, NWS advisories and nearby measurements between two snapshots of a trip.
 *
 * Rules (DIRECTION §8): only the same zone and the same valid-time key (`${date}|day|night`) are compared, never
 * periods by position; a reissue with the same stated numbers is not a change (issuance alone, or a reworded
 * sentence with the same numbers, is not reported); a period that has ended since planning is said once; a forecast
 * whose issuance is more than STALE_HOURS before `now` is said to be old. Wave-detail components shift slightly with
 * every bulletin and are left out of the feed (the departure comparison shows them). Source availability (a forecast
 * or alert feed that failed) belongs to the snapshot diff (diff.ts), which composes this; here a source that did not
 * answer simply produces no claim either way.
 *
 * Advisories and observations are read from snapshot evidence of kind "advisory" / "observation"; build them with
 * `advisoryEvidence` / `observationEvidence` below so their identity and version are the same on every fetch.
 */

export const STALE_HOURS = 12;

const hash = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
};

// ---------------------------------------------------------------- evidence builders (shared identity)

/** An NWS alert as snapshot evidence. key: zone + event (NWS reissues an advisory under new ids); version: its meaning. */
export function advisoryEvidence(a: AdvisoryIn, zone: string | null, opts: { asOf?: string | null } = {}): EvidenceItem {
  const fields = { event: a.event, headline: a.headline, onset: a.onset ?? null, ends: a.ends ?? null };
  return {
    key: `advisory:${zone ?? "?"}:${a.event.toLowerCase().replace(/\s+/g, "-")}`,
    kind: "advisory",
    source: "NWS",
    title: a.event,
    text: a.headline,
    url: a.url,
    asOf: opts.asOf ?? a.onset ?? null,
    scope: zone,
    version: hash(JSON.stringify([fields.event, fields.onset, fields.ends])),
    fields,
    verified: null,
    conflict: null,
    stale: false,
  };
}

export type BuoyIn = { id: string; name: string; km: number; url: string; at: string | null; waveFt: number | null; periodS: number | null; windKt: number | null; windDir: string | null; gustKt: number | null };

/** A buoy reading as snapshot evidence. key: the station; version: the reading (time and values). */
export function observationEvidence(b: BuoyIn): EvidenceItem {
  const fields = { station: b.id, km: Math.round(b.km), waveFt: b.waveFt, periodS: b.periodS, windKt: b.windKt, windDir: b.windDir, gustKt: b.gustKt };
  return {
    key: `observation:ndbc:${b.id}`,
    kind: "observation",
    source: "NDBC",
    title: `Buoy ${b.id}, ${b.name}`,
    text: obsWords(fields) ?? "No wave or wind value",
    url: b.url,
    asOf: b.at,
    scope: `${Math.round(b.km)} km from port`,
    version: hash(JSON.stringify([b.at, b.waveFt, b.periodS, b.windKt, b.windDir, b.gustKt])),
    fields,
    verified: null,
    conflict: null,
    stale: false,
  };
}

function obsWords(f: Record<string, string | number | boolean | null>): string | null {
  const parts = [f.waveFt != null ? `waves ${f.waveFt} ft${f.periodS != null ? ` at ${f.periodS} s` : ""}` : null, f.windKt != null ? `wind ${f.windDir ? `${f.windDir} ` : ""}${f.windKt} kt` : null].filter(Boolean);
  return parts.length ? parts.join(", ") : null;
}

// ---------------------------------------------------------------- the diff

const ok = (s: Snapshot, k: keyof Snapshot["sources"]) => s.sources?.[k]?.state === "ok" || s.sources?.[k]?.state === "absent";
const dated = (p: ForecastPeriod) => !!p.date && !p.key.startsWith("undated|");
const DIR_ARROW = " → ";

function departureDate(base: Snapshot, next: Snapshot): string | null {
  return next.assumptions?.depart?.date ?? base.assumptions?.depart?.date ?? null;
}

const where = (p: ForecastPeriod, onDep: boolean) => (onDep ? "during your selected departure period" : `for ${keyLabel(p.key)}`);

function change(c: Omit<Change, "evidenceKey" | "where" | "when"> & Partial<Pick<Change, "evidenceKey" | "where" | "when">>): Change {
  return { evidenceKey: null, where: null, when: null, ...c };
}

/** Forecast, advisory and observation changes from `base` (usually the planning snapshot) to `next`. */
export function forecastChanges(base: Snapshot, next: Snapshot, now: string): Change[] {
  const out: Change[] = [];
  const nowMs = Date.parse(now);
  const dep = departureDate(base, next);
  const a = base.forecast;
  const b = next.forecast;
  const url = b?.url ?? a?.url ?? null;
  const zone = b?.zone ?? a?.zone ?? null;

  if (a && b && a.zone && b.zone && a.zone !== b.zone) {
    out.push(change({ id: `forecast:zone:${a.zone}:${b.zone}`, priority: 2, category: "forecast", kind: "revised", title: `The forecast zone differs from the one you planned with (${a.zone} → ${b.zone}); its periods are not compared.`, detail: null, before: a.zone, after: b.zone, url, periodKey: null, where: b.zone }));
  } else if (a && b && a.periods.length && b.periods.length) {
    const nextBy = new Map(b.periods.filter(dated).map((p) => [p.key, p]));
    const ended: ForecastPeriod[] = [];
    for (const p of a.periods.filter(dated)) {
      const onDep = !!dep && p.date === dep;
      const q = nextBy.get(p.key);
      if (p.end && Date.parse(p.end) <= nowMs) {
        // over since planning (not already over when the base was taken)
        if (!base.takenAt || Date.parse(p.end) > Date.parse(base.takenAt)) ended.push(p);
        continue;
      }
      if (!q) continue;
      const w = where(p, onDep);
      const when = b.issued;
      const pr = (big: boolean) => (onDep ? 1 : big ? 2 : 3) as 1 | 2 | 3;
      if (p.windHi !== q.windHi || p.windLo !== q.windLo) {
        const stronger = q.windHi != null && (p.windHi == null || q.windHi > p.windHi);
        const weaker = p.windHi != null && (q.windHi == null || q.windHi < p.windHi);
        const what = q.windHi == null ? "no wind speed stated now" : p.windHi == null ? "a wind speed now stated" : stronger ? "stronger winds" : weaker ? "lighter winds" : "a different wind range";
        out.push(change({ id: `forecast:wind:${p.key}:${p.windLo}-${p.windHi}:${q.windLo}-${q.windHi}`, priority: pr(stronger), category: "forecast", kind: "revised", title: `Marine forecast updated since you planned: ${what} ${w}.`, detail: q.text, before: windRangeWords(p) ?? "not stated", after: windRangeWords(q) ?? "not stated", url, periodKey: p.key, when, where: zone }));
      }
      if (p.gustKt !== q.gustKt) {
        const what = q.gustKt == null ? "gusts no longer stated" : p.gustKt == null ? `gusts to ${q.gustKt} kt now stated` : q.gustKt > p.gustKt ? "stronger gusts" : "lighter gusts";
        out.push(change({ id: `forecast:gust:${p.key}:${p.gustKt}:${q.gustKt}`, priority: pr(q.gustKt != null && (p.gustKt == null || q.gustKt > p.gustKt)), category: "forecast", kind: "revised", title: `Marine forecast updated since you planned: ${what} ${w}.`, detail: q.text, before: p.gustKt != null ? `${p.gustKt} kt` : "not stated", after: q.gustKt != null ? `${q.gustKt} kt` : "not stated", url, periodKey: p.key, when, where: zone }));
      }
      if (p.seasHi !== q.seasHi || p.seasLo !== q.seasLo) {
        const higher = q.seasHi != null && (p.seasHi == null || q.seasHi > p.seasHi);
        const lower = p.seasHi != null && (q.seasHi == null || q.seasHi < p.seasHi);
        const what = q.seasHi == null ? "no seas stated now" : p.seasHi == null ? "seas now stated" : higher ? "higher seas" : lower ? "lower seas" : "a different seas range";
        out.push(change({ id: `forecast:seas:${p.key}:${p.seasLo}-${p.seasHi}:${q.seasLo}-${q.seasHi}`, priority: pr(higher), category: "forecast", kind: "revised", title: `Marine forecast updated since you planned: ${what} ${w}.`, detail: q.text, before: seasRangeWords(p) ?? "not stated", after: seasRangeWords(q) ?? "not stated", url, periodKey: p.key, when, where: zone }));
      }
      const da = directionSequence(p.text).join(DIR_ARROW);
      const db = directionSequence(q.text).join(DIR_ARROW);
      if (da !== db) {
        out.push(change({ id: `forecast:dir:${p.key}:${da}:${db}`, priority: onDep ? 2 : 3, category: "forecast", kind: "revised", title: `Marine forecast updated since you planned: a different wind direction ${w}.`, detail: q.text, before: da || "not stated", after: db || "not stated", url, periodKey: p.key, when, where: zone }));
      }
    }
    // the forecast now reaches the departure date when it did not before
    if (dep && !a.periods.some((p) => p.date === dep) && b.periods.some((p) => p.date === dep)) {
      out.push(change({ id: `forecast:covers:${dep}:${b.issued}`, priority: 2, category: "forecast", kind: "available", title: "The marine forecast now covers your selected departure date.", detail: null, before: "beyond the forecast", after: "forecast", url, periodKey: `${dep}|day`, when: b.issued, where: zone }));
    }
    const depEnded = ended.filter((p) => dep && p.date === dep);
    const otherEnded = ended.filter((p) => !(dep && p.date === dep));
    for (const p of depEnded) out.push(change({ id: `forecast:expired:${p.key}`, priority: 2, category: "forecast", kind: "expired", title: `The forecast period you planned to depart in has ended (${keyLabel(p.key)}).`, detail: null, before: `until ${dayTime(p.end!)}`, after: "ended", url, periodKey: p.key, when: p.end, where: zone }));
    if (otherEnded.length) {
      const last = otherEnded[otherEnded.length - 1];
      out.push(change({ id: `forecast:expired:${otherEnded.map((p) => p.key).join(",")}`, priority: 3, category: "forecast", kind: "expired", title: otherEnded.length === 1 ? `A forecast period you planned with has ended (${keyLabel(last.key)}).` : `${otherEnded.length} forecast periods you planned with have ended (${keyLabel(otherEnded[0].key)} to ${keyLabel(last.key)}).`, detail: otherEnded.map((p) => keyLabel(p.key)).join(", "), before: null, after: null, url, periodKey: null, when: last.end, where: zone }));
    }
  }

  // stale: the newest forecast is old relative to now
  if (b?.issued && nowMs - Date.parse(b.issued) > STALE_HOURS * 3600_000) {
    const h = Math.floor((nowMs - Date.parse(b.issued)) / 3600_000);
    out.push(change({ id: `forecast:stale:${b.issued}`, priority: 2, category: "forecast", kind: "stale", title: `The newest marine forecast on hand was issued ${h} hours ago; NWS has likely issued a newer one.`, detail: null, before: null, after: `issued ${dayTime(b.issued)}`, url, periodKey: null, when: b.issued, where: zone }));
  }

  // advisories (only when both alert reads completed)
  if (ok(base, "nwsAlerts") && ok(next, "nwsAlerts")) {
    const av = new Map(base.evidence.filter((e) => e.kind === "advisory").map((e) => [e.key, e]));
    const bv = new Map(next.evidence.filter((e) => e.kind === "advisory").map((e) => [e.key, e]));
    const span = (e: EvidenceItem) => [e.fields.onset, e.fields.ends] as [string | null, string | null];
    const onDep = (e: EvidenceItem) => {
      if (!dep) return false;
      const [on, off] = span(e);
      const d0 = Date.parse(zonedIso(dep, 0, 0, NWS_TZ));
      const d1 = Date.parse(zonedIso(dep, 23, 59, NWS_TZ));
      return (on ? Date.parse(String(on)) : -Infinity) < d1 && (off ? Date.parse(String(off)) : Infinity) > d0;
    };
    for (const [k, e] of bv) {
      const old = av.get(k);
      const tail = onDep(e) ? " during your selected departure period" : "";
      if (!old) out.push(change({ id: `advisory:new:${k}:${e.version}`, priority: onDep(e) ? 1 : 2, category: "advisory", kind: "new", title: `NWS issued a ${e.title}${tail} since you planned.`, detail: e.text, before: null, after: e.fields.ends ? `until ${dayTime(String(e.fields.ends))}` : "no end time given", url: e.url, periodKey: null, evidenceKey: k, when: e.asOf, where: e.scope }));
      else if (old.version !== e.version) out.push(change({ id: `advisory:revised:${k}:${e.version}`, priority: onDep(e) ? 1 : 2, category: "advisory", kind: "revised", title: `NWS revised the ${e.title}${tail}.`, detail: e.text, before: old.fields.ends ? `until ${dayTime(String(old.fields.ends))}` : "no end time given", after: e.fields.ends ? `until ${dayTime(String(e.fields.ends))}` : "no end time given", url: e.url, periodKey: null, evidenceKey: k, when: e.asOf, where: e.scope }));
    }
    for (const [k, e] of av) {
      if (bv.has(k)) continue;
      out.push(change({ id: `advisory:ended:${k}:${e.version}`, priority: 2, category: "advisory", kind: "removed", title: `The ${e.title} you planned with is no longer in effect.`, detail: e.text, before: e.fields.ends ? `until ${dayTime(String(e.fields.ends))}` : "in effect", after: "not in NWS's active alerts", url: e.url, periodKey: null, evidenceKey: k, when: null, where: e.scope }));
    }
  }

  // observations: one change per station with a newer reading (only when the buoy read completed)
  if (ok(next, "buoy")) {
    const av = new Map(base.evidence.filter((e) => e.kind === "observation").map((e) => [e.key, e]));
    const seen = new Set<string>();
    for (const e of next.evidence.filter((x) => x.kind === "observation")) {
      if (seen.has(e.key)) continue;
      seen.add(e.key);
      const old = av.get(e.key);
      if (old && old.version === e.version) continue;
      if (old && old.asOf && e.asOf && Date.parse(e.asOf) <= Date.parse(old.asOf)) continue;
      out.push(change({ id: `observation:${e.key}:${e.version}`, priority: 3, category: "observation", kind: old ? "revised" : "new", title: `Newer buoy reading near port: ${e.text}.`, detail: e.title, before: old ? `${old.text}${old.asOf ? ` (${old.asOf})` : ""}` : null, after: `${e.text}${e.asOf ? ` (${e.asOf})` : ""}`, url: e.url, periodKey: null, evidenceKey: e.key, when: e.asOf, where: e.scope }));
    }
  }

  const ids = new Set<string>();
  return out.filter((c) => (ids.has(c.id) ? false : (ids.add(c.id), true))).sort((x, y) => x.priority - y.priority);
}

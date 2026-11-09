import type { PortFact } from "../contracts";
import type { DataAge } from "./types";

/** Data age and staleness — system text computed from `as_of`, never from an answerer. */

const TZ = "America/Los_Angeles";

export function pacificDate(d: Date): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" }).format(d);
}

export function addDays(ymd: string, n: number): string {
  const t = Date.parse(`${ymd}T12:00:00Z`) + n * 86_400_000;
  return new Date(t).toISOString().slice(0, 10);
}

export function parseAsOf(asOf: string | null): number | null {
  if (!asOf) return null;
  const t = Date.parse(asOf.length === 10 ? `${asOf}T20:00:00Z` : asOf); // a bare date ≈ midday Pacific
  return Number.isFinite(t) ? t : null;
}

export function ageText(asOf: string | null, now: Date): string {
  const t = parseAsOf(asOf);
  if (t === null) return "age unknown";
  const h = (now.getTime() - t) / 3_600_000;
  if (h < 0) return asOf && asOf.length === 10 ? "today" : "just now";
  if (h < 1) return "under an hour old";
  if (h < 36) return `${Math.round(h)} hour${Math.round(h) === 1 ? "" : "s"} old`;
  const d = Math.round(h / 24);
  return `${d} days old`;
}

const HOURS = 3_600_000;

/** A plain note when a fact is past its useful life, else null. Thresholds per kind (R21). */
export function staleNote(f: PortFact, now: Date): string | null {
  const t = parseAsOf(f.as_of);
  const date = f.as_of?.slice(0, 10) ?? "an unknown date";
  const today = pacificDate(now);
  if (f.valid_for && f.valid_for.to < today) return `${f.source.name} of ${date} is out of date: it covered ${f.valid_for.from} to ${f.valid_for.to}.`;
  if (t === null) return `${f.source.name} has no date, so its age is unknown.`;
  const age = now.getTime() - t;
  switch (f.kind) {
    case "sea":
    case "weather":
      return age > 18 * HOURS ? `${f.source.name} was issued ${ageText(f.as_of, now).replace(" old", " ago")}; look for a newer forecast.` : null;
    case "observation":
      if (f.as_of && f.as_of.length > 10) return age > 6 * HOURS ? `${f.source.name} reading is ${ageText(f.as_of, now)}.` : null;
      return age > 4 * 24 * HOURS ? `${f.source.name} value is ${ageText(f.as_of, now)}.` : null;
    case "status":
    case "whale":
    case "notice":
      return age > 10 * 24 * HOURS ? `${f.source.name} record was last checked ${ageText(f.as_of, now).replace(" old", " ago")}; check the agency page.` : null;
    case "test":
      return age > 30 * 24 * HOURS ? `The ${f.source.name} sample of ${date} is ${ageText(f.as_of, now)}; a newer set may exist.` : null;
    case "bloom_forecast":
      return age > 2 * 24 * HOURS ? `${f.source.name} run of ${date} is ${ageText(f.as_of, now)}.` : null;
    case "data_age":
      return age > 36 * HOURS ? `Neritic's data feed has not refreshed for ${ageText(f.as_of, now).replace(" old", "")}.` : null;
    default:
      return null;
  }
}

export function isStale(f: PortFact, now: Date): boolean {
  return staleNote(f, now) !== null;
}

export function dataAge(facts: PortFact[], now: Date): DataAge | null {
  if (!facts.length) return null;
  const dated = facts.filter((f) => parseAsOf(f.as_of) !== null).sort((a, b) => parseAsOf(a.as_of)! - parseAsOf(b.as_of)!);
  const oldest = dated[0] ?? null;
  const newest = dated[dated.length - 1] ?? null;
  const stale = [...new Set(facts.map((f) => staleNote(f, now)).filter((x): x is string => !!x))];
  const label = oldest
    ? oldest === newest || (oldest.as_of!.slice(0, 10) === newest!.as_of!.slice(0, 10) && oldest.source.name === newest!.source.name)
      ? `Data: ${oldest.source.name}, ${oldest.as_of!.slice(0, 10)} (${ageText(oldest.as_of, now)}).`
      : `Oldest record here: ${oldest.source.name}, ${oldest.as_of!.slice(0, 10)} (${ageText(oldest.as_of, now)}). Newest: ${newest!.source.name} (${ageText(newest!.as_of, now)}).`
    : "These records carry no dates; their age is unknown.";
  return { label, oldest_as_of: oldest?.as_of ?? null, newest_as_of: newest?.as_of ?? null, stale };
}

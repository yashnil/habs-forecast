import { evaluateWorkspace, fromDraft, money, breakEvenCatchText } from "@/nx/coast/workspace";
import { audienceOf, tripFishery } from "@/nx/coast/fisheries";
import { knownView } from "../known";
import { crossings } from "../windows";
import type { Change, EvidenceItem, ForecastPeriod, Snapshot, SourceKey, Trip } from "../types";

/**
 * The departure briefing's content, from a trip and its snapshots (pure; the unit test reads it). Everything is the
 * trip's own record: the evidence as Neritic saved it, with when and how it was read, the reader's own limits, and
 * the economics through the engine of record. Nothing here is a verdict: periods are compared with the reader's own
 * limits, never called safe; an unstated value is "not stated", never favourable; a saved copy is never "current".
 */
export const SOURCE_WORDS: Record<SourceKey, string> = {
  season: "CDFW season status",
  whale: "CDFW whale-zone measures",
  cdphTest: "CDPH domoic acid tests",
  nwsForecast: "NWS marine forecast",
  nwsAlerts: "NWS advisories",
  buoy: "NDBC buoy",
  tides: "NOAA tide predictions",
};

export type Basis = {
  kind: "planning" | "recheck";
  snapshotId: string;
  takenAt: string;
  mode: Snapshot["mode"];
  /** for a saved copy: when the copy itself was fetched */
  savedFrom: string | null;
  /** every source read (ok or absent) */
  complete: boolean;
  unchecked: { source: SourceKey; label: string; why: string }[];
};

export type PeriodCell = { state: "within" | "over" | "unstated" | "noLimit"; text: string };
export type BriefPeriod = {
  key: string;
  name: string;
  night: boolean;
  start: string | null;
  end: string | null;
  wind: string | null;
  seas: string | null;
  windCell: PeriodCell;
  seasCell: PeriodCell;
  text: string;
};

/** Where a known value came from when it is not from the newest check: that check's read time. */
export type Held = { readAt: string; latest: boolean };

export type BriefOfficial = {
  held: Held;
  key: string;
  source: EvidenceItem["source"];
  kind: EvidenceItem["kind"];
  title: string;
  text: string;
  url: string | null;
  asOf: string | null;
  verified: boolean | null;
  conflict: string | null;
  stale: boolean;
};

export type BriefEconomics =
  | { audience: "commercial"; net: string | null; gross: string | null; breakEven: string | null; waiting: boolean }
  | { audience: "recreational"; tripCost: string | null; perPerson: string | null; waiting: boolean };

export type Briefing = {
  title: string;
  fishery: string;
  activity: Trip["activity"];
  depart: Trip["depart"];
  area: string | null;
  limits: Trip["limits"];
  state: Trip["state"];
  basis: Basis | null;
  forecast: { zone: string | null; issued: string | null; url: string | null; held: Held } | null;
  /** the periods that touch the departure day; empty with `periodsWhy` when none can be shown */
  periods: BriefPeriod[];
  periodsWhy: string | null;
  official: BriefOfficial[];
  changes: Change[];
  economics: BriefEconomics | null;
  verify: { label: string; url: string }[];
};

const OFFICIAL_KINDS = new Set<EvidenceItem["kind"]>(["season", "whale", "cdphTest", "advisory"]);

export function basisOf(s: Snapshot): Basis {
  const unchecked = (Object.keys(s.sources) as SourceKey[])
    .filter((k) => s.sources[k].state === "failed" || s.sources[k].state === "notChecked")
    .map((k) => ({ source: k, label: SOURCE_WORDS[k] ?? k, why: s.sources[k].detail ?? (s.sources[k].state === "failed" ? "did not answer" : "not checked") }));
  return { kind: s.kind, snapshotId: s.id, takenAt: s.takenAt, mode: s.mode, savedFrom: s.savedFrom, complete: unchecked.length === 0 && s.mode === "live", unchecked };
}

const range = (lo: number | null, hi: number | null, unit: string) => (lo == null && hi == null ? null : lo != null && hi != null && lo !== hi ? `${lo}–${hi} ${unit}` : `${hi ?? lo} ${unit}`);

/** One reported value against the reader's own limit: over it, within it, not stated, or no limit set. */
export function cell(value: number | null, limit: number | null, unit: string, what: string): PeriodCell {
  if (value == null) return { state: "unstated", text: `${what} not stated` };
  if (limit == null) return { state: "noLimit", text: `${what} stated up to ${value} ${unit}; no limit set` };
  return value > limit ? { state: "over", text: `${what} stated up to ${value} ${unit}; your limit ${limit} ${unit}` } : { state: "within", text: `${what} stated up to ${value} ${unit}; under your limit of ${limit} ${unit}` };
}

/** The periods that start on the departure day, or the night before it; with no date set, the first three. */
export function departurePeriods(f: Snapshot["forecast"], depart: Trip["depart"]): ForecastPeriod[] {
  if (!f?.periods.length) return [];
  if (!depart) return f.periods.slice(0, 3);
  const prev = new Date(`${depart.date}T12:00:00Z`);
  prev.setUTCDate(prev.getUTCDate() - 1);
  const before = prev.toISOString().slice(0, 10);
  return f.periods.filter((p) => p.date === depart.date || (p.date === before && p.night));
}

function economicsOf(trip: Trip): BriefEconomics | null {
  if (!trip.economics) return null;
  const f = tripFishery(trip.fishery);
  const r = evaluateWorkspace(fromDraft(trip.economics, trip.fishery), f);
  const v = (k: keyof typeof r.out) => (r.out[k].status === "value" ? (r.out[k] as { value: number }).value : null);
  if (audienceOf(f) === "recreational") {
    const cost = v("operating");
    const per = v("perPerson");
    return { audience: "recreational", tripCost: cost != null ? money(cost) : null, perPerson: per != null ? money(per) : null, waiting: cost == null };
  }
  const net = v("net");
  const gross = v("gross");
  const be = v("breakEvenCatch");
  return { audience: "commercial", net: net != null ? money(net) : null, gross: gross != null ? money(gross) : null, breakEven: be != null ? breakEvenCatchText(be, f?.unit ?? "lb") : null, waiting: net == null };
}

/**
 * `snaps.all`: every snapshot of the trip (planning, rechecks, superseded). What the briefing shows of each source is
 * the newest copy the device holds (known.ts knownView): a recheck whose read failed, or one made offline, never
 * replaces evidence the device already had; such evidence keeps the time it was read (`held`).
 */
export function briefing(trip: Trip, snaps: { planning: Snapshot | null; latest: Snapshot | null; all?: Snapshot[] }, o: { portName?: string | null; changes?: Change[] } = {}): Briefing {
  const shown = snaps.latest ?? snaps.planning;
  const all = snaps.all ?? [snaps.planning, snaps.latest].filter((x): x is Snapshot => !!x);
  const known = knownView(all);
  const fishery = tripFishery(trip.fishery);
  const f = known.forecast;
  const periods = departurePeriods(f?.value ?? null, trip.depart);
  const sea = shown?.sources.nwsForecast;
  const periodsWhy = periods.length
    ? null
    : !shown
      ? "No snapshot saved for this trip yet."
      : f?.value.periods.length
        ? "The saved forecast does not reach your departure day."
        : sea && (sea.state === "failed" || sea.state === "notChecked")
          ? "The NWS marine forecast could not be read, and no earlier copy is on this device."
          : "No marine forecast periods were saved.";
  const official: BriefOfficial[] = known.evidence
    .filter((k) => OFFICIAL_KINDS.has(k.value.kind))
    .map(({ value: e, readAt, latest }) => ({ held: { readAt, latest }, key: e.key, source: e.source, kind: e.kind, title: e.title, text: e.text, url: e.url, asOf: e.asOf, verified: e.verified, conflict: e.conflict, stale: e.stale }));
  const verify = official.filter((o) => o.url && (o.source === "CDFW" || o.source === "CDPH")).map((o) => ({ label: `${o.source}: ${o.title}`, url: o.url! }));
  return {
    title: trip.name,
    fishery: fishery?.label ?? trip.fishery,
    activity: trip.activity,
    depart: trip.depart,
    area: trip.area,
    limits: trip.limits,
    state: trip.state,
    basis: shown ? basisOf(shown) : null,
    forecast: f ? { zone: f.value.zone, issued: f.value.issued, url: f.value.url, held: { readAt: f.readAt, latest: f.latest } } : null,
    periods: periods.map((p) => {
      // the reader's limits against the stated numbers, as the departure windows read them (windows.ts crossings)
      const x = crossings(p, trip.limits);
      const windX = x.find((c) => c.limit === "wind") ?? x.find((c) => c.limit === "gust");
      const seasX = x.find((c) => c.limit === "seas");
      const windCell: PeriodCell = windX ? { state: "over", text: windX.words } : cell(p.windHi ?? p.windLo, trip.limits.windKt, "kt", "Wind");
      const seasCell: PeriodCell = seasX ? { state: "over", text: seasX.words } : cell(p.seasHi ?? p.seasLo, trip.limits.seasFt, "ft", "Seas");
      return {
        key: p.key,
        name: p.name,
        night: p.night,
        start: p.start,
        end: p.end,
        wind: p.windLo == null && p.windHi == null ? null : `${p.windDir ? `${p.windDir} ` : ""}${range(p.windLo, p.windHi, "kt")}${p.gustKt != null ? `, gusts ${p.gustKt} kt` : ""}`,
        seas: range(p.seasLo, p.seasHi, "ft"),
        windCell,
        seasCell,
        text: p.text,
      };
    }),
    periodsWhy,
    official,
    changes: o.changes ?? [],
    economics: economicsOf(trip),
    verify: [...new Map(verify.map((v) => [v.url + v.label, v])).values()],
  };
}

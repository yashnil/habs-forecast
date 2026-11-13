import type { CoastError, CoastGates, CoastSea } from "@/nx/coast/brief";
import { day, dayTime } from "@/nx/ds/format";
import { tripFishery } from "@/nx/coast/fisheries";
import { forecastFromSea } from "./forecast";
import { advisoryEvidence, observationEvidence } from "./forecastDiff";
import { SNAPSHOT_SCHEMA, type EvidenceItem, type ForecastSnap, type Snapshot, type SourceKey, type SourceRead, type Trip } from "./types";

/**
 * What Neritic showed, captured (v15 §7). Pure: the answers of the two My Coast endpoints (`/coast/data?…&fishery=`
 * for the gates, `…&part=sea` for the sea) and the trip become a Snapshot that later checks are compared against.
 *
 * Identity and version: every piece of evidence gets a `key` that names *what* it is (the same CDFW record across
 * fetches) and a `version` hashed from the fields that carry meaning only (level, the record's words with whitespace
 * and case normalized, conflict, verified, measures, the next window, a test's date and values). Fetch and "checked"
 * times never enter the version, so a refetch of an unchanged record compares equal, and a reformatted sentence is
 * not a change.
 *
 * Honesty: a source that did not answer is "failed" (never "absent"); "absent" is an answer that nothing of that kind
 * applies here. Official records stay unverified until a person reviews them. Nothing here is a verdict.
 */

export type Answer<T> = T | CoastError | "failed" | null;

export type SnapshotInput = {
  trip: Trip;
  /** the gates answer; "failed": the request did not answer; null: not asked */
  gates: Answer<CoastGates>;
  sea: Answer<CoastSea>;
  /** the zone periods; null: derived here from the sea answer with forecastFromSea */
  forecast: ForecastSnap | null;
  /** device clock (ISO) */
  now: string;
  mode: Snapshot["mode"];
  savedFrom: string | null;
  kind?: Snapshot["kind"];
  id?: string;
};

export const SOURCE_KEYS: SourceKey[] = ["season", "whale", "cdphTest", "nwsForecast", "nwsAlerts", "buoy", "tides"];

/** Who publishes each source, as the reader knows them. */
export const SOURCE_LABEL: Record<SourceKey, string> = {
  season: "CDFW season record",
  whale: "CDFW whale measures",
  cdphTest: "CDPH domoic acid test",
  nwsForecast: "NWS zone forecast",
  nwsAlerts: "NWS advisories",
  buoy: "NDBC buoy",
  tides: "CO-OPS tide predictions",
};

/** The source an evidence item is read from. */
export const SOURCE_OF: Record<EvidenceItem["kind"], SourceKey> = { season: "season", whale: "whale", cdphTest: "cdphTest", advisory: "nwsAlerts", observation: "buoy" };

// ---------------------------------------------------------------- identity

/** Text as it carries meaning: whitespace collapsed, case folded, trailing stop dropped. */
export const norm = (s: string | null | undefined): string => (s ?? "").replace(/\s+/g, " ").trim().replace(/[.]+$/, "").toLowerCase();

/** A short stable hash (FNV-1a, two seeds → 16 hex digits). Not cryptographic; identity only. */
export function stableHash(s: string): string {
  const h = (seed: number) => {
    let x = seed >>> 0;
    for (let i = 0; i < s.length; i++) {
      x ^= s.charCodeAt(i);
      x = Math.imul(x, 0x01000193) >>> 0;
    }
    return x.toString(16).padStart(8, "0");
  };
  return h(0x811c9dc5) + h(0x01000193 ^ 0x5bd1e995);
}

type Meaning = Record<string, string | number | boolean | null>;

/** Strings normalized, keys sorted: the same meaning always hashes the same. */
export function versionOf(m: Meaning): string {
  const canon = Object.keys(m)
    .sort()
    .map((k) => `${k}=${typeof m[k] === "string" ? norm(m[k] as string) : JSON.stringify(m[k])}`)
    .join("\u0001");
  return stableHash(canon);
}

const slug = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

/** The fields of an item that make up its version (the rest of `fields` is context, e.g. a measurement time). */
export const VERSION_FIELDS: Record<EvidenceItem["kind"], string[]> = {
  season: ["level", "text", "conflict", "differ", "verified", "seasonal"],
  whale: ["measures", "next", "verified"],
  cdphTest: ["date", "values", "tissue", "action"],
  advisory: ["event", "headline", "ends"],
  observation: ["at", "windKt", "windDir", "gustKt", "waveFt", "periodS"],
};

function item(e: Omit<EvidenceItem, "version">): EvidenceItem {
  const m: Meaning = {};
  for (const k of VERSION_FIELDS[e.kind]) m[k] = e.fields[k] ?? null;
  return { ...e, version: versionOf(m) };
}

// ---------------------------------------------------------------- sources

const read = (state: SourceRead["state"], at: string | null, detail: string | null): SourceRead => ({ state, at, detail });

/** NWS/NDBC/CO-OPS "none" reasons: the service answered that nothing of that kind is near (conditionsServer.ts). */
const NOTHING_NEAR = /^No NDBC station reports within|^No tide prediction station within|not inside an NWS coastal marine zone/i;

const reasonOf = (a: Answer<unknown>, what: string): string =>
  a === "failed" ? `${what} did not answer.` : a && typeof a === "object" && "ok" in a && a.ok === false ? `${what} did not answer${(a as CoastError).error ? ` (${(a as CoastError).error})` : ""}.` : `${what} was not asked.`;

// ---------------------------------------------------------------- the snapshot

const hoursBetween = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 3_600_000;

export function takeSnapshot(input: SnapshotInput): Snapshot {
  const { trip, gates, sea, now, mode, savedFrom } = input;
  const kind = input.kind ?? "planning";
  const sources = {} as Record<SourceKey, SourceRead>;
  const evidence: EvidenceItem[] = [];
  const limitations: string[] = [];
  const g = gates && typeof gates === "object" && gates.ok ? gates : null;
  const s = sea && typeof sea === "object" && sea.ok ? sea : null;
  const fishery = tripFishery(trip.fishery);
  const fisheryWords = fishery?.label ?? trip.fishery;

  // ---- the gates: season, whale measures, the latest CDPH test
  if (!g) {
    const why = gates === null ? "Not checked: no connection, or not asked." : reasonOf(gates, "Neritic's copy of the CDFW and CDPH records");
    const st = gates === null ? "notChecked" : "failed";
    for (const k of ["season", "whale", "cdphTest"] as const) sources[k] = read(st, null, why);
  } else {
    const at = g.now;
    if (g.statusError && !g.season) sources.season = read("failed", at, "The CDFW season list could not be read.");
    else if (!g.season) sources.season = read("absent", at, `No season record for ${fisheryWords} at this port.`);
    else sources.season = read("ok", g.season.checked ?? at, g.statusError ? "Read, though part of the season list did not load." : null);

    if (g.whale) sources.whale = read("ok", g.whale.assessed ?? at, null);
    else if (g.statusError && fishery?.ramp) sources.whale = read("failed", at, "CDFW's whale measures could not be read.");
    else sources.whale = read("absent", at, fishery?.ramp ? "No whale measures on record for this zone." : `CDFW's whale measures do not apply to ${fisheryWords}.`);

    if (g.testsError) sources.cdphTest = read("failed", at, "CDPH's test results could not be read.");
    else if (!g.test) sources.cdphTest = read("absent", at, `No CDPH test for ${fisheryWords} informs this port.`);
    else sources.cdphTest = read("ok", g.test.date, null);

    if (g.season) {
      const x = g.season;
      evidence.push(
        item({
          key: `season:${trip.fishery}:${trip.placeKey}`,
          kind: "season",
          source: /cdph/i.test(x.source) ? "CDPH" : "CDFW",
          title: `${x.source} record, ${fisheryWords}`,
          text: x.text,
          url: x.url,
          asOf: x.checked,
          scope: fisheryWords,
          fields: { level: x.level, text: x.text, conflict: x.conflict, differ: x.differ, verified: x.verified, seasonal: x.seasonal },
          verified: x.verified,
          conflict: x.conflict ? (x.differ ?? "Sources differ; awaiting a person's review.") : null,
          stale: x.stale,
        }),
      );
    }
    if (g.whale) {
      const w = g.whale;
      evidence.push(
        item({
          key: `whale:${w.zone}:${trip.fishery}`,
          kind: "whale",
          source: "CDFW",
          title: `Whale measures, Zone ${w.zone}`,
          text: w.measures,
          url: w.url,
          asOf: w.assessed,
          scope: `Zone ${w.zone}, ${w.area}`,
          fields: { measures: w.measures, next: w.next, verified: w.verified, zone: w.zone },
          verified: w.verified,
          conflict: null,
          stale: false,
        }),
      );
    }
    if (g.test) {
      const t = g.test;
      evidence.push(
        item({
          key: `cdph:${slug(t.site)}:${slug(t.species)}`,
          kind: "cdphTest",
          source: "CDPH",
          title: `CDPH test, ${t.species}, ${t.site}`,
          text: `Sampled ${day(t.date, { ref: now })}: ${t.raw.join(", ")} ppm${t.action != null ? ` (action level ${t.action} ppm)` : ""}.`,
          url: t.url,
          asOf: t.date,
          scope: t.site,
          fields: { date: t.date, values: t.raw.join(","), tissue: t.tissue, action: t.action, max: t.max, over: t.over, ageDays: t.ageDays, lat: t.lat, lon: t.lon },
          verified: null,
          conflict: null,
          stale: t.stale,
        }),
      );
      limitations.push(`CDPH's latest test is from ${day(t.date, { ref: now })}; it says nothing about today.`);
    }
    if (g.season?.conflict) limitations.push("Two of Neritic's sources disagree about this fishery; neither is chosen until a person reviews them.");
  }

  // ---- the sea: NWS zone forecast and advisories, the nearest buoy, CO-OPS tides
  let forecast: ForecastSnap | null = null;
  let tides: Snapshot["tides"] = null;
  if (!s) {
    const why = sea === null ? "Not checked: no connection, or not asked." : reasonOf(sea, "The marine conditions request");
    const st = sea === null ? "notChecked" : "failed";
    for (const k of ["nwsForecast", "nwsAlerts", "buoy", "tides"] as const) sources[k] = read(st, null, why);
  } else {
    const v = s.sea;
    if (v.lanes.length) sources.nwsForecast = read("ok", v.issued, null);
    else if (v.marineReason && NOTHING_NEAR.test(v.marineReason)) sources.nwsForecast = read("absent", s.now, v.marineReason);
    else sources.nwsForecast = read("failed", s.now, v.marineReason ?? "NWS returned no forecast periods.");

    // SeaView carries the advisories but not whether their read failed: an empty list is reported as none listed,
    // and the limitation says so (request to the lead: expose NWS's alerts status in CoastSea)
    if (sources.nwsForecast.state === "failed") sources.nwsAlerts = read("failed", s.now, "Read with the zone forecast, which did not answer.");
    else if (v.alertsReason) sources.nwsAlerts = read("failed", s.now, v.alertsReason);
    else sources.nwsAlerts = read(v.alerts.length ? "ok" : "absent", s.now, v.alerts.length ? null : "None listed in NWS's answer.");

    if (v.buoy) sources.buoy = read("ok", v.buoy.at, null);
    else if (v.buoyReason && NOTHING_NEAR.test(v.buoyReason)) sources.buoy = read("absent", s.now, v.buoyReason);
    else sources.buoy = read("failed", s.now, v.buoyReason ?? "No buoy reading in the answer.");

    if (v.tides.length) sources.tides = read("ok", s.now, null);
    else if (v.tideReason && NOTHING_NEAR.test(v.tideReason)) sources.tides = read("absent", s.now, v.tideReason);
    else sources.tides = read("failed", s.now, v.tideReason ?? "No tide predictions in the answer.");

    forecast = input.forecast ?? (v.lanes.length ? forecastFromSea(s) : null);
    tides = v.tides.length || v.tideStation ? { station: v.tideStation?.name ?? null, url: v.tideStation?.url ?? null, events: v.tides.map((t) => ({ at: t.at, ft: t.ft, type: t.type })) } : null;

    // advisories and readings use the forecast diff's builders, so their identity is the one forecastDiff.ts compares
    for (const a of v.alerts) evidence.push(advisoryEvidence({ event: a.event, headline: a.headline, url: a.url || v.marineUrl || "", ends: a.ends }, v.zone, { asOf: v.issued }));
    // positions ride in `fields` for the trip chart; they are not part of any version
    const obs = (o: { id: string; name: string; km: number; url: string; at: string | null; windKt: number | null; windDir: string | null; gustKt: number | null; waveFt?: number | null; periodS?: number | null }, at: { lat: number | null; lon: number | null } | null) => {
      const e = observationEvidence({ ...o, waveFt: o.waveFt ?? null, periodS: o.periodS ?? null });
      evidence.push({ ...e, fields: { ...e.fields, id: o.id, lat: at?.lat ?? null, lon: at?.lon ?? null }, stale: o.at ? hoursBetween(o.at, s.now) > 3 : true });
    };
    if (v.buoy) obs(v.buoy, { lat: v.buoy.lat, lon: v.buoy.lon });
    if (v.windBuoy) obs(v.windBuoy, s.windAt ?? null);

    if (forecast) limitations.push(`NWS zone forecast covers the whole zone${forecast.zone ? ` (${forecast.zone})` : ""}, not your route.`);
    if (v.buoy) limitations.push(`The buoy reading is one point ${v.buoy.km.toFixed(0)} km from the port at the time it was measured; it is not a forecast.`);
    if (!v.alerts.length && sources.nwsAlerts.state === "absent") limitations.push("No NWS advisory was listed for this zone when Neritic read it. NWS's zone page has the full list.");
  }

  if (evidence.some((e) => (e.kind === "season" || e.kind === "whale") && e.verified !== true)) limitations.push("Official notices are not verified by a person yet.");
  for (const k of SOURCE_KEYS) if (sources[k].state === "failed") limitations.push(`${SOURCE_LABEL[k]} could not be read; this snapshot says nothing about it.`);
  if (mode === "saved") limitations.push(`Rebuilt from a copy saved on this device${savedFrom ? ` on ${dayTime(savedFrom)}` : ""}. Nothing was fetched for it.`);
  if (trip.limits.windKt != null || trip.limits.seasFt != null) limitations.push("Your wind and sea limits are your own thresholds; no agency sets or checks them.");

  const serverNow = g?.now ?? s?.now ?? null;
  return {
    schema: SNAPSHOT_SCHEMA,
    id: input.id ?? `${kind}-${trip.id}-${Date.parse(now).toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
    tripId: trip.id,
    kind,
    takenAt: now,
    serverNow: mode === "saved" ? null : serverNow,
    mode,
    savedFrom: mode === "saved" ? savedFrom : null,
    sources,
    evidence,
    forecast,
    tides,
    assumptions: { limits: { ...trip.limits }, depart: trip.depart ? { ...trip.depart } : null, economics: trip.economics },
    limitations: [...new Set(limitations)],
  };
}

/** A completed read: it answered, with something or with nothing that applies. */
export const readOk = (r: SourceRead | undefined): boolean => !!r && (r.state === "ok" || r.state === "absent");

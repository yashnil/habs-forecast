import type { Association } from "@/lib/applicability";
import { AUDIENCE } from "@/lib/applicability";
import type { Audience } from "@/lib/coast";
import { compass, marineHeadline, stationName, type Conditions, type MarinePeriod, type Tide } from "@/lib/conditions";
import type { Freshness } from "@/lib/freshness";
import { ageWords, fmtMeasured, fmtPct, EVIDENCE_RADIUS_KM, NEIGHBOURHOOD_KM, RECENT_DAYS, type Neighbourhood, type StationEvidence } from "@/lib/hab";
import type { StationLite } from "@/lib/data";
import type { Verification } from "@/lib/official";
import { runAgeWords } from "@/lib/charmDays";
import { recordNamesSpecies, speciesPhrase, type Species } from "@/content/species";
import { rankWords, type RecordContext } from "@/lib/history";
import { READINGS_FOR, SPECIES_EVIDENCE, TOXIN_WORDS, type Toxin } from "@/content/speciesEvidence";

/**
 * The harbor evidence graph: everything Neritic knows about one place, as typed nodes that each
 * carry how they are known (measured, retrieved from a satellite, published as a forecast,
 * reconstructed, or Neritic's own experiment), where and when they are true, how far they are
 * from the place and whether that distance lets them describe it, their freshness and their
 * caveats; plus the gaps, said as gaps; plus deterministic claims that cite the nodes they rest on.
 *
 * It is the one source of every briefing sentence: My Coast renders the claims as its "right now"
 * lines and its answer line, the evidence section lists the nodes, and GET /api/evidence returns
 * the same graph, so a sentence on the page, an API answer and, later, an assistant cannot
 * disagree about the same observation. It is pure. Nothing here interpolates, scores risk or
 * speaks of safety.
 */

/** record_statistic: a measurement compared with the same station's own earlier samples (lib/history) */
export type EvidenceClass = "measurement" | "record_statistic" | "satellite_retrieval" | "published_forecast" | "reconstruction" | "experimental";

/** How well a reading at `km` from the place can stand for the place, by the variable's own scale. */
export type Representative = "at" | "near" | "context" | "far";

export type EvidenceNode = {
  id: string;
  class: EvidenceClass;
  variable: string;
  units: string;
  value: number | null;
  /** the value as the briefing says it ("4,900 cells/L") */
  text: string;
  source: { id: string; name: string; url?: string | null; product?: string; version?: string };
  time: { observed?: string | null; valid?: string | null; issued?: string | null; ageDays: number | null };
  support: { kind: "point" | "cell" | "cells" | "pixel" | "pixels" | "zone"; nativeM?: number | null; radiusKm?: number; n?: number };
  where: { lat: number | null; lon: number | null; name: string | null; km: number; bearing: string | null; representative: Representative; setting?: Setting | null };
  freshness: { state: Freshness["state"] | "unknown"; note?: string };
  quality: { flags: string[] };
  lineage: string[];
  caveats: string[];
};

export type Gap = { id: string; variable: string; reason: string; nearest?: { name: string; km: number; date?: string | null } | null };

/** What the map shows when a claim or node is chosen. */
export type EvidenceFocus = "toxin" | "cells" | "chl" | "satellite" | "model" | "official" | "conditions" | null;

export type ClaimKind = "official" | "measured" | "model" | "satellite" | "conditions" | "gap";

export type Claim = {
  id: string;
  kind: ClaimKind;
  /** alert: the reader must act on it (a fishery closed, a notice disputed); note: worth knowing; gap: not known here */
  tone: "alert" | "note" | "gap";
  /** the line as the page shows it */
  headline: string;
  detail?: string;
  /** the measurement against the station's own record (lib/history): one sentence, its own node */
  history?: string;
  /** the agency's own words, quoted */
  quote?: string;
  /** a second statement from the agency about the same fishery, quoted with its own page */
  also?: { label: string; quote: string; url: string };
  source?: { label: string; url: string };
  focus: EvidenceFocus;
  /** the evidence is older than it should be for the claim it makes */
  aged?: boolean;
  /** the claim as one plain statement (the API and a future assistant) */
  text: string;
  /** the nodes the claim rests on */
  supports: string[];
};

export type EvidenceGraph = {
  place: { name: string; lat: number; lon: number };
  generatedAt: string;
  window: { from: string; to: string };
  nodes: EvidenceNode[];
  gaps: Gap[];
  /** most consequential first: official words, then the measurement, the model, the satellite, the sea */
  claims: Claim[];
  /** what the graph does not try to say */
  limits: string[];
};

// ---------------------------------------------------------------- representativeness, by variable

/**
 * Distances within which a reading can stand for a place, by variable. They are not one threshold:
 * - Pseudo-nitzschia counts and particulate domoic acid are sampled at piers weekly; bloom patches
 *   along this coast are of order 10 km and the nowcast/forecast skill study (Anderson et al. 2016)
 *   matched model cells to piers within about 10–15 km. A pier farther than EVIDENCE_RADIUS_KM
 *   (60 km) is regional context at best.
 * - Satellite chlorophyll pixels (300–750 m) describe the pixel; a median within 5 km is "near"
 *   (lib/chlNear's place rule), with shore-adjacent pixels set aside.
 * - C-HARM cells are 3 km; the waters' own cell is "at", offshore cells within 15 km are context
 *   only (the published port rule), and a nearest cell beyond that is named with its distance.
 * - Wind and seas are synoptic (a buoy within 30 km describes the same weather, though not the
 *   same exposure); water temperature from a harbor tide station is local (2 km) and, within
 *   6 km, the same harbour or bay; beyond that it is another water body.
 */
export const REPRESENTATIVE_KM: Record<string, { at: number; near: number; context: number }> = {
  pn_cells: { at: 2, near: 15, context: EVIDENCE_RADIUS_KM },
  pda: { at: 2, near: 15, context: EVIDENCE_RADIUS_KM },
  chl_satellite: { at: 0.5, near: 5, context: 15 },
  charm_cell: { at: 1.6, near: 1.6, context: NEIGHBOURHOOD_KM },
  wind: { at: 5, near: 30, context: 60 },
  waves: { at: 5, near: 30, context: 60 },
  water_temp: { at: 2, near: 6, context: 20 },
  tide: { at: 5, near: 40, context: 80 },
};

export function representative(variable: keyof typeof REPRESENTATIVE_KM, km: number): Representative {
  const r = REPRESENTATIVE_KM[variable];
  return km <= r.at ? "at" : km <= r.near ? "near" : km <= r.context ? "context" : "far";
}

/**
 * The setting of a shore station, from its name: an open pier or wharf samples exposed coastal
 * water; a station inside a harbor or lagoon samples sheltered water that can differ from the
 * coast outside. A heuristic, said as one; it never changes a value.
 */
export type Setting = "exposed-pier" | "sheltered-harbor" | "offshore" | "unclassified";
export function stationSetting(name: string): Setting {
  const n = name.toLowerCase();
  if (/\b(harbor|harbour|lagoon|slough|marina|back bay|inner)\b/.test(n)) return "sheltered-harbor";
  if (/\b(pier|wharf|jetty|point|beach|cove)\b/.test(n)) return "exposed-pier";
  if (/\b(buoy|nm|offshore|canyon)\b/.test(n)) return "offshore";
  return "unclassified";
}

// ---------------------------------------------------------------- inputs

export type OutlookDayLite = { lead: number; valid: string | null; pda: Neighbourhood | null; pn: Neighbourhood | null };

export type SatLite =
  | { value: number; date: string | null; km: number; sensor: string; layerId: string; basis: { n: number; min: number; max: number; radiusKm: number } | null; aside: { value: number; reason: "shore" | "spike" } | null }
  | { value: null; sensor: string; windowDays: number; flagged: boolean; radiusKm: number }
  | null;

export type Statement = { id: string; agency: string; statement: string; topic: string; source: { url: string } };

export type EvidenceInputs = {
  place: { name: string; lat: number; lon: number };
  now: Date;
  /** the trip day the model and the forecast are read for */
  tripDate: string | null;
  /** today (Pacific), to pick the NWS period */
  today?: string | null;
  measured: { cells: StationEvidence | null; toxin: StationEvidence | null; nearest: { station: StationLite; km: number; date: string; ageDays: number } | null } | null;
  measuredError?: string | null;
  model: { days: OutlookDayLite[]; issued: string | null; fresh: Freshness | null; version?: string } | null | "error" | "loading";
  /** the trip day lies beyond the run's days */
  modelBeyond?: boolean;
  sat: SatLite | "error";
  conditions: Conditions | null;
  official: { assocs: Association[]; verification: Verification | null; audience: Audience; species: Species; statements?: Statement[] } | null;
};

const km = (k: number) => (k < 10 ? k.toFixed(1) : String(Math.round(k)));
const DAY = 86400000;
const CLOSED = /\b(closed|closure|delay(ed)?|not open|prohibit|do not (eat|consume|take))\b/i;
const RESTRICTIVE = new Set(["fishery_closure", "take_restriction", "consumption_advisory", "quarantine", "special_advisory"]);
const fits = (a: Association, audience: Audience) => audience === "all" || AUDIENCE[a.record.fishery].groups.includes(audience);
/** "Dungeness crab, commercial — marine toxin closures" → "marine toxin closures" */
export const topicWords = (topic: string) => (topic.includes("—") ? topic.split("—").pop()!.trim() : topic.charAt(0).toLowerCase() + topic.slice(1));
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** "Oct 15" for a YYYY-MM-DD date. */
export const shortDate = (d: string) => new Date(`${d.slice(0, 10)}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
/** "September 30", or "August 27, 2025" when the date is not from the last 300 days of `now`. */
export function longDate(d: string, now?: Date): string {
  const t = new Date(`${d.slice(0, 10)}T12:00:00Z`);
  const old = now ? now.getTime() - t.getTime() > 300 * DAY : false;
  return t.toLocaleDateString("en-US", { month: "long", day: "numeric", ...(old ? { year: "numeric" } : {}), timeZone: "UTC" });
}

/** The trip day's tides, low tides first for shellfish harvest. */
export function tidesOn(tides: Tide[], date: string | null): Tide[] {
  return date ? tides.filter((t) => t.at.startsWith(date)) : [];
}

/** The NWS forecast period for the trip day: "Tonight"/"Today" for today, else the day's name ("Thu"), never a night period first. */
export function periodFor(periods: MarinePeriod[], date: string | null, today: string | null): MarinePeriod | null {
  if (!periods.length) return null;
  if (!date || !today || date === today) return periods[0];
  const day = new Date(`${date}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" }).toLowerCase();
  return periods.find((p) => p.name.toLowerCase() === day || p.name.toLowerCase() === day.slice(0, 3)) ?? null;
}

/** "11:02 AM" from a CO-OPS local time "2026-10-15 11:02". */
const clock = (t: string) => {
  const [, hm] = t.split(" ");
  const [h, m] = hm.split(":").map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
};

// ---------------------------------------------------------------- the graph

export function harborEvidence(x: EvidenceInputs): EvidenceGraph {
  const nodes: EvidenceNode[] = [];
  const gaps: Gap[] = [];
  const claims: Claim[] = [];
  const P = x.place;
  const today = x.today ?? x.now.toISOString().slice(0, 10);

  // ---- official: the agency's own words lead (season status, then a disputed notice, then the notices here)
  if (x.official) {
    const o = x.official;
    const forWhat = o.species.id === "any" ? (o.audience === "all" ? "" : ` for ${{ commercial: "commercial fishing", recreational: "recreational fishing", shellfish: "shellfish harvest", all: "" }[o.audience]}`) : ` for ${speciesPhrase(o.species)}`;
    const named = (a: Association) => recordNamesSpecies(a.record.species, o.species);
    const disputed = o.assocs.filter((a) => a.dispute && named(a));
    const applying = o.assocs.filter((a) => !a.dispute && a.tier === "geographic" && fits(a, o.audience) && named(a)).sort((a, b) => Number(RESTRICTIVE.has(b.record.action)) - Number(RESTRICTIVE.has(a.record.action)));
    const unresolved = o.assocs.filter((a) => !a.dispute && a.tier === "unresolved" && fits(a, o.audience) && named(a));
    const unverified = !o.verification || (o.verification.state !== "verified" && o.verification.state !== "unavailable");
    for (const a of [...disputed, ...applying]) {
      const url = a.record.sources.find((s) => s.url.startsWith("https://"))?.url ?? null;
      nodes.push({
        id: `official:${a.record.id}`,
        class: "measurement",
        variable: "official notice",
        units: "",
        value: null,
        text: a.record.title,
        source: { id: "official", name: a.record.agency, url, product: a.record.id },
        time: { issued: a.record.effective_date ?? null, ageDays: null },
        support: { kind: "zone" },
        where: { lat: null, lon: null, name: a.record.area.description ?? null, km: 0, bearing: null, representative: "at" },
        freshness: { state: a.dispute ? "stale" : "current", ...(a.dispute ? { note: `${a.dispute.agency} release ${a.dispute.release} appears to lift it; awaiting review` } : {}) },
        quality: { flags: unverified ? ["Neritic's transcription, not verified by a person"] : [] },
        lineage: ["agency page", "transcribed by Neritic's pipeline", unverified ? "awaiting human review" : "reviewed"],
        caveats: ["The agency's wording controls. A notice missing here does not mean open or safe."],
      });
    }
    const statements = o.statements ?? [];
    const season = statements.find((s) => /season status/i.test(s.topic));
    const other = statements.find((s) => s !== season) ?? null;
    // 1. a fishery's own status (quoted): "closed" or "delayed" says more than any notice count
    if (season) {
      nodes.push(statementNode(season));
      if (other) nodes.push(statementNode(other));
      claims.push({
        id: "season",
        kind: "official",
        tone: CLOSED.test(season.statement) ? "alert" : "note",
        headline: `${season.agency} season status for ${speciesPhrase(o.species)}`,
        quote: season.statement,
        also: other ? { label: `${other.agency} on ${topicWords(other.topic)}`, quote: other.statement, url: other.source.url } : undefined,
        source: { label: `${season.agency} page`, url: season.source.url },
        focus: null,
        text: `${season.agency} season status for ${speciesPhrase(o.species)}: “${season.statement}”`,
        supports: [`statement:${season.id}`, ...(other ? [`statement:${other.id}`] : [])],
      });
    } else if (other) {
      nodes.push(statementNode(other));
      claims.push({
        id: `statement-${other.id}`,
        kind: "official",
        tone: CLOSED.test(other.statement) && !/\bno (closures?|delay)/i.test(other.statement) ? "alert" : "note",
        headline: `${other.agency} on ${topicWords(other.topic)}`,
        quote: other.statement,
        source: { label: `${other.agency} page`, url: other.source.url },
        focus: null,
        text: `${other.agency} on ${topicWords(other.topic)}: “${other.statement}”`,
        supports: [`statement:${other.id}`],
      });
    }
    // 2. a notice an agency release appears to lift: never in effect, never lifted, until a person reviews it
    for (const a of disputed.slice(0, 1)) {
      const src = a.record.sources.find((s) => s.url.startsWith("https://"));
      claims.push({
        id: `disputed-${a.record.id}`,
        kind: "official",
        tone: "alert",
        headline: `${a.dispute!.agency} appears to have lifted “${a.record.title}”`,
        detail: `Release ${a.dispute!.release}. Confirm with ${a.dispute!.agency} which applies before relying on either${src ? "; the original notice is linked in the evidence below" : ""}.`,
        source: { label: `${a.dispute!.agency} ${a.dispute!.release}`, url: a.dispute!.url },
        focus: "official",
        text: `${a.dispute!.agency} release ${a.dispute!.release} appears to lift “${a.record.title}”; no person has reviewed it, so Neritic shows it as neither in effect nor lifted.`,
        supports: [`official:${a.record.id}`],
      });
    }
    // 3. the notices that cover these waters (by their own geography), the most restrictive first
    if (applying.length) {
      const first = applying[0];
      const src = first.record.sources.find((s) => s.url.startsWith("https://"));
      const limit = first.caveats.find((c) => /limit/.test(c));
      claims.push({
        id: "notices",
        kind: "official",
        tone: applying.some((a) => RESTRICTIVE.has(a.record.action) || CLOSED.test(a.record.title)) ? "alert" : "note",
        headline: applying.length === 1 ? first.record.title : `${applying.length} notices cover these waters${forWhat}`,
        detail: applying.length === 1 ? `${first.record.agency}${first.record.effective_date ? ` · since ${shortDate(first.record.effective_date)}` : ""}${limit ? ` · ${limit}` : ""}` : applying.slice(0, 3).map((a) => a.record.title).join(" · "),
        source: src ? { label: `${first.record.agency} notice`, url: src.url } : undefined,
        focus: "official",
        text: `${applying.length} agency notice${applying.length === 1 ? "" : "s"} ${applying.length === 1 ? "names" : "name"} these waters${forWhat}: ${applying.slice(0, 3).map((a) => `${a.record.agency}: ${a.record.title}`).join("; ")}.${unverified ? " Neritic's copy, not verified by a person; the agency pages are the authority." : ""}`,
        supports: applying.map((a) => `official:${a.record.id}`),
      });
    } else if (!season) {
      claims.push({
        id: "notices",
        kind: "gap",
        tone: "gap",
        headline: `No notice in Neritic's copy names these waters${forWhat}`,
        detail: unresolved.length ? `${unresolved.length === 1 ? "One more" : `${unresolved.length} more`} may apply; ${unresolved.length === 1 ? "its" : "their"} stated area does not settle it. Not the same as open or safe.` : "Not the same as open or safe: the copy can lag the agencies.",
        focus: "official",
        text: `No notice in Neritic's copy names these waters${forWhat}. Not the same as open or safe: the copy can lag the agencies.`,
        supports: [],
      });
    }
  }

  // ---- shore measurements (CalHABMAP): the only toxin measurements that exist near shore
  const station = (e: StationEvidence, variable: "pn_cells" | "pda"): EvidenceNode => {
    const total = e.kind === "cells" ? e.values.reduce((a, v) => a + v.value, 0) : e.values[0].value;
    const rep = representative(variable, e.km);
    const setting = stationSetting(e.station.name);
    return {
      id: `${variable}:${e.station.station_id}`,
      class: "measurement",
      variable: variable === "pda" ? "particulate domoic acid in seawater" : "Pseudo-nitzschia cells",
      units: variable === "pda" ? "ng/mL" : "cells/L",
      value: total,
      text: fmtMeasured(variable === "pda" ? "pDA" : "pn_seriata", total),
      source: { id: "calhabmap", name: "CalHABMAP via SCCOOS ERDDAP", url: e.station.source_url, product: "HABs shore station" },
      time: { observed: e.date, ageDays: e.ageDays },
      support: { kind: "point" },
      where: { lat: e.station.lat, lon: e.station.lon, name: e.station.name, km: e.km, bearing: e.bearing, representative: rep, setting },
      freshness: { state: e.ageDays <= RECENT_DAYS ? "current" : e.ageDays <= 90 ? "stale" : "historical" },
      quality: { flags: [] },
      lineage: ["sampled at the pier", "counted or assayed in a laboratory", "published on SCCOOS ERDDAP", "read by Neritic's pipeline"],
      caveats: ["Seawater, not seafood: only CDPH and CDFW test shellfish and fish.", ...(setting === "sheltered-harbor" ? ["Sampled inside a harbor: sheltered water can differ from the coast outside."] : setting === "exposed-pier" ? ["Sampled at an open pier: exposed coastal water, which can differ from sheltered harbor water."] : [])],
    };
  };
  // the model, read once: the trip day's cells (used by the measured claim's context and by the model claim)
  const m = x.model !== "error" && x.model !== "loading" ? x.model : null;
  const day = m ? (x.tripDate ? (m.days.find((d) => d.valid === x.tripDate) ?? null) : (m.days[0] ?? null)) : null;
  const nb = day?.pda ?? null;
  const freshNote = m?.fresh && m.fresh.state !== "current" ? runAgeWords(m.fresh) : null;
  const modelWords = x.modelBeyond
    ? "NOAA's model does not reach this day"
    : nb?.here != null
      ? `NOAA's model gives these waters' own 3 km cell a ${fmtPct(nb.here)} chance of toxin above threshold`
      : nb && nb.n > 0
        ? `NOAA's C-HARM has no valid estimate for these nearshore waters; its offshore cells within ${NEIGHBOURHOOD_KM} km read ${fmtPct(nb.min)}–${fmtPct(nb.max)} (context only, not the harbour)`
        : m
          ? "these waters are outside NOAA's toxin model"
          : null;
  const modelLine = modelWords && freshNote && !x.modelBeyond ? `${modelWords} (${freshNote})` : modelWords;

  if (x.measuredError) {
    gaps.push({ id: "measured", variable: "shore measurements", reason: `Shore measurements could not be loaded (${x.measuredError}).` });
    claims.push({ id: "hab", kind: "gap", tone: "gap", headline: "Shore measurements could not be loaded", focus: null, text: "Shore measurements could not be loaded.", supports: [] });
  } else if (x.measured) {
    const used: { node: EvidenceNode; e: StationEvidence }[] = [];
    for (const [e, v] of [
      [x.measured.cells, "pn_cells"],
      [x.measured.toxin, "pda"],
    ] as const) {
      if (!e) continue;
      const n = station(e, v);
      nodes.push(n);
      if (e.near && e.ageDays <= RECENT_DAYS) used.push({ node: n, e });
      else gaps.push({ id: `${v}-recent`, variable: n.variable, reason: e.near ? `No ${n.variable} sample within ${EVIDENCE_RADIUS_KM} km in the last ${RECENT_DAYS} days; the latest nearby is ${ageWords(e.ageDays)} at ${e.station.name} (${km(e.km)} km).` : `No shore station within ${EVIDENCE_RADIUS_KM} km measures ${n.variable}; the nearest that does, ${e.station.name}, is ${km(e.km)} km away, too far to describe these waters.`, nearest: { name: e.station.name, km: e.km, date: e.date } });
    }
    const best = [...used].sort((a, b) => b.e.date.localeCompare(a.e.date) || a.e.km - b.e.km)[0];
    const c = x.measured.nearest;
    if (best) {
      const e = best.e;
      const closer = c && c.station.station_id !== e.station.station_id && c.km < e.km ? c : null;
      // the closer station's silence is evidence too: its own node, with its last sample date
      const closerNode: EvidenceNode | null = closer
        ? {
            id: `silent:${closer.station.station_id}`,
            class: "measurement",
            variable: "shore station's last sample",
            units: "",
            value: null,
            text: `last sample ${longDate(closer.date, x.now)}`,
            source: { id: "calhabmap", name: "CalHABMAP via SCCOOS ERDDAP", url: closer.station.source_url, product: "HABs shore station" },
            time: { observed: closer.date, ageDays: closer.ageDays },
            support: { kind: "point" },
            where: { lat: closer.station.lat, lon: closer.station.lon, name: closer.station.name, km: closer.km, bearing: null, representative: representative("pn_cells", closer.km), setting: stationSetting(closer.station.name) },
            freshness: { state: closer.ageDays <= RECENT_DAYS ? "current" : closer.ageDays <= 90 ? "stale" : "historical" },
            quality: { flags: [] },
            lineage: ["CalHABMAP shore station", "read by Neritic's pipeline"],
            caveats: ["A station with no comparable recent sample: no sample is not a clean result."],
          }
        : null;
      if (closerNode) nodes.push(closerNode);
      const v = e.kind === "cells" ? `Pseudo-nitzschia ${best.node.text}` : `domoic acid in seawater ${best.node.text}`;
      claims.push({
        id: "hab",
        kind: "measured",
        tone: "note",
        headline: `${e.station.name}, ${km(e.km)} km ${e.bearing}: ${v}, ${shortDate(e.date)}`,
        detail: `${cap(ageWords(e.ageDays))}. ${closer ? `${closer.station.name}, ${km(closer.km)} km, is closer but its last sample is from ${shortDate(closer.date)} (${ageWords(closer.ageDays)}). ` : ""}${modelLine ? `${cap(modelLine)}. ` : ""}Seawater, not seafood.`,
        focus: e.kind === "cells" ? "cells" : "toxin",
        aged: e.ageDays > 14,
        text: `${e.station.name} reported ${best.node.text} ${e.kind === "cells" ? "Pseudo-nitzschia" : "particulate domoic acid"} on ${longDate(e.date)}, ${km(e.km)} km ${e.bearing} of ${P.name}.${closer ? ` ${closer.station.name} is closer (${km(closer.km)} km) but has no comparable sample since ${longDate(closer.date, x.now)}.` : ""}`,
        supports: [best.node.id, ...(closerNode ? [closerNode.id] : [])],
      });
      // the same sample against the station's own record (only when the record's newest sample is this one)
      const rv = e.kind === "cells" ? "pn" : "pDA";
      const r = e.station.record?.[rv];
      if (r && r.date === e.date) {
        const rn = recordNode(r, best.node, e.station.name, x.now);
        nodes.push(rn);
        const hab = claims.at(-1)!;
        // its own field, not appended to `text`: the HAB detail's summary paragraph repeats `text`, and nothing is said twice
        hab.history = historySentence(r, e.station.name, x.now);
        hab.supports.push(rn.id);
      }
    } else if (nb?.here != null) {
      claims.push({
        id: "hab",
        kind: "model",
        tone: "note",
        headline: `NOAA model: ${fmtPct(nb.here)} chance toxin in the water exceeds its threshold here${freshNote ? ` (${freshNote})` : ""}`,
        detail: `A probability for this 3 km cell${day?.valid ? ` on ${shortDate(day.valid)}` : ""}, not a measurement.${freshNote ? " NOAA has published no newer run." : ""} No shore sample within ${EVIDENCE_RADIUS_KM} km in the last ${RECENT_DAYS} days.`,
        focus: "model",
        aged: !!freshNote,
        text: `No shore sample within ${EVIDENCE_RADIUS_KM} km in the last ${RECENT_DAYS} days. NOAA's C-HARM gives the 3 km cell containing ${P.name} a ${fmtPct(nb.here)} chance that particulate domoic acid exceeds its threshold${day?.valid ? ` on ${longDate(day.valid)}` : ""}${freshNote ? ` (${freshNote})` : ""}; a probability, not a measurement.`,
        supports: day?.valid ? [`charm:own:${day.valid}`] : [],
      });
    } else {
      if (!x.measured.cells && !x.measured.toxin) gaps.push({ id: "measured-none", variable: "shore measurements", reason: "No shore station measures bloom algae or toxin near these waters." });
      claims.push({
        id: "hab",
        kind: "gap",
        tone: "gap",
        headline: `No recent HAB sample within ${EVIDENCE_RADIUS_KM} km`,
        // the model is the next line's own claim (pageClaims keeps it when the HAB line is a gap), so
        // this line says only what the shore stations say: nothing is said twice
        detail: `${c ? `The nearest station, ${c.station.name} (${km(c.km)} km), last reported ${ageWords(c.ageDays)}. ` : `No shore station within ${EVIDENCE_RADIUS_KM} km samples bloom algae or toxin. `}No sample is not a clean result.`,
        focus: null,
        text: `No shore station within ${EVIDENCE_RADIUS_KM} km of ${P.name} has a bloom or toxin sample from the last ${RECENT_DAYS} days${c ? `; the nearest, ${c.station.name} (${km(c.km)} km), last reported ${ageWords(c.ageDays)}` : ""}. No sample is not a clean result.`,
        supports: [],
      });
    }
  }

  // ---- the chosen species: what the agencies test it for, and what Neritic's readings can say about that
  if (x.official && x.official.species.id !== "any") speciesClaim(x.official.species, x.official.assocs, nodes, claims);

  // ---- C-HARM (published forecast): the waters' own cell, or the offshore cells as context
  if (x.model === "error") gaps.push({ id: "model", variable: "C-HARM", reason: "No C-HARM run could be read." });
  else if (m) {
    const base = {
      class: "published_forecast" as const,
      variable: "probability that particulate domoic acid exceeds 500 ng/L",
      units: "probability",
      source: { id: "charm", name: "NOAA C-HARM", url: "https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.html", product: "C-HARM", version: m.version ?? "3.1" },
      time: { valid: day?.valid ?? null, issued: m.issued, ageDays: m.fresh?.ageDays ?? null },
      freshness: { state: (m.fresh?.state ?? "unknown") as EvidenceNode["freshness"]["state"], ...(freshNote ? { note: freshNote } : {}) },
      quality: { flags: [] as string[] },
      lineage: ["NOAA's statistical model on satellite and ocean-model inputs", "published on NOAA CoastWatch ERDDAP", "read by Neritic's pipeline, values unchanged"],
      caveats: ["A probability for a 3 km cell, not a toxin measurement or a closure decision.", "A low probability does not mean an area is safe.", ...(freshNote ? [`NOAA's runs are often missing a day or more: ${freshNote}.`] : [])],
    };
    const habKind = claims.find((cl) => cl.id === "hab")?.kind;
    if (x.modelBeyond || !day) {
      const last = m.days.at(-1)?.valid ?? null;
      gaps.push({ id: "model-day", variable: "C-HARM", reason: `NOAA's model does not reach ${x.tripDate ? shortDate(x.tripDate) : "that day"}${last ? ` (last day ${shortDate(last)})` : ""}.` });
      claims.push({ id: "model", kind: "gap", tone: "gap", headline: `NOAA's model does not reach ${x.tripDate ? shortDate(x.tripDate) : "this day"}`, detail: last ? `Its newest run ends ${shortDate(last)}.` : undefined, focus: null, text: `NOAA's C-HARM run does not reach ${x.tripDate ? longDate(x.tripDate) : "that day"}.`, supports: [] });
    } else if (nb?.here != null) {
      const n: EvidenceNode = { ...base, id: `charm:own:${day.valid}`, value: nb.here, text: fmtPct(nb.here), support: { kind: "cell", nativeM: 3000 }, where: { lat: P.lat, lon: P.lon, name: "these waters' own 3 km cell", km: 0, bearing: null, representative: "at" } };
      nodes.push(n);
      claims.push({ id: "model", kind: "model", tone: "note", headline: `NOAA model: ${fmtPct(nb.here)} chance toxin exceeds its threshold in these waters' own cell${freshNote ? ` (${freshNote})` : ""}`, detail: `A probability for the 3 km cell on ${shortDate(day.valid!)}, not a measurement.`, focus: "model", aged: !!freshNote, text: `NOAA's C-HARM gives the 3 km cell containing ${P.name} a ${fmtPct(nb.here)} chance that particulate domoic acid exceeds its threshold on ${longDate(day.valid!)}${freshNote ? ` (${freshNote})` : ""}. A probability, not a measurement.`, supports: [n.id] });
    } else if (nb && nb.n > 0) {
      const n: EvidenceNode = { ...base, id: `charm:offshore:${day.valid}`, value: nb.median, text: `${fmtPct(nb.min)}–${fmtPct(nb.max)}`, support: { kind: "cells", nativeM: 3000, radiusKm: NEIGHBOURHOOD_KM, n: nb.n }, where: { lat: nb.nearest?.cell.lat ?? null, lon: nb.nearest?.cell.lon ?? null, name: `${nb.n} offshore cells within ${NEIGHBOURHOOD_KM} km`, km: nb.nearest?.km ?? NEIGHBOURHOOD_KM, bearing: null, representative: "context" }, caveats: [...base.caveats, "Offshore context only: none of these cells contains the waters."] };
      nodes.push(n);
      gaps.push({ id: "model-here", variable: "C-HARM", reason: `NOAA's C-HARM model has no valid estimate in these nearshore waters; its nearest cell with a value is ${km(nb.nearest!.km)} km out.`, nearest: { name: "nearest C-HARM cell", km: nb.nearest!.km } });
      claims.push({ id: "model", kind: "gap", tone: "gap", headline: "No valid NOAA model estimate for these nearshore waters", detail: `Nearest cell ${km(nb.nearest!.km)} km out; the ${nb.n} offshore cells within ${NEIGHBOURHOOD_KM} km read ${fmtPct(nb.min)}–${fmtPct(nb.max)} for ${shortDate(day.valid!)}, context for the open water, not the harbour.`, focus: "model", text: `NOAA's C-HARM model has no valid estimate in these nearshore waters. Its nearest cell is ${km(nb.nearest!.km)} km out; the ${nb.n} offshore cells within ${NEIGHBOURHOOD_KM} km read ${fmtPct(nb.min)}–${fmtPct(nb.max)} for ${longDate(day.valid!)}, context for the open water, not the harbor.`, supports: [n.id] });
    } else if (nb?.nearest) {
      gaps.push({ id: "model-here", variable: "C-HARM", reason: `These waters are outside NOAA's model: no cell within ${NEIGHBOURHOOD_KM} km. Its nearest cell, ${km(nb.nearest.km)} km away, reads ${fmtPct(nb.nearest.value)}; that describes that cell, not these waters.`, nearest: { name: "nearest C-HARM cell", km: nb.nearest.km } });
      claims.push({ id: "model", kind: "gap", tone: "gap", headline: `Outside NOAA's toxin model: no cell within ${NEIGHBOURHOOD_KM} km`, detail: `Nearest cell ${km(nb.nearest.km)} km away reads ${fmtPct(nb.nearest.value)}; it describes that cell, not these waters.`, focus: null, text: `NOAA's C-HARM model has no estimate within ${NEIGHBOURHOOD_KM} km of ${P.name}; its nearest cell is ${km(nb.nearest.km)} km away.`, supports: [] });
    } else if (nb) {
      gaps.push({ id: "model-here", variable: "C-HARM", reason: "These waters are outside NOAA's model: no cell within 80 km." });
      claims.push({ id: "model", kind: "gap", tone: "gap", headline: "Outside NOAA's toxin model", detail: "No model cell within 80 km.", focus: null, text: `${P.name} is outside NOAA's C-HARM model: no cell within 80 km.`, supports: [] });
    }
    // the HAB line already carried the model (its own cell, or the context sentence): the model's own line is a
    // detail for the API and the evidence list, not a second line on the page
    if (habKind === "model" || habKind === "measured") {
      const mc = claims.find((cl) => cl.id === "model");
      if (mc) mc.detail = mc.detail ? `${mc.detail} (Said with the measurement above.)` : mc.detail;
    }
  }

  // ---- satellite chlorophyll (retrieval): algae of every kind, not toxin
  if (x.sat === "error") gaps.push({ id: "satellite", variable: "satellite chlorophyll", reason: "The satellite grid could not be read." });
  else if (x.sat && x.sat.value != null) {
    const s = x.sat;
    const age = s.date ? Math.floor((x.now.getTime() - Date.parse(`${s.date}T12:00:00Z`)) / DAY) : null;
    const n: EvidenceNode = {
      id: `chl:${s.layerId}`,
      class: "satellite_retrieval",
      variable: "chlorophyll-a",
      units: "mg/m³",
      value: s.value,
      text: `${Number(s.value.toPrecision(2))} mg/m³`,
      source: { id: "satellite_chl", name: `NOAA CoastWatch, ${s.sensor}`, product: s.layerId },
      time: { observed: s.date, ageDays: age },
      support: s.basis ? { kind: "pixels", radiusKm: s.basis.radiusKm, n: s.basis.n, nativeM: /300/.test(s.sensor) ? 300 : 750 } : { kind: "pixel", nativeM: /300/.test(s.sensor) ? 300 : 750 },
      where: { lat: null, lon: null, name: s.basis ? `median of ${s.basis.n} clear pixels within ${s.basis.radiusKm} km` : "the pixel at these waters", km: s.km, bearing: null, representative: representative("chl_satellite", s.km) },
      freshness: { state: age == null ? "unknown" : age <= 3 ? "current" : age <= 7 ? "stale" : "historical" },
      quality: { flags: s.aside ? [`a ${s.aside.reason === "shore" ? "shore-adjacent" : "spike"} pixel (${Number(s.aside.value.toPrecision(2))}) was set aside`] : [] },
      lineage: ["ocean-colour retrieval by NOAA CoastWatch", "latest clear pixel within 7 days", "median within 5 km when the pixel is unclear"],
      caveats: ["Algae of every kind, not toxin: high chlorophyll does not confirm a harmful bloom, and low does not rule one out."],
    };
    nodes.push(n);
    claims.push({
      id: "satellite",
      kind: "satellite",
      tone: "note",
      headline: `Satellite chlorophyll ${n.text}${s.date ? `, ${shortDate(s.date)}` : ""}`,
      detail: `${s.basis ? `Median of ${s.basis.n} clear pixels within ${s.basis.radiusKm} km` : "The pixel at these waters"} · ${s.sensor}${age != null ? ` · ${ageWords(age)}` : ""}. Algae of every kind, not toxin.`,
      focus: "satellite",
      aged: age != null && age > 3,
      text: `Satellite chlorophyll ${s.basis ? `near ${P.name} (median of ${s.basis.n} clear pixels within ${s.basis.radiusKm} km)` : `at ${P.name}`} was ${n.text}${s.date ? ` on ${longDate(s.date)}` : ""} (${s.sensor}). Algae, not toxin.`,
      supports: [n.id],
    });
  } else if (x.sat) gaps.push({ id: "satellite", variable: "satellite chlorophyll", reason: x.sat.flagged ? `No open-water chlorophyll reading within ${x.sat.radiusKm} km: the pixels there are shore-adjacent or flagged.` : `No clear satellite view within ${x.sat.radiusKm} km in the last ${x.sat.windowDays} days: cloud, fog or no overpass, not low chlorophyll.` });

  // ---- live conditions (measurements and an official forecast), context never a HAB forecast
  const c = x.conditions;
  if (c) {
    const parts: string[] = [];
    const supports: string[] = [];
    let source: Claim["source"];
    let nws: string | null = null;
    if (c.marine.status === "ok") {
      const p = periodFor(c.marine.data.periods, x.tripDate, x.today ?? x.tripDate);
      if (p) {
        const n: EvidenceNode = {
          id: `nws:${c.marine.data.zone}:${p.name.replace(/\s+/g, "-")}`,
          class: "published_forecast",
          variable: "marine forecast",
          units: "",
          value: null,
          text: p.text,
          source: { id: "nws", name: "NOAA National Weather Service", url: c.marine.url, product: c.marine.data.zone },
          time: { issued: c.marine.data.updated, ageDays: null },
          support: { kind: "zone" },
          where: { lat: null, lon: null, name: c.marine.data.zoneName, km: 0, bearing: null, representative: "near" },
          freshness: { state: "current" },
          quality: { flags: [] },
          lineage: ["written by NWS forecasters for the whole zone", "read from NWS's text product"],
          caveats: ["A forecast for the whole marine zone, in NWS's words."],
        };
        nodes.push(n);
        supports.push(n.id);
        nws = `NWS ${/tonight|today|this afternoon/i.test(p.name) ? p.name.toLowerCase() : `for ${p.name}`}: ${marineHeadline(p)}`;
        source = { label: `NWS forecast, ${c.marine.data.zoneName}`, url: c.marine.url };
      }
    }
    // the harbour's own water, measured at the tide station inside it (never the offshore buoy's)
    if (c.harbor?.status === "ok" && c.harbor.station) {
      const h = c.harbor;
      const st = h.station!;
      const inHarbour = st.km <= REPRESENTATIVE_KM.water_temp.at;
      const n: EvidenceNode = {
        id: `coops:${st.id}:water_temperature`,
        class: "measurement",
        variable: "water temperature",
        units: "°C",
        value: h.data.waterC.value,
        text: `${h.data.waterC.value.toFixed(1)} °C`,
        source: { id: "coops", name: "NOAA CO-OPS", url: h.url, product: `tide station ${st.id}` },
        time: { observed: h.data.waterC.at, ageDays: 0 },
        support: { kind: "point" },
        where: { lat: st.lat ?? null, lon: st.lon ?? null, name: `${stationName(st.name)} tide station`, km: st.km, bearing: null, representative: representative("water_temp", st.km), setting: inHarbour ? "sheltered-harbor" : "unclassified" },
        freshness: { state: "current" },
        quality: { flags: h.data.flags ? [`NOAA flags ${h.data.flags} (pass)`] : [] },
        lineage: ["thermistor at the tide station, 6-minute average", "CO-OPS data API", "read live by Neritic's server"],
        caveats: ["Water temperature at the station's pier, ±0.2 °C; the harbour entrance and the open coast can differ."],
      };
      nodes.push(n);
      supports.push(n.id);
      parts.push(`Water ${n.text} ${inHarbour ? "in the harbour" : `at ${stationName(st.name)} tide station, ${km(st.km)} km`}`);
      source ??= { label: `NOAA tide station ${st.id}`, url: h.url };
    }
    if (c.buoy.status === "ok" && c.buoy.station) {
      const b = c.buoy.data;
      const st = c.buoy.station;
      const setting = stationSetting(st.name);
      const add = (key: keyof typeof b, variable: string, units: string, rep: keyof typeof REPRESENTATIVE_KM, fmt: (v: number) => string) => {
        const o = b[key];
        if (!o) return null;
        const n: EvidenceNode = {
          id: `buoy:${st.id}:${key}`,
          class: "measurement",
          variable,
          units,
          value: o.value,
          text: fmt(o.value),
          source: { id: "ndbc", name: "NOAA National Data Buoy Center", url: c.buoy.status === "ok" ? c.buoy.url : null, product: st.id },
          time: { observed: o.at, ageDays: 0 },
          support: { kind: "point" },
          where: { lat: st.lat ?? null, lon: st.lon ?? null, name: stationName(st.name), km: st.km, bearing: null, representative: representative(rep, st.km), setting },
          freshness: { state: "current" },
          quality: { flags: [] },
          lineage: ["measured at the station", "NDBC realtime2 text", "read live by Neritic's server"],
          caveats: [setting === "offshore" ? "An offshore buoy: seas at the harbor entrance can differ." : setting === "sheltered-harbor" ? "A station inside the harbor: conditions outside can differ." : "Wind and seas, not a HAB or fishing forecast."],
        };
        nodes.push(n);
        supports.push(n.id);
        return n;
      };
      const wind = add("windKt", "wind", "kt", "wind", (v) => `${Math.round(v)} kt${b.windDirDeg ? ` ${compass(b.windDirDeg.value)}` : ""}`);
      const wave = add("waveFt", "significant wave height", "ft", "waves", (v) => `${v.toFixed(1)} ft${b.periodS ? ` at ${Math.round(b.periodS.value)} s` : ""}`);
      const w = [wind && `wind ${wind.text}`, wave && `waves ${wave.text}`].filter(Boolean).join(", ");
      if (w) {
        parts.push(`${cap(w)} at ${stationName(st.name)} buoy now`);
        source ??= { label: `NDBC ${st.id}`, url: c.buoy.url };
      }
    }
    if (c.tides.status === "ok") {
      const dayTides = tidesOn(c.tides.data, x.tripDate);
      const lows = dayTides.filter((t) => t.type === "L").sort((a, b) => a.ft - b.ft);
      if (dayTides.length) {
        const st = c.tides.station;
        const n: EvidenceNode = {
          id: `tides:${st?.id ?? "nearest"}:${x.tripDate}`,
          class: "published_forecast",
          variable: "tide predictions",
          units: "ft above mean lower low water",
          value: lows[0]?.ft ?? null,
          text: dayTides.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft at ${clock(t.at)}`).join(", "),
          source: { id: "coops-tides", name: "NOAA CO-OPS tide predictions", url: c.tides.url, product: st ? `station ${st.id}` : undefined },
          time: { valid: x.tripDate, ageDays: null },
          support: { kind: "point" },
          where: { lat: st?.lat ?? null, lon: st?.lon ?? null, name: st ? `${stationName(st.name)} tide station` : "the nearest tide station", km: st?.km ?? 0, bearing: null, representative: st ? representative("tide", st.km) : "near" },
          freshness: { state: "current" },
          quality: { flags: [] },
          lineage: ["astronomical tide predictions by NOAA CO-OPS", "read live by Neritic's server"],
          caveats: ["Predicted tides: wind and swell change the water level actually reached."],
        };
        nodes.push(n);
        supports.push(n.id);
      }
      if (x.official?.audience === "shellfish" && lows.length) parts.unshift(`Lowest tide ${lows[0].ft.toFixed(1)} ft at ${clock(lows[0].at)}`);
      else if (dayTides.length) parts.push(`tides ${dayTides.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft`).slice(0, 2).join(", ")}`);
      source ??= { label: "NOAA tide predictions", url: c.tides.url };
    }
    const marine = c.marine.status === "ok" ? null : (c.marine.url ?? null);
    const advisory = c.alerts.status === "ok" ? c.alerts.data[0] : null;
    if (parts.length || marine || nws) {
      const head = nws ?? (parts.length ? parts.join(" · ") : "No buoy or tide station reports near these waters");
      const rest = nws ? parts : [];
      claims.push({
        id: "conditions",
        kind: "conditions",
        tone: advisory ? "alert" : "note",
        headline: advisory ? `${advisory.event} · ${head}` : head,
        detail: [rest.length ? `${cap(rest.join(" · "))}.` : "", nws ? (advisory ? `NWS: “${advisory.headline}”.` : "Wind and seas, not a HAB or fishing forecast.") : "Marine forecast and warnings: read NWS for your zone."].filter(Boolean).join(" "),
        source: source ?? (marine ? { label: "NWS marine forecast", url: marine } : undefined),
        focus: "conditions",
        text: `${advisory ? `${advisory.event}. ` : ""}${[nws, ...parts].filter(Boolean).join(". ")}. Wind, seas and tides, not a HAB or fishing forecast.`,
        supports,
      });
    }
  }

  // every claim cites nodes that exist
  const ids = new Set(nodes.map((n) => n.id));
  for (const cl of claims) cl.supports = cl.supports.filter((id) => ids.has(id));

  return {
    place: P,
    generatedAt: x.now.toISOString(),
    window: { from: new Date(x.now.getTime() - 365 * DAY).toISOString().slice(0, 10), to: x.tripDate ?? today },
    nodes,
    gaps,
    claims,
    limits: [
      "No node is a seafood-safety statement, a closure decision or a catch prediction.",
      "No value is interpolated between cells or stations; distances are stated and representativeness is per variable.",
      "Neritic's own experimental estimates are not included until independently validated.",
    ],
  };
}

/** The claims the page shows as its "right now" lines: the model's own line is folded into the HAB line when that line already said it. */
export function pageClaims(g: EvidenceGraph): Claim[] {
  const hab = g.claims.find((c) => c.id === "hab");
  const saidWithHab = hab && (hab.kind === "measured" || hab.kind === "model");
  return g.claims.filter((c) => !(c.id === "model" && saidWithHab));
}

/**
 * "What the readings mean for Dungeness crab": the agency's own sentences about testing the animal,
 * then which of Neritic's readings bear on those toxins. Never says a species is safe or affected.
 */
/** the animal as the species line names it ("not in the crab") */
const ANIMAL: Partial<Record<Species["id"], string>> = { dungeness: "crab", "rock-crab": "crab", lobster: "lobster", "razor-clam": "clams", mussel: "mussels", bivalve: "shellfish", "anchovy-sardine": "fish", salmon: "fish", groundfish: "fish", halibut: "fish", squid: "squid" };

function speciesClaim(sp: Species, assocs: Association[], nodes: EvidenceNode[], claims: Claim[]) {
  const ev = SPECIES_EVIDENCE[sp.id];
  const phrase = speciesPhrase(sp);
  // the toxins the agencies name for it: their pages, and the notices here that name the species
  const fromNotices = assocs.filter((a) => recordNamesSpecies(a.record.species, sp)).flatMap((a) => a.record.toxins ?? []);
  const toxins = [...new Set<Toxin>([...(ev?.toxins ?? []), ...fromNotices.filter((t): t is Toxin => t === "domoic_acid" || t === "psp")])];
  const factNodes = (ev?.facts ?? []).map((f, i): EvidenceNode => ({
    id: `agency-fact:${sp.id}:${i}`,
    class: "measurement",
    variable: "agency page",
    units: "",
    value: null,
    text: f.text,
    source: { id: "official", name: f.source.label, url: f.source.url },
    time: { ageDays: null },
    support: { kind: "zone" },
    where: { lat: null, lon: null, name: "statewide", km: 0, bearing: null, representative: "at" },
    freshness: { state: "current" },
    quality: { flags: [`the agency's sentence, read ${f.read}`] },
    lineage: ["agency page", "quoted by Neritic"],
    caveats: ["The agency's page is the authority; it can change between Neritic's readings of it."],
  }));
  nodes.push(...factNodes);
  const animal = ANIMAL[sp.id] ?? "animal";
  const readings = toxins.length
    ? toxins.map((t) => (READINGS_FOR[t] === "carried" ? `Neritic's shore readings and NOAA's model are about ${TOXIN_WORDS[t]} in the water, not in the ${animal}.` : `Neritic carries no seawater reading of the algae behind ${TOXIN_WORDS[t]}; CDPH's tests of the shellfish are the evidence.`))
    : [`The shore readings describe the water, not the ${animal}.`];
  const first = ev?.facts[0];
  claims.push({
    id: "species",
    kind: toxins.length ? "official" : "gap",
    tone: toxins.length ? "note" : "gap",
    headline: toxins.length ? `For ${phrase}, the agencies watch for ${toxins.map((t) => TOXIN_WORDS[t]).join(" and ")}` : `Neritic's sources name no biotoxin test specific to ${phrase}`,
    detail: [...(ev?.facts.slice(1).map((f) => f.text) ?? []), ...readings].join(" "),
    quote: first?.quote,
    source: first?.source,
    focus: null,
    text: `${[...(ev?.facts.map((f) => f.text) ?? []), ...readings].join(" ")}`,
    supports: factNodes.map((n) => n.id),
  });
}

const fmtRecord = (r: RecordContext, v: number) => fmtMeasured(r.variable === "pDA" ? "pDA" : "pn_seriata", v);

/** "Compared with earlier years at Santa Cruz Wharf: higher than 98% of the Aug–Oct samples there in 2014–2025. Last as high or higher: September 2 (…). Previous sample September 23: …." */
export function historySentence(r: RecordContext, station: string, now: Date): string {
  const rank = rankWords(r);
  const first = r.since.slice(0, 4);
  const head = rank ? `Compared with earlier years at ${station}: ${rank}.` : `Too few samples at ${station} to compare by season (${r.season.n} from the same months of earlier years; its record starts in ${first}).`;
  // "as high or higher" says nothing about a zero (every earlier sample is at least 0)
  const high = r.value === 0 ? "" : r.lastAsHigh ? ` Last as high or higher: ${longDate(r.lastAsHigh.date, now)} (${fmtRecord(r, r.lastAsHigh.value)}).` : ` The highest there since the record began in ${first}.`;
  const prev = r.previous ? ` Previous sample ${longDate(r.previous.date, now)}: ${fmtRecord(r, r.previous.value)}.` : "";
  return `${head}${high}${prev}`;
}

function recordNode(r: RecordContext, m: EvidenceNode, station: string, now: Date): EvidenceNode {
  return {
    id: `record:${r.variable}:${m.id.split(":").slice(1).join(":")}`,
    class: "record_statistic",
    variable: `${m.variable}, against the station's own record`,
    units: "share of same-season samples from earlier years that were lower",
    value: r.season.below,
    // the sentence the briefing says, so every number in it is this node's
    text: historySentence(r, station, now),
    source: m.source,
    time: { observed: r.date, ageDays: m.time.ageDays },
    support: { kind: "point", n: r.season.n },
    where: m.where,
    freshness: m.freshness,
    quality: { flags: r.season.below == null ? [`${r.season.n} same-season samples from ${r.season.years} earlier years; at least 12 from 3 years are needed`] : [] },
    lineage: [`${station}'s CalHABMAP series since ${longDate(r.since, now)} (${r.n} samples)`, `${r.season.n} samples from the same months in ${r.season.years} earlier years`, "compared by Neritic, values unchanged"],
    caveats: [
      "A comparison with this station's own past samples: not with other waters, not a statement about seafood, and not a forecast.",
      "Sampling has not been even across years; weeks without a sample are not counted.",
    ],
  };
}

function statementNode(s: Statement): EvidenceNode {
  return {
    id: `statement:${s.id}`,
    class: "measurement",
    variable: "agency statement",
    units: "",
    value: null,
    text: s.statement,
    source: { id: "official", name: s.agency, url: s.source.url, product: s.topic },
    time: { ageDays: null },
    support: { kind: "zone" },
    where: { lat: null, lon: null, name: "statewide", km: 0, bearing: null, representative: "at" },
    freshness: { state: "current" },
    quality: { flags: ["Neritic's transcription of the agency page"] },
    lineage: ["agency page", "transcribed by Neritic's pipeline"],
    caveats: ["The agency's page is the authority; it can change between Neritic's checks."],
  };
}

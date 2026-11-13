import type { FisheryId, StatusLevel } from "@/pilot/contracts";
import type { SeaView, TodayModel } from "@/pilot/facts";
import { windWords } from "@/pilot/today/marine";
import { day } from "@/nx/ds/format";
import type { CoastFishery } from "./fisheries";
import { sectorOf, whaleMeasures } from "@/nx/regulation/whale";

/**
 * What My Coast's margin shows above the worksheet, built from the same model Today uses
 * (pilot/facts `buildTodayModel`, so a status line here and on Today is the same statement) and trimmed to
 * one fishery: the gates (season, whale measures, the latest test) and the sea window. Pure: the route
 * (app/(neritic)/coast/data) loads, this shapes. Regulatory attribution is kept as the records give it:
 * CDFW acts on fisheries, CDPH tests and advises (T42), and nothing official is ever called verified (T38).
 */

export type Tone = "ink" | "regulation" | "closure";

export type SeasonGate = {
  /** the record's own sentence */
  text: string;
  tone: Tone;
  /** the record's own level; "delayed" and "closed" are what make a planned trip one that cannot be made now */
  level: StatusLevel | "unknown";
  /** the record is in conflict with another of Neritic's sources (its level is then not relied on) */
  conflict: boolean;
  /** an ordinary closed season on the calendar, not an action taken this year */
  seasonal: boolean;
  /** the agency the record quotes ("CDFW") */
  source: string;
  url: string | null;
  /** when the record was last read against the agency's page (ISO) */
  checked: string | null;
  verified: boolean | null;
  stale: boolean;
  /** two of Neritic's sources disagree: said in words, never resolved (T40) */
  differ: string | null;
};

export type WhaleGate = {
  zone: string;
  area: string;
  /** what applies in the zone, the season delay already said in the season line left out */
  measures: string;
  /** the next assessment as CDFW words it, dates in the house style ("Dec 15–18, then Jan 13–15, subject to change") */
  next: string | null;
  assessed: string | null;
  url: string | null;
  verified: boolean;
};

export type TestGate = {
  /** "Monterey Bay" */
  site: string;
  /** "Dungeness crab" */
  species: string;
  date: string;
  tissue: string;
  /** each sample as CDPH printed it ("<2.5", "ND", "9.8"): censored values stay censored (T7) */
  raw: string[];
  values: (number | null)[];
  max: number | null;
  avg: number | null;
  /** samples at or over the action level */
  over: number;
  action: number | null;
  ageDays: number | null;
  stale: boolean;
  source: string;
  url: string;
  /** where CDPH collected it, when CDPH names a known place; null otherwise (never guessed) */
  lat: number | null;
  lon: number | null;
};

export type CoastGates = {
  ok: true;
  now: string;
  fixture: boolean;
  fishery: FisheryId;
  season: SeasonGate | null;
  whale: WhaleGate | null;
  /** null: no CDPH test for this fishery informs this port */
  test: TestGate | null;
  /** the season list could not be read */
  statusError: boolean;
  testsError: boolean;
};

export type SeaPeriod = { name: string; date: string | null; night: boolean; text: string; wind: string | null; seasLo: number | null; seasHi: number | null };
export type CoastSea = { ok: true; now: string; sea: Omit<SeaView, "lanes"> & { lanes: SeaPeriod[] } };
export type CoastError = { ok: false; error: string };

const TONE: Record<string, Tone> = { closed: "closure", delayed: "regulation", restricted: "regulation", watch: "regulation" };

const MONTHS: Record<string, string> = { January: "Jan", February: "Feb", March: "Mar", April: "Apr", August: "Aug", September: "Sep", October: "Oct", November: "Nov", December: "Dec" };

/** CDFW's window words in the house style: "December 15-18 (then January 13-15), subject to change" → "Dec 15–18, then Jan 13–15, subject to change". */
export function windowWords(w: string): string {
  return w
    .replace(/\b(January|February|March|April|August|September|October|November|December)\b/g, (m) => MONTHS[m])
    .replace(/(\d)\s*-\s*(\d)/g, "$1–$2")
    .replace(/\s*\(then ([^)]+)\)/g, ", then $1")
    .trim();
}

const sentence = (s: string) => {
  const t = s.trim().replace(/\s+/g, " ");
  return t ? `${t.charAt(0).toUpperCase()}${t.slice(1)}${/[.!?]$/.test(t) ? "" : "."}` : "";
};

const species = (s: string) => (/dungeness/i.test(s) ? "Dungeness crab" : s.charAt(0).toUpperCase() + s.slice(1).toLowerCase());

/** The gates for one fishery at one port, from Today's model. */
export function gatesFrom(m: TodayModel, f: CoastFishery, opts: { now: string; fixture: boolean }): CoastGates {
  const row = m.status.find((r) => r.fishery === f.id) ?? null;
  const season: SeasonGate | null = row
    ? {
        text: row.conflict ? `Sources differ about this fishery here; awaiting a person's review.` : sentence(row.headline),
        tone: row.conflict ? "regulation" : row.seasonal ? "ink" : (TONE[row.level] ?? "ink"),
        level: row.level,
        conflict: !!row.conflict,
        seasonal: !!row.seasonal,
        source: row.source?.name ?? "CDFW",
        url: row.source?.url ?? null,
        checked: row.checked_at,
        verified: row.verified,
        stale: row.stale,
        differ: row.conflict?.words ?? null,
      }
    : null;
  const seasonSaysIt = !!row && (row.level === "delayed" || row.level === "closed") && row.cause === "whale_entanglement";
  const w = f.ramp ? m.whale : null;
  const whale: WhaleGate | null = w
    ? {
        zone: w.zone,
        area: w.label.replace(/\s*\([^)]*\)/g, ""),
        measures: whaleMeasures(w, sectorOf(f.id), { seasonSaysIt }).sentences.join(" "),
        next: w.next_window ? windowWords(w.next_window) : w.next_assessment ? day(w.next_assessment) : null,
        assessed: w.last_assessment,
        url: w.source_url,
        verified: w.verified === true,
      }
    : null;
  const t = m.tests.find((x) => x.fishery === f.id && x.latest) ?? null;
  const latest = t?.latest ?? null;
  const test: TestGate | null =
    t && latest
      ? {
          site: t.site.site_label || t.site.bma_label || "CDPH site",
          species: species(t.site.species),
          date: latest.date,
          tissue: latest.tissue,
          raw: (latest as typeof latest & { values_raw?: string[] }).values_raw ?? latest.values_ppm.map((x) => (x == null ? "ND" : String(x))),
          values: latest.values_ppm,
          max: latest.max_ppm,
          avg: latest.avg_ppm,
          over: t.action == null ? 0 : latest.values_ppm.filter((x) => x != null && x >= t.action!).length,
          action: t.action,
          ageDays: t.ageDays,
          stale: t.stale,
          source: "CDPH",
          url: t.fact.source.url ?? "https://www.cdph.ca.gov/",
          lat: t.site.lat ?? null,
          lon: t.site.lon ?? null,
        }
      : null;
  return { ok: true, now: opts.now, fixture: opts.fixture, fishery: f.id, season, whale, test, statusError: m.statusError, testsError: m.testsError };
}

/** The sea window from Today's model: every period with NWS's own sentence and only the numbers it states. */
export function seaFrom(m: TodayModel, now: string): CoastSea {
  const { lanes, ...rest } = m.sea;
  return { ok: true, now, sea: { ...rest, lanes: lanes.map((p) => ({ name: p.name, date: p.date, night: p.night, text: p.text, wind: windWords(p), seasLo: p.seasLo, seasHi: p.seasHi })) } };
}

/** Ports inside the Coast Guard's regulated bar areas (33 CFR 165.1196): Humboldt Bay, Noyo River, Crescent City, Morro Bay. */
export const REGULATED_BARS: Record<string, { name: string; guide: string | null }> = {
  "port-201": { name: "Crescent City bar", guide: "https://www.weather.gov/media/eka/northcoasthazards.pdf" },
  "port-220": { name: "Humboldt Bay bar", guide: "https://www.weather.gov/media/eka/northcoasthazards.pdf" },
  "port-223": { name: "Noyo River bar", guide: "https://www.weather.gov/media/eka/northcoasthazards.pdf" },
  "port-606": { name: "Morro Bay bar", guide: null },
};
export const BAR_RULE_URL = "https://www.law.cornell.edu/cfr/text/33/165.1196";

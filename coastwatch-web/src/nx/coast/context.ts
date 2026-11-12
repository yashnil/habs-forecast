import { stableFormat } from "@/lib/time";
import type { SeasonGate } from "./brief";
import type { CoastFishery } from "./fisheries";

/**
 * What the trip workspace's context bar carries besides the port and fishery (v13): that every figure is
 * hypothetical, whether an official record says this trip cannot be made now, and the point and date the
 * reader brought from Explore. Pure.
 */

/** Said wherever a worksheet figure is shown. */
export const HYPOTHETICAL = "Hypothetical, from your numbers.";

export type TripFlag = {
  level: "delayed" | "closed";
  /** an ordinary closed season on the calendar, not an action taken this year */
  seasonal: boolean;
  /** a person reviewed the record; otherwise it is the agency's page as Neritic transcribed it */
  verified: boolean;
  source: string;
  url: string | null;
  checked: string | null;
  /** the short line for the context bar */
  short: string;
  /** the sentence the hero carries */
  words: string;
};

/**
 * The season record says this fishery is delayed or closed: the figures are then unmistakably hypothetical.
 * The record is quoted as what it is (the agency's, verified by a person or not), never as Neritic's ruling,
 * and a record in conflict with another source is not relied on here (the gates say the sources differ).
 */
export function tripFlag(season: SeasonGate | null, fishery: CoastFishery | null): TripFlag | null {
  if (!season || season.conflict || (season.level !== "delayed" && season.level !== "closed")) return null;
  const verified = season.verified === true;
  const what = (fishery?.label ?? "this fishery").replace(/, recreational$/, "");
  const state = season.level === "delayed" ? "delayed" : season.seasonal ? "closed for the season" : "closed";
  const record = verified ? `${season.source}'s record says` : `${season.source}'s record, which nobody at Neritic has verified, says`;
  return {
    level: season.level,
    seasonal: season.seasonal,
    verified,
    source: season.source,
    url: season.url,
    checked: season.checked,
    short: `Season ${state} · ${season.source}${verified ? "" : " · not verified"}`,
    words: `${record} the ${what.charAt(0).toLowerCase()}${what.slice(1)} season here is ${state}. These figures are hypothetical: they do not describe a trip that can be made now. Check ${season.source} before you plan around them.`,
  };
}

// ---------------------------------------------------------------- carried from Explore

/** `/coast?place=<port>&at=<lat>,<lon>&date=<iso>`: the point and date the reader was looking at. */
export type Carried = { at: { lat: number; lon: number } | null; date: string | null };

type Query = { get(name: string): string | null } | Record<string, string | string[] | undefined>;
const param = (q: Query, name: string): string | null => {
  if (typeof (q as { get?: unknown }).get === "function") return (q as { get(n: string): string | null }).get(name);
  const v = (q as Record<string, string | string[] | undefined>)[name];
  return typeof v === "string" ? v : null;
};

/** Read and check the carried point and date; anything malformed is dropped, never guessed at. */
export function readCarried(q: Query): Carried {
  let at: Carried["at"] = null;
  const m = /^\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$/.exec(param(q, "at") ?? "");
  if (m) {
    const lat = Number(m[1]);
    const lon = Number(m[2]);
    if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) at = { lat, lon };
  }
  const raw = param(q, "date") ?? "";
  const date = /^\d{4}-\d{2}-\d{2}(T[\d:.]+(Z|[+-]\d{2}:\d{2})?)?$/.test(raw) && !Number.isNaN(Date.parse(raw)) ? raw : null;
  return { at, date };
}

const dm = (v: number) => {
  const a = Math.abs(v);
  const deg = Math.floor(a);
  return `${deg}°${((a - deg) * 60).toFixed(1)}′`;
};

/**
 * The carried point and date as the context bar says them: "Point from Explore: 36°48.0′ N, 121°54.0′ W",
 * "Date from Explore: Nov 14". They are where and when the reader was looking, nothing more: the worksheet
 * does not use them, and they are never a forecast for the trip.
 */
export function carriedWords(c: Carried): { point: string | null; date: string | null } {
  const point = c.at ? `Point from Explore: ${dm(c.at.lat)} ${c.at.lat >= 0 ? "N" : "S"}, ${dm(c.at.lon)} ${c.at.lon >= 0 ? "E" : "W"}` : null;
  let date: string | null = null;
  if (c.date) {
    const dayOnly = /^\d{4}-\d{2}-\d{2}$/.test(c.date);
    const d = new Date(dayOnly ? `${c.date}T12:00:00Z` : c.date);
    const f = new Intl.DateTimeFormat("en-US", dayOnly ? { month: "short", day: "numeric", timeZone: "UTC" } : { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles", timeZoneName: "short" });
    date = `Date from Explore: ${stableFormat(f, d)}`;
  }
  return { point, date };
}

/** The carried values back as query parameters, so a link onward (or a reload) keeps them. */
export function carriedQuery(c: Carried): Record<string, string> {
  return { ...(c.at ? { at: `${c.at.lat},${c.at.lon}` } : {}), ...(c.date ? { date: c.date } : {}) };
}

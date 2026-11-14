import { coastHref, parseAudience, type Audience } from "@/lib/coast";
import { LEGACY_HOME_KEY, dropLegacy, readPortRecord, updatePortRecord } from "@/lib/portRecord";

/**
 * The reader's home port, on this device only (the saved-port record, `lib/portRecord`), never assumed
 * to be where anyone fishes. Kept with {for, species} so the home briefing reopens the reader's own trip.
 */
export type HomePort = { place: string; for: Audience; species: string };

export function parseHome(raw: string | null | undefined): HomePort | null {
  if (!raw) return null;
  if (!raw.startsWith("{")) return /^[\w.,-]+$/.test(raw) ? { place: raw, for: "all", species: "any" } : null;
  try {
    const o = JSON.parse(raw) as Partial<HomePort>;
    if (typeof o.place !== "string" || !o.place) return null;
    return { place: o.place, for: parseAudience(o.for), species: typeof o.species === "string" && o.species ? o.species : "any" };
  } catch {
    return null;
  }
}

export function readHome(): HomePort | null {
  const r = readPortRecord();
  return r.place ? { place: r.place, for: parseAudience(r.for), species: r.species || "any" } : null;
}

export function writeHome(h: HomePort | null) {
  updatePortRecord((r) => (h ? { ...r, place: h.place, for: h.for, species: h.species } : { ...r, place: null, for: "all", species: "any" }));
  if (!h) dropLegacy(LEGACY_HOME_KEY);
}

/** The home briefing's link: the place with the reader's own activity and species. */
export const homeHref = (h: HomePort) => coastHref({ place: h.place, for: h.for, species: h.species });

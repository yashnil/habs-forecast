import { coastHref, parseAudience, type Audience } from "@/lib/coast";

/**
 * The reader's home port, on this device only (localStorage "cw-home-port"), never assumed to be where
 * anyone fishes. Stored as {place, for, species} so the home briefing reopens the reader's own trip;
 * an older plain place key ("port-550") is still read, as that place with no trip.
 */
export const HOME_KEY = "cw-home-port";
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
  try {
    return parseHome(window.localStorage.getItem(HOME_KEY));
  } catch {
    return null;
  }
}

export function writeHome(h: HomePort | null) {
  try {
    if (h) window.localStorage.setItem(HOME_KEY, JSON.stringify(h));
    else window.localStorage.removeItem(HOME_KEY);
  } catch {
    // not remembered on this device
  }
}

/** The home briefing's link: the place with the reader's own activity and species. */
export const homeHref = (h: HomePort) => coastHref({ place: h.place, for: h.for, species: h.species });

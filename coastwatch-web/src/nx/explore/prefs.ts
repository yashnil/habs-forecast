/**
 * Explore's choices kept on this device (v15.1 round 2, A): the base (the plain chart or a data layer), the field's
 * opacity, and which chart elements and overlays are on. Versioned and read defensively: storage that is missing,
 * blocked or holds something else gives the defaults, never an error. An address's own parameters win over these.
 */
export const PREFS_KEY = "nx-explore-prefs-v1";

export type ExplorePrefs = {
  v: 1;
  /** "chart": no data layer, the bathymetric chart alone; otherwise a product id */
  base: string;
  /** the data layer's opacity, 0.2..1 */
  opacity: number;
  depth: boolean;
  regulation: boolean;
  currents: boolean | null;
  stations: boolean;
};

export const DEFAULT_PREFS: ExplorePrefs = { v: 1, base: "chl", opacity: 1, depth: true, regulation: false, currents: null, stations: false };

export const clampOpacity = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? Math.min(1, Math.max(0.2, x)) : 1);

/** Parse a stored value; anything that is not a version-1 record gives the defaults. */
export function parsePrefs(raw: string | null | undefined): ExplorePrefs {
  if (!raw) return DEFAULT_PREFS;
  try {
    const o = JSON.parse(raw) as Partial<ExplorePrefs>;
    if (!o || o.v !== 1) return DEFAULT_PREFS;
    return {
      v: 1,
      base: typeof o.base === "string" && /^[a-z]{2,12}$/.test(o.base) ? o.base : DEFAULT_PREFS.base,
      opacity: clampOpacity(o.opacity),
      depth: typeof o.depth === "boolean" ? o.depth : DEFAULT_PREFS.depth,
      regulation: typeof o.regulation === "boolean" ? o.regulation : DEFAULT_PREFS.regulation,
      currents: typeof o.currents === "boolean" ? o.currents : null,
      stations: typeof o.stations === "boolean" ? o.stations : DEFAULT_PREFS.stations,
    };
  } catch {
    return DEFAULT_PREFS;
  }
}

export function readPrefs(): ExplorePrefs {
  try {
    return parsePrefs(window.localStorage.getItem(PREFS_KEY));
  } catch {
    return DEFAULT_PREFS;
  }
}

export function writePrefs(p: Partial<ExplorePrefs>) {
  try {
    const next = { ...readPrefs(), ...p, v: 1 as const };
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(next));
  } catch {
    // this visit only
  }
}

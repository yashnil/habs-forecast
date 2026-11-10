/**
 * The reader's saved port: one record on this device (localStorage "nx-port-v1"), read and written by
 * My Coast, Today and the account page. It stays in the browser. Only signed-in accounts send anything
 * on, through the account's own sync, and that sync can never see a point or a harvest site (see
 * `pilot/account/local.ts`).
 *
 * Two place fields, because the pages mean two different things and this record does not change either:
 *   `place`     the home the reader chose on My Coast: a port, or a harvest site or point (`site-12`,
 *               `point-36.8,-121.9`), so never assume it is a port;
 *   `lastPort`  the port last opened on Today, always `port-N`: Today is port-only.
 *
 * The first read migrates `cw-home-port`, `nx-today-place-v1`, `nx-today-fisheries-v1` and the account's
 * fisheries, home and vessel length. It runs only while the record is absent or unreadable, writes only
 * when there is something to keep (a first visit must not gain a key), and leaves the old keys in place
 * so that an earlier build still finds them after a rollback. No imports: the top bar of every page
 * reads this module, and values are only shape-checked here, then validated by the reader that knows them.
 */
export const PORT_KEY = "nx-port-v1";
export const LEGACY_HOME_KEY = "cw-home-port";
export const LEGACY_TODAY_PLACE_KEY = "nx-today-place-v1";
export const LEGACY_TODAY_FISHERIES_KEY = "nx-today-fisheries-v1";
const LEGACY_ACCOUNT_KEY = "nrt-pilot-account-v1";

export type PortRecord = {
  v: 1;
  /** My Coast's home: a port, site or point key */
  place: string | null;
  /** My Coast's trip for that home (validated by `parseAudience` where it is read) */
  for: string;
  species: string;
  /** the port last opened on Today: always `port-N` */
  lastPort: string | null;
  fisheries: string[];
  /** the account's vessel length class */
  vessel: string | null;
};

export const EMPTY_PORT: PortRecord = { v: 1, place: null, for: "all", species: "any", lastPort: null, fisheries: [], vessel: null };

const PLACE = /^[\w.,:-]{1,96}$/;
const PORT = /^port-\d{1,4}$/;
export const isPortKey = (p: string | null | undefined): p is string => typeof p === "string" && PORT.test(p);

const text = (v: unknown, max: number): string | null => (typeof v === "string" && v !== "" && v.length <= max ? v : null);
const place = (v: unknown): string | null => (typeof v === "string" && PLACE.test(v) ? v : null);
const lastPort = (v: unknown): string | null => (typeof v === "string" && PORT.test(v) ? v : null);
const vessel = (v: unknown): string | null => (typeof v === "string" && /^[a-z0-9_]{1,24}$/.test(v) ? v : null);
function fisheries(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return [...new Set(v.filter((x): x is string => typeof x === "string" && /^[a-z_]{1,40}$/.test(x)))].slice(0, 12);
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

function clean(o: Record<string, unknown>): PortRecord {
  return {
    v: 1,
    place: place(o.place),
    for: text(o.for, 20) ?? "all",
    species: text(o.species, 40) ?? "any",
    lastPort: lastPort(o.lastPort),
    fisheries: fisheries(o.fisheries),
    vessel: vessel(o.vessel),
  };
}

const hasData = (r: PortRecord) => r.place !== null || r.lastPort !== null || r.fisheries.length > 0 || r.vessel !== null;

/** A value the device refused (private window, full storage): good for this page view, as in `pilot/account/storage`. */
let memory: string | null = null;
export const resetPortMemory = () => {
  memory = null;
};

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function get(s: Storage | null, k: string): string | null {
  try {
    return s?.getItem(k) ?? null;
  } catch {
    return null;
  }
}

function parse(raw: string | null): unknown {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/** `cw-home-port`: a plain key ("port-550"), or {place, for, species}. */
function legacyHome(raw: string | null): Pick<PortRecord, "place" | "for" | "species"> {
  const none = { place: null, for: "all", species: "any" };
  if (!raw) return none;
  if (raw === "null" || raw === "undefined") return none;
  if (!raw.trimStart().startsWith("{")) return { ...none, place: place(raw) };
  const o = parse(raw);
  if (!isObj(o)) return none;
  const p = place(o.place);
  return p ? { place: p, for: text(o.for, 20) ?? "all", species: text(o.species, 40) ?? "any" } : none;
}

/** What the older keys hold, whatever state they are in. */
export function fromLegacy(s: Storage | null): PortRecord {
  const account = parse(get(s, LEGACY_ACCOUNT_KEY));
  const acc = isObj(account) ? account : {};
  const home = legacyHome(get(s, LEGACY_HOME_KEY));
  const today = fisheries(get(s, LEGACY_TODAY_FISHERIES_KEY)?.split(",").map((x) => x.trim()) ?? []);
  return {
    v: 1,
    // the home chosen on My Coast names a trip; the account's is only a port
    place: home.place ?? place(acc.home_place_id),
    for: home.for,
    species: home.species,
    lastPort: lastPort(get(s, LEGACY_TODAY_PLACE_KEY)),
    // Today's list is the one the reader sees the app use
    fisheries: today.length ? today : fisheries(acc.fisheries),
    vessel: vessel(acc.vessel_length_class),
  };
}

function current(): { record: PortRecord; newer: boolean } {
  const s = storage();
  const stored = parse(memory ?? get(s, PORT_KEY));
  if (isObj(stored) && typeof stored.v === "number") {
    if (stored.v > 1) return { record: EMPTY_PORT, newer: true }; // written by a newer build: neither read nor replaced
    if (stored.v === 1) return { record: clean(stored), newer: false };
  }
  const migrated = fromLegacy(s);
  if (!hasData(migrated)) return { record: EMPTY_PORT, newer: false };
  try {
    s?.setItem(PORT_KEY, JSON.stringify(migrated));
  } catch {
    // read from the old keys again next time
  }
  return { record: migrated, newer: false };
}

export function readPortRecord(): PortRecord {
  return current().record;
}

/** Change the record. True when it is kept on the device; false when it lasts only for this page view. */
export function updatePortRecord(fn: (r: PortRecord) => PortRecord): boolean {
  if (typeof window === "undefined") return false;
  const { record, newer } = current();
  if (newer) return false;
  const raw = JSON.stringify(clean(fn(record) as unknown as Record<string, unknown>));
  try {
    const s = storage();
    if (!s) throw new Error("no storage");
    s.setItem(PORT_KEY, raw);
    memory = null;
    return true;
  } catch {
    memory = raw;
    return false;
  }
}

/** Remove an older key's copy once its value has been cleared, so a cleared value is cleared everywhere. */
export function dropLegacy(...keys: string[]) {
  const s = storage();
  for (const k of keys) {
    try {
      s?.removeItem(k);
    } catch {
      // nothing kept
    }
  }
}

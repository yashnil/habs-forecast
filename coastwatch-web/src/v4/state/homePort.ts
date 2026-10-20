/**
 * The reader's home port, as My Coast keeps it on this device (localStorage "cw-home-port"): a place
 * key ("port-550"), or an object whose `place` is one ({ place, for, species }). Read tolerantly: any
 * other value, or no storage, is no home port. Never written here (My Coast owns it).
 */
export const HOME_KEY = "cw-home-port";

export function readHomePlace(): string | null {
  try {
    const raw = window.localStorage.getItem(HOME_KEY);
    if (!raw) return null;
    if (raw.trimStart().startsWith("{")) {
      const o = JSON.parse(raw) as { place?: unknown } | null;
      return typeof o?.place === "string" && o.place ? o.place : null;
    }
    return raw;
  } catch {
    return null;
  }
}

/** The home port's CDFW port code, when the home place is a harbour. */
export function readHomePort(): number | null {
  const k = readHomePlace();
  const code = k?.startsWith("port-") ? Number(k.slice(5)) : NaN;
  return Number.isFinite(code) ? code : null;
}

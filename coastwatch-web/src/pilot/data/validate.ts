/** Small runtime guards shared by the pilot sidecar loaders (no dependency, never throws). */
export type Obj = Record<string, unknown>;

export const isObj = (x: unknown): x is Obj => !!x && typeof x === "object" && !Array.isArray(x);
export const isStr = (x: unknown): x is string => typeof x === "string" && x.length > 0;
export const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
export const isDay = (x: unknown): x is string => typeof x === "string" && /^\d{4}-\d{2}-\d{2}$/.test(x);
export const isIso = (x: unknown): x is string => typeof x === "string" && /^\d{4}-\d{2}-\d{2}(T[\d:.]+Z?)?/.test(x);
export const strOrNull = (x: unknown): x is string | null => x === null || isStr(x);
export const numOrNull = (x: unknown): x is number | null => x === null || isNum(x);
export const dayOrNull = (x: unknown): x is string | null => x === null || isDay(x);
export const strList = (x: unknown): x is string[] => Array.isArray(x) && x.every(isStr);
export const oneOf = <T extends string>(x: unknown, values: readonly T[]): x is T => typeof x === "string" && (values as readonly string[]).includes(x);

/**
 * Fetch `${base}/${rel}` once per URL and parse it; a missing, unreadable or invalid file is null.
 * Offline (navigator.onLine === false) nothing is requested: the product is simply not offered.
 */
export function cachedLoader<T>(rel: string, parse: (j: unknown) => T | null) {
  const cache = new Map<string, Promise<T | null>>();
  return (baseUrl: string, fetcher: typeof fetch = fetch): Promise<T | null> => {
    if (typeof navigator !== "undefined" && navigator.onLine === false) return Promise.resolve(null);
    const url = `${baseUrl.replace(/\/$/, "")}/${rel}`;
    let p = cache.get(url);
    if (!p) {
      p = Promise.resolve()
        .then(() => fetcher(url))
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => (j == null ? null : parse(j)))
        .catch(() => null);
      cache.set(url, p);
    }
    return p;
  };
}

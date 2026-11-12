/**
 * A browser's storage for unit tests (vitest runs in node): localStorage and sessionStorage with `length` and
 * `key(i)` as the real ones have, Cache Storage as a map of maps, and an address. `mode` makes storage refuse
 * writes ("full") or throw on access ("blocked"), as private windows and full disks do.
 */
export type Mode = "ok" | "full" | "blocked";

export function fakeStorage(seed: Record<string, string> = {}, mode: () => Mode = () => "ok") {
  const m = new Map(Object.entries(seed));
  const s = {
    get length() {
      return m.size;
    },
    key: (i: number) => [...m.keys()][i] ?? null,
    getItem: (k: string) => {
      if (mode() === "blocked") throw new Error("SecurityError");
      return m.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (mode() !== "ok") throw new Error("QuotaExceededError");
      m.set(k, String(v));
    },
    removeItem: (k: string) => void m.delete(k),
    clear: () => m.clear(),
  };
  return { map: m, storage: s as unknown as Storage };
}

export function fakeCaches(seed: Record<string, string[]> = {}) {
  const m = new Map(Object.entries(seed).map(([k, v]) => [k, new Set(v)]));
  const caches = {
    keys: async () => [...m.keys()],
    delete: async (name: string) => m.delete(name),
    has: async (name: string) => m.has(name),
    open: async (name: string) => {
      if (!m.has(name)) m.set(name, new Set());
      return { keys: async () => [...m.get(name)!].map((url) => ({ url })) };
    },
  };
  return { map: m, caches: caches as unknown as CacheStorage };
}

export type FakeDevice = ReturnType<typeof device>;

/** Install a fake `window`; call `gone()` (or `afterEach(uninstall)`) to remove it. */
export function device(o: { local?: Record<string, string>; session?: Record<string, string>; caches?: Record<string, string[]>; search?: string } = {}) {
  let mode: Mode = "ok";
  const local = fakeStorage(o.local, () => mode);
  const session = fakeStorage(o.session, () => mode);
  const caches = fakeCaches(o.caches);
  const location = { search: o.search ?? "" };
  (globalThis as unknown as { window: unknown }).window = {
    get localStorage() {
      if (mode === "blocked") throw new Error("SecurityError");
      return local.storage;
    },
    get sessionStorage() {
      if (mode === "blocked") throw new Error("SecurityError");
      return session.storage;
    },
    caches: caches.caches,
    location,
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent() {
      return true;
    },
  };
  return {
    local: local.map,
    session: session.map,
    caches: caches.map,
    location,
    setMode: (m: Mode) => {
      mode = m;
    },
    json: (k: string) => JSON.parse(local.map.get(k) ?? "null"),
  };
}

export function uninstall() {
  delete (globalThis as unknown as { window?: unknown }).window;
}

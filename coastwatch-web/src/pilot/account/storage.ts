/**
 * On-device storage for the pilot's account features. Every read and write tolerates blocked,
 * full or cleared storage (private windows, Safari ITP, quota): it then falls back to memory for
 * this page view and says so (`persistent() === false`), so the page can tell the person their
 * choices will not survive a reload. Nothing here is sent anywhere.
 */
type Listener = () => void;

const memory = new Map<string, string>();
const listeners = new Set<Listener>();

function local(): Storage | null {
  try {
    if (typeof window === "undefined") return null;
    const s = window.localStorage;
    const probe = "nrt-probe";
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

/** true when choices are kept on this device across visits */
export function persistent(): boolean {
  return local() !== null;
}

export function readJSON<T>(key: string, fallback: T, valid: (v: unknown) => v is T): T {
  let raw: string | null = null;
  // a value kept in memory is this page's newest write that the device refused
  raw = memory.get(key) ?? null;
  if (raw == null) {
    try {
      raw = local()?.getItem(key) ?? null;
    } catch {
      raw = null;
    }
  }
  if (raw == null) return fallback;
  try {
    const v: unknown = JSON.parse(raw);
    return valid(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

/** Returns true when written to the device, false when only kept in memory for this page view. */
export function writeJSON(key: string, value: unknown): boolean {
  const raw = JSON.stringify(value);
  let ok = false;
  try {
    const s = local();
    if (s) {
      s.setItem(key, raw);
      ok = true;
    }
  } catch {
    ok = false;
  }
  if (ok) memory.delete(key);
  else memory.set(key, raw);
  emit();
  return ok;
}

export function removeKeys(keys: string[]): void {
  for (const k of keys) {
    memory.delete(k);
    try {
      local()?.removeItem(k);
    } catch {
      // nothing kept
    }
  }
  emit();
}

let version = 0;
function emit() {
  version++;
  for (const l of listeners) l();
}

/** for useSyncExternalStore: re-render when any pilot key changes (this tab, or another tab) */
export function subscribe(l: Listener): () => void {
  listeners.add(l);
  const onStorage = (e: StorageEvent) => {
    if (e.key == null || e.key.startsWith("nrt-pilot-")) emit();
  };
  if (typeof window !== "undefined") window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(l);
    if (typeof window !== "undefined") window.removeEventListener("storage", onStorage);
  };
}

export function storeVersion(): number {
  return version;
}

/** test hook: forget the in-memory fallback */
export function __resetMemory(): void {
  memory.clear();
  emit();
}

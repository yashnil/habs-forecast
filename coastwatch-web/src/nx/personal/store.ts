/**
 * This device's storage, read and written tolerantly: a private window, blocked site data or a full disk
 * never throws here. A write the device refused is kept in memory for this page view (`persisted` false),
 * so the page can say the choice will not survive a reload. Nothing here is sent anywhere.
 *
 * One change event for every key written through this module (this tab) and for storage events (other tabs),
 * so `useSyncExternalStore` readers stay in step.
 */
type Listener = () => void;

const memory = new Map<string, string>();
const listeners = new Set<Listener>();
let version = 0;

function area(kind: "local" | "session"): Storage | null {
  try {
    if (typeof window === "undefined") return null;
    return kind === "local" ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
}

const mem = (kind: "local" | "session", key: string) => `${kind}:${key}`;

export function readRaw(key: string, kind: "local" | "session" = "local"): string | null {
  const held = memory.get(mem(kind, key));
  if (held !== undefined) return held;
  try {
    return area(kind)?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

export function readJSON<T>(key: string, fallback: T, valid: (v: unknown) => v is T, kind: "local" | "session" = "local"): T {
  const raw = readRaw(key, kind);
  if (raw == null) return fallback;
  try {
    const v: unknown = JSON.parse(raw);
    return valid(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

/** True when the device kept it; false when it lasts only for this page view. */
export function writeJSON(key: string, value: unknown, kind: "local" | "session" = "local"): boolean {
  const raw = JSON.stringify(value);
  let ok = false;
  try {
    const s = area(kind);
    if (s) {
      s.setItem(key, raw);
      ok = true;
    }
  } catch {
    ok = false;
  }
  if (ok) memory.delete(mem(kind, key));
  else memory.set(mem(kind, key), raw);
  emit();
  return ok;
}

export function remove(key: string, kind: "local" | "session" = "local"): void {
  memory.delete(mem(kind, key));
  try {
    area(kind)?.removeItem(key);
  } catch {
    // nothing kept
  }
  emit();
}

/** Does this browser keep anything across visits? */
export function persistent(): boolean {
  try {
    const s = area("local");
    if (!s) return false;
    s.setItem("nx-today-probe", "1");
    s.removeItem("nx-today-probe");
    return true;
  } catch {
    return false;
  }
}

export function emit() {
  version++;
  for (const l of listeners) l();
}

export function subscribe(l: Listener): () => void {
  listeners.add(l);
  const onStorage = () => {
    version++;
    l();
  };
  if (typeof window !== "undefined") window.addEventListener?.("storage", onStorage);
  return () => {
    listeners.delete(l);
    if (typeof window !== "undefined") window.removeEventListener?.("storage", onStorage);
  };
}

export const storeVersion = () => version;

/** Forget the in-memory fallback (forget.ts, and tests). */
export function resetMemory() {
  memory.clear();
  emit();
}

import { resetPortMemory } from "@/lib/portRecord";
import { entryFor, isForeign, keepsOnForget, KIND_ORDER, type KeyEntry, type KeyKind } from "./keys";
import { emit, resetMemory } from "./store";

/**
 * "Delete everything on this device", and the inventory and export that show what there is to delete.
 *
 * Deleting is not driven by the registry: it removes EVERY key this origin holds in localStorage and
 * sessionStorage and every cache, so a key nobody registered is still deleted. Two things stay, both said
 * on the page: a signed-in account's session (`sb-…`; signing out ends it) and the app's own code cache
 * (`neritic-static-…`, nothing about the reader). The registry (keys.ts) names what each key holds.
 */
export type Device = {
  local: Storage | null;
  session: Storage | null;
  caches: Pick<CacheStorage, "keys" | "delete" | "open"> | null;
};

export function thisDevice(): Device {
  const pick = <T>(f: () => T): T | null => {
    try {
      return typeof window === "undefined" ? null : (f() ?? null);
    } catch {
      return null;
    }
  };
  return { local: pick(() => window.localStorage), session: pick(() => window.sessionStorage), caches: pick(() => window.caches) };
}

function names(s: Storage | null): string[] {
  if (!s) return [];
  const out: string[] = [];
  try {
    for (let i = 0; i < s.length; i++) {
      const k = s.key(i);
      if (k != null) out.push(k);
    }
  } catch {
    // blocked: nothing readable, nothing kept
  }
  return out;
}

async function cacheNames(d: Device): Promise<string[]> {
  try {
    return (await d.caches?.keys()) ?? [];
  } catch {
    return [];
  }
}

export type Forgotten = { local: string[]; session: string[]; caches: string[]; /** what could not be removed, by name */ failed: string[] };

/** Remove everything Neritic keeps in this browser. Safe to call twice; never throws. */
export async function forgetEverything(d: Device = thisDevice()): Promise<Forgotten> {
  const out: Forgotten = { local: [], session: [], caches: [], failed: [] };
  for (const [s, list] of [
    [d.local, out.local],
    [d.session, out.session],
  ] as const) {
    for (const k of names(s)) {
      if (isForeign(k)) continue;
      try {
        s!.removeItem(k);
        list.push(k);
      } catch {
        out.failed.push(k);
      }
    }
  }
  for (const name of await cacheNames(d)) {
    if (keepsOnForget(name)) continue;
    try {
      if (await d.caches!.delete(name)) out.caches.push(name);
    } catch {
      out.failed.push(name);
    }
  }
  // values a blocked browser was holding in memory for this page view
  resetPortMemory();
  resetMemory();
  emit();
  return out;
}

/** The synchronous half (storage only), for callers that cannot wait; the caches follow through `forgetEverything`. */
export function forgetStorage(d: Device = thisDevice()): string[] {
  const gone: string[] = [];
  for (const s of [d.local, d.session]) {
    for (const k of names(s)) {
      if (isForeign(k)) continue;
      try {
        s!.removeItem(k);
        gone.push(k);
      } catch {
        // left for forgetEverything to report
      }
    }
  }
  resetPortMemory();
  resetMemory();
  return gone;
}

// ------------------------------------------------------------------ inventory

export type InventoryItem = {
  name: string;
  store: KeyEntry["store"];
  kind: KeyKind;
  holds: string;
  /** bytes for a storage value; files for a cache */
  size: number;
  unit: "bytes" | "files";
  /** not in the registry: still deleted, listed as "Other data Neritic kept" */
  unregistered: boolean;
};

const UNKNOWN = "Other data Neritic kept";

/** What is on this device now, most personal first. */
export async function inventory(d: Device = thisDevice()): Promise<InventoryItem[]> {
  const out: InventoryItem[] = [];
  for (const [store, s] of [
    ["local", d.local],
    ["session", d.session],
  ] as const) {
    for (const k of names(s)) {
      if (isForeign(k)) continue;
      const e = entryFor(k, store);
      let size = 0;
      try {
        size = s!.getItem(k)?.length ?? 0;
      } catch {
        size = 0;
      }
      out.push({ name: k, store, kind: e?.kind ?? "personal", holds: e?.holds ?? UNKNOWN, size, unit: "bytes", unregistered: !e });
    }
  }
  for (const name of await cacheNames(d)) {
    const e = entryFor(name, "cache");
    let files = 0;
    try {
      files = (await (await d.caches!.open(name)).keys()).length;
    } catch {
      files = 0;
    }
    out.push({ name, store: "cache", kind: e?.kind ?? "offline", holds: e?.holds ?? UNKNOWN, size: files, unit: "files", unregistered: !e });
  }
  return out.sort((a, b) => KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) || a.name.localeCompare(b.name));
}

// ------------------------------------------------------------------ export

export type LocalExport = {
  app: "neritic-device";
  schema: 1;
  exportedAt: string;
  /** every stored value, parsed when it is JSON, with what it holds */
  local: Array<{ key: string; holds: string; value: unknown }>;
  /** saved-for-offline caches: named and counted, their files are not copied into the export */
  caches: Array<{ name: string; holds: string; files: number }>;
};

const parsed = (raw: string | null): unknown => {
  if (raw == null) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
};

/**
 * A copy of what this device holds, as one JSON document the reader can keep. Housekeeping keys and the tab's
 * session are left out (they say nothing about the reader); a signed-in session is never exported.
 */
export async function exportLocalData(d: Device = thisDevice(), now: Date = new Date()): Promise<LocalExport> {
  const items = await inventory(d);
  const local = items
    .filter((i) => i.store === "local" && i.kind !== "transient")
    .map((i) => {
      let raw: string | null = null;
      try {
        raw = d.local!.getItem(i.name);
      } catch {
        raw = null;
      }
      return { key: i.name, holds: i.holds, value: parsed(raw) };
    });
  const caches = items.filter((i) => i.store === "cache" && i.kind !== "transient").map((i) => ({ name: i.name, holds: i.holds, files: i.size }));
  return { app: "neritic-device", schema: 1, exportedAt: now.toISOString(), local, caches };
}

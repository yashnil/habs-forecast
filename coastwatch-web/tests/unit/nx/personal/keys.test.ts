import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { KEYS, entryFor, isForeign, isRegistered, keepsOnForget } from "@/nx/personal/keys";

/**
 * The registry of everything Neritic keeps in a browser (src/nx/personal/keys.ts) against the source: a
 * storage key or cache name a source file uses must be registered, so the account page can name it, the
 * export can include it and nobody forgets it exists.
 */
const ROOT = path.join(__dirname, "..", "..", "..", "..");

function files(dir: string, out: string[] = []): string[] {
  for (const f of readdirSync(dir)) {
    const p = path.join(dir, f);
    if (statSync(p).isDirectory()) files(p, out);
    else if (/\.(ts|tsx|js|mjs)$/.test(f) && !/\.test\./.test(f)) out.push(p);
  }
  return out;
}

const SOURCES = [...files(path.join(ROOT, "src")), path.join(ROOT, "public", "sw.js"), path.join(ROOT, "public", "sw-push.js")];

/** How a storage key or a cache name appears in source. Each pattern's first group is the key, or its prefix up to a `${…}`. */
const PATTERNS: RegExp[] = [
  // a versioned Neritic key anywhere: "nx-port-v1", `nx-ask-seen-v1:${place}`
  /["'`]((?:nx|nrt|cw|neritic)-[a-z0-9-]*-v\d+:?)/g,
  // a literal handed straight to storage or Cache Storage
  /(?:get|set|remove)Item\(\s*["'`]([^"'`$]+)/g,
  /caches\.(?:open|delete|has)\(\s*["'`]([^"'`$]+)/g,
  // a constant named as a key, prefix, cache or namespace
  /const\s+[A-Za-z_]*(?:KEY|PREFIX|CACHE|NS|PACK|STATIC|KEPT|PROBE|MARK)[A-Za-z_]*\s*(?::[^=]+)?=\s*(?:\([^)]*\)\s*=>\s*)?["'`]((?:nx|nrt|cw|neritic)-[^"'`$]+)/gi,
  // any cache-shaped name
  /["'`](neritic-(?:pack|static|kept)[a-z0-9-]*)/g,
];

function used(): Map<string, string> {
  const found = new Map<string, string>();
  for (const f of SOURCES) {
    // the registry itself lists every key; it is what is being checked
    if (f.endsWith(path.join("nx", "personal", "keys.ts"))) continue;
    const text = readFileSync(f, "utf8");
    for (const re of PATTERNS) for (const m of text.matchAll(re)) if (m[1] && !found.has(m[1])) found.set(m[1], path.relative(ROOT, f));
  }
  return found;
}

describe("the registry of what Neritic keeps in a browser", () => {
  const found = used();

  it("finds the keys the app is known to use (the scan itself works)", () => {
    for (const k of ["nx-port-v1", "nx-trip-v1:", "nx-coast-v1", "nx-today-seen-v1:", "nx-ask-seen-v1:", "nx-field-v1", "neritic-pack-field", "neritic-pack-today", "nrt-pilot-account-v1", "nx-follow-v1", "nx-inbox-v1", "cw-glide-"]) {
      expect([...found.keys()], k).toContain(k);
    }
    expect(found.size).toBeGreaterThan(40);
  });

  it("every storage key and cache name in the source is registered", () => {
    const missing = [...found].filter(([k]) => !isRegistered(k) && !isForeign(k)).map(([k, f]) => `${k}  (${f})`);
    expect(missing, `register these in src/nx/personal/keys.ts, saying what each holds:\n${missing.join("\n")}`).toEqual([]);
  });

  it("every registered key is still used somewhere (no stale entries), apart from keys only older builds wrote", () => {
    const names = [...found.keys()];
    const stale = KEYS.filter((e) => !e.legacy && !names.some((n) => n === e.key || (e.match === "prefix" && n.startsWith(e.key)) || e.key.startsWith(n))).map((e) => e.key);
    expect(stale).toEqual([]);
  });

  it("no key is registered twice, and every entry says what it holds in words", () => {
    const seen = new Set<string>();
    for (const e of KEYS) {
      const id = `${e.store}:${e.key}`;
      expect(seen.has(id), id).toBe(false);
      seen.add(id);
      expect(e.holds.length, e.key).toBeGreaterThan(12);
      expect(e.holds, e.key).not.toMatch(/localStorage|JSON|\bkey\b/i);
    }
  });

  it("matches exact keys exactly and prefixed keys by their prefix, the longest first", () => {
    expect(entryFor("nx-port-v1")?.kind).toBe("personal");
    expect(entryFor("nx-port-v10")).toBeNull();
    expect(entryFor("nx-today-seen-v1:port-592")?.kind).toBe("history");
    expect(entryFor("nx-trip-v1:scenarios")?.holds).toMatch(/trips/);
    expect(entryFor("neritic-pack-field", "cache")?.kind).toBe("offline");
    expect(entryFor("neritic-pack-saved-ca", "local")?.key).toBe("neritic-pack-saved-");
    expect(entryFor("someone-elses-key")).toBeNull();
  });

  it("the trip worksheet's and My Coast's keys are personal: they are exported and deleted with everything else", () => {
    for (const k of ["nx-trip-v1:scenarios", "nx-trip-v1:vessels", "nx-trip-v1:meta", "nx-trip-v1:corrupt:scenarios", "nx-coast-v1"]) expect(entryFor(k, "local")?.kind, k).toBe("personal");
  });

  it("only the app's own code cache survives a delete, and only a sign-in session is foreign", () => {
    expect(keepsOnForget("neritic-static-abc123")).toBe(true);
    expect(keepsOnForget("neritic-pack-field")).toBe(false);
    expect(keepsOnForget("neritic-pack-today")).toBe(false);
    expect(isForeign("sb-abc-auth-token")).toBe(true);
    expect(isForeign("nx-port-v1")).toBe(false);
  });
});

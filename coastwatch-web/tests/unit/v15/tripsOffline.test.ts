import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { cacheTripsPage, TRIPS_CACHE, TRIPS_META, TRIPS_PAGE } from "@/nx/offline/trips";
import { forgetEverything, onForget } from "@/nx/personal/forget";
import { entryFor } from "@/nx/personal/keys";
import { device, uninstall } from "../nx/personal/device";

/** A Cache Storage with real Response bodies (the offline cache stores and reads them). */
function cacheStorage() {
  const all = new Map<string, Map<string, Response>>();
  const key = (r: RequestInfo | URL) => new URL(typeof r === "string" ? r : r instanceof URL ? r.href : r.url, "http://x").pathname;
  const open = async (name: string) => {
    if (!all.has(name)) all.set(name, new Map());
    const m = all.get(name)!;
    return {
      match: async (r: RequestInfo) => m.get(key(r))?.clone(),
      put: async (r: RequestInfo, resp: Response) => void m.set(key(r), resp),
      keys: async () => [...m.keys()].map((u) => ({ url: `http://x${u}` })),
      delete: async (r: RequestInfo | { url: string }) => m.delete(key(typeof r === "string" ? r : (r as { url: string }).url)),
    };
  };
  return { all, caches: { open, has: async (n: string) => all.has(n), keys: async () => [...all.keys()], delete: async (n: string) => all.delete(n) } as unknown as CacheStorage };
}

const html = (build: string) =>
  `<!doctype html><html><head><link rel="stylesheet" href="/_next/static/css/${build}.css"><link rel="icon" href="/icon.png"></head><body><script src="/_next/static/chunks/app/coast/trips/page-${build}.js"></script></body></html>`;

function server(build: string, seen: string[]) {
  return async (input: RequestInfo | URL) => {
    const u = String(input);
    seen.push(u);
    if (u === TRIPS_PAGE) return new Response(html(build), { headers: { "Content-Type": "text/html; charset=utf-8" } });
    if (u.endsWith(".css")) return new Response(`@font-face{src:url(/_next/static/media/figtree-${build}.woff2)}`, { headers: { "Content-Type": "text/css" } });
    return new Response("x");
  };
}

afterEach(() => uninstall());

describe("your trips offline: the page and its code are kept", () => {
  it("keeps /coast/trips with the time it was kept, its code, fonts and the baked chart, and no data", async () => {
    const cs = cacheStorage();
    const seen: string[] = [];
    const kept = await cacheTripsPage({ fetch: server("a1", seen) as typeof fetch, caches: cs.caches, now: () => new Date("2026-10-13T06:00:00Z"), register: null });
    expect(kept).toMatchObject({ keptAt: "2026-10-13T06:00:00.000Z" });
    const c = cs.all.get(TRIPS_CACHE)!;
    expect([...c.keys()]).toEqual(expect.arrayContaining([TRIPS_PAGE, TRIPS_META, "/_next/static/css/a1.css", "/_next/static/chunks/app/coast/trips/page-a1.js", "/_next/static/media/figtree-a1.woff2", "/icon.png", "/chart/chart-day-ca.webp"]));
    expect(c.get(TRIPS_PAGE)!.headers.get("x-neritic-saved-at")).toBe("2026-10-13T06:00:00.000Z");
    // the trips are in storage: no data request is made to keep the page
    expect(seen.some((u) => u.includes("/coast/data") || u.includes("/api/"))).toBe(false);
  });

  it("a new build's page replaces the old build's code; nothing is kept offline when the page does not load", async () => {
    const cs = cacheStorage();
    await cacheTripsPage({ fetch: server("a1", []) as typeof fetch, caches: cs.caches, now: () => new Date(), register: null });
    await cacheTripsPage({ fetch: server("b2", []) as typeof fetch, caches: cs.caches, now: () => new Date(), register: null });
    const keys = [...cs.all.get(TRIPS_CACHE)!.keys()];
    expect(keys).toContain("/_next/static/css/b2.css");
    expect(keys).not.toContain("/_next/static/css/a1.css");
    const down = cacheStorage();
    expect(await cacheTripsPage({ fetch: (async () => new Response("", { status: 503 })) as typeof fetch, caches: down.caches, now: () => new Date(), register: null })).toBeNull();
    expect(down.all.get(TRIPS_CACHE)?.has(TRIPS_PAGE) ?? false).toBe(false);
  });
});

describe("the service worker answers /coast/trips from the kept copy", () => {
  const sw = readFileSync(path.resolve(__dirname, "../../../public/sw.js"), "utf8");
  it("is a new worker version and names the trips cache", () => {
    expect(sw).toMatch(/const VERSION = "neritic-sw-7"/);
    expect(sw).toContain('const TRIPS = "neritic-trips-v1"');
  });
  it("marks the kept page as a saved copy and asks for it offline, on a failed network and after a slow one", () => {
    expect(sw).toMatch(/async function tripsPage[\s\S]*?marked\(r,/);
    expect(sw.match(/tripsPage\(url\.pathname\)/g)?.length).toBeGreaterThanOrEqual(3);
  });
});

describe("Delete everything removes your trips", () => {
  it("every nx-trips-v1 key, the kept trips page, and what a module holds in memory", async () => {
    const d = device({
      local: { "nx-trips-v1:index": '{"schema":1,"ids":["t1"]}', "nx-trips-v1:trip:t1": "{}", "nx-trips-v1:snap:p1": "{}", "nx-trips-v1:corrupt:trip:t2": "{" },
      caches: { "neritic-trips-v1": ["/coast/trips"], "neritic-static-b1": ["/_next/static/chunks/a.js"] },
    });
    let reset = 0;
    const off = onForget(() => (reset += 1));
    const gone = await forgetEverything();
    off();
    expect([...d.local.keys()]).toEqual([]);
    expect(gone.caches).toContain("neritic-trips-v1");
    expect([...d.caches.keys()]).toEqual(["neritic-static-b1"]);
    expect(reset).toBe(1);
  });
  it("both are in the registry, so the inventory names them and the export carries the trips", () => {
    expect(entryFor("nx-trips-v1:trip:t1", "local")?.kind).toBe("personal");
    expect(entryFor("neritic-trips-v1", "cache")?.kind).toBe("offline");
  });
});

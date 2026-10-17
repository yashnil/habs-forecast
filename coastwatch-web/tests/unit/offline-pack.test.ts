import { afterEach, describe, expect, it, vi } from "vitest";
import type { Manifest } from "@/generated/schema";
import { currentsHourly, currentsMean } from "@/lib/currents";
import { artifactUrl, tileIndexUrl } from "@/lib/layers";
import { basemapWater, setBasemapArchive } from "@/lib/waterMask";
import { PACKS, keepsAcrossDownloads, plan, satelliteTileSets } from "@/v4/offline/packs";
import manifestJson from "../fixture-data/v1/manifest.json";

/** The offline field pack's plan holds everything Explore can show inside the box, and an offline
 *  map's water comes from the pack's own basemap, never from the tile server. Fixture data. */
const m = manifestJson as unknown as Manifest;
const BASE = "/data/fixture/v1";
const items = plan(PACKS.monterey, m, BASE, []);
const urls = new Set(items.map((i) => i.url));

describe("the pack plan", () => {
  it("saves every currents hour Explore's timebar offers, and the 24 h mean", () => {
    const hours = currentsHourly(m);
    expect(hours.length).toBeGreaterThan(1);
    for (const l of [...hours, currentsMean(m)!]) {
      expect(urls).toContain(artifactUrl(BASE, l.vectors!.u_grid.url));
      expect(urls).toContain(artifactUrl(BASE, l.vectors!.v_grid.url));
    }
  });

  it("saves each satellite view's tiles (each sensor, both together, the age of each pixel, which sensor)", () => {
    const sets = satelliteTileSets(m);
    const names = sets.map((t) => t.url_template);
    expect(names.some((u) => u.includes("olci300") && u.includes("/tiles/"))).toBe(true);
    expect(names.some((u) => u.includes("viirs750") && u.includes("/tiles/"))).toBe(true);
    expect(names.some((u) => u.includes("multisensor") && u.includes("/tiles/"))).toBe(true);
    expect(names.some((u) => u.includes("/age-tiles/"))).toBe(true);
    for (const t of sets) {
      const idx = tileIndexUrl(BASE, t);
      if (idx) expect(urls).toContain(idx);
      // the box's own tile at the deepest native zoom
      expect([...urls].some((u) => u.startsWith(artifactUrl(BASE, t.url_template.split("{z}")[0])) && u.includes(`/${t.max_native_zoom}/`))).toBe(true);
    }
  });

  it("keeps only files that cannot change under their URL when a pack is downloaded again", () => {
    expect(keepsAcrossDownloads("/_next/static/chunks/abc.js")).toBe(true);
    expect(keepsAcrossDownloads(`${BASE}/charm/2026-10-08/lead0-78fefc1fd4/particulate_domoic.u16.gz`)).toBe(true);
    expect(keepsAcrossDownloads("/api/evidence?place=port-550")).toBe(false);
    expect(keepsAcrossDownloads("/?sel=harbor:550&layer=forecast")).toBe(false);
    expect(keepsAcrossDownloads("/offline/monterey/basemap.pmtiles")).toBe(false);
    expect(keepsAcrossDownloads("/basemap/relief-sea.webp")).toBe(false);
  });
});

describe("the water that clips the data, offline", () => {
  afterEach(() => {
    setBasemapArchive(null);
    vi.unstubAllGlobals();
  });

  it("is read from the pack's basemap archive, and the tile server is never asked", async () => {
    const fetchSpy = vi.fn(() => Promise.reject(new Error("no network")));
    vi.stubGlobal("fetch", fetchSpy);
    vi.stubGlobal("OffscreenCanvas", class {});
    const asked: number[][] = [];
    // a tile the archive does not hold (outside the box): no water from it, the bundled water is used
    setBasemapArchive({ getZxy: async (z, x, y) => (asked.push([z, x, y]), undefined) });
    expect(await basemapWater(16, 10560, 25600, 256)).toBeNull();
    // above the archive's deepest zoom (14) its ancestor tile is read
    expect(asked).toEqual([[14, 2640, 6400]]);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

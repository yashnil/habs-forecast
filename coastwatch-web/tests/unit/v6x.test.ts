import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { bayAt, BAYS, clearBays } from "@/lib/bays";
import { parseDerived, type EstimateLayer } from "@/lib/derived";
import { renderGridTile, stripAt } from "@/lib/gridTiles";
import { outsideLabel, outsideWords } from "@/lib/gridTilesContext";
import { classWords, ESTIMATE_WORDS } from "@/components/map/HoverReadout";
import { rankHits, siteHits, speciesHits, type SearchHit } from "@/v4/explore/Search";
import { crossHrefs, researchVariable, tripOf } from "@/v4/explore/Inspect";
import { nearestStation, researchVar } from "@/v4/research/Research";
import type { ValueGrid } from "@/generated/schema";

/** v6, track X: bays, the estimate's class grid, the science wording and the cross-links. */

describe("bays (lib/bays)", () => {
  it("contains the bay water the outlook was drawn over, and none of the open coast next to it", () => {
    // the 34 outlook cells over SF and San Pablo Bays (audit C): lat 37.52–38.06, lon −122.40 to −122.10
    for (const [lat, lon] of [[37.8, -122.35], [37.6, -122.2], [38.05, -122.35], [37.52, -122.2], [38.06, -122.1]]) expect(bayAt(lat, lon)?.id).toBe("sf-bay");
    expect(bayAt(38.15, -122.88)?.id).toBe("tomales-bay");
    expect(bayAt(40.8, -124.15)?.id).toBe("humboldt-bay");
    expect(bayAt(35.35, -120.84)?.id).toBe("morro-bay");
    expect(bayAt(36.82, -121.75)?.id).toBe("elkhorn-slough");
    expect(bayAt(32.68, -117.15)?.id).toBe("san-diego-bay");
    // open coast: outside the Golden Gate, Half Moon Bay, Monterey Bay off Moss Landing, Point Reyes beach, off Coronado
    for (const [lat, lon] of [[37.81, -122.52], [37.75, -122.52], [37.48, -122.47], [36.81, -121.8], [38.12, -122.98], [32.68, -117.2], [40.8, -124.22]]) expect(bayAt(lat, lon)).toBeNull();
    expect(BAYS.map((b) => b.id)).toEqual(["sf-bay", "tomales-bay", "humboldt-bay", "morro-bay", "elkhorn-slough", "san-diego-bay"]);
  });

  it("clears tile pixels inside a bay only, and leaves a tile far from every bay untouched", () => {
    // zoom 10 tile over the central bay (x 164, y 395) and one off Big Sur (x 166, y 403)
    const fill = () => new Uint8ClampedArray(64 * 64 * 4).fill(255);
    const bay = fill();
    clearBays(bay, 10, 164, 395, 64);
    let cleared = 0;
    for (let i = 3; i < bay.length; i += 4) if (bay[i] === 0) cleared++;
    expect(cleared).toBeGreaterThan(0);
    const sea = fill();
    clearBays(sea, 10, 166, 403, 64);
    expect(sea.every((v) => v === 255)).toBe(true);
  });
});

describe("the estimate's class grid (lib/derived, lib/gridTiles)", () => {
  const fixture = JSON.parse(readFileSync(path.resolve(__dirname, "../fixture-derived/v1/derived/index.json"), "utf8"));
  it("reads the class grid and its legend; ignores one of another shape", () => {
    const est = parseDerived(fixture)!.layers[1] as EstimateLayer;
    expect(est.class_grid?.transform).toBe("none");
    expect(est.class_legend?.[2]).toMatch(/extrapolated toward shore/);
    const bad = structuredClone(fixture);
    bad.layers[1].class_grid.width = 3;
    expect((parseDerived(bad)!.layers[1] as EstimateLayer).class_grid).toBeNull();
    const none = structuredClone(fixture);
    delete none.layers[1].class_grid;
    expect((parseDerived(none)!.layers[1] as EstimateLayer).class_grid).toBeNull();
    // the list form of the legend is read too
    const list = structuredClone(fixture);
    list.layers[1].class_legend = [{ value: 3, label: "Bay" }];
    expect((parseDerived(list)!.layers[1] as EstimateLayer).class_legend).toEqual({ 3: "Bay" });
  });

  it("draws class 2 (the shoreline strip) with its own marks", () => {
    const g: ValueGrid = { width: 2, height: 1, lat_first: 2, lon_first: 2, lat_step: 1, lon_step: 1, scale_factor: 1, add_offset: 0, nodata: 65535, url: "x" } as ValueGrid;
    const lut = new Uint32Array(65536).fill(0xff000001);
    const fill = new Uint32Array(65536).fill(0xff000002);
    const dot = new Uint32Array(65536).fill(0xff000003);
    const sLut = new Uint32Array(65536).fill(0xff000004);
    const sDot = new Uint32Array(65536).fill(0xff000005);
    const px = new Uint32Array(renderGridTile(g, new Uint16Array([10, 10]), lut, 6, 32, 31, 512, { stipple: { lut: fill, dot }, strip: { lut: sLut, dot: sDot, cls: new Uint16Array([1, 2]) } }).buffer);
    const seen = new Set(px);
    expect(seen.has(0xff000004) && seen.has(0xff000005)).toBe(true);
    expect(seen.has(0xff000002)).toBe(true);
    expect(stripAt(3, 6)).toBe(true);
  });

  it("quotes the published class legend verbatim, never a hard-coded multiplier, and words every estimate conservatively without a class", () => {
    const est = parseDerived(fixture)!.layers[1] as EstimateLayer;
    expect(classWords(est, 2)).toBe("Fixture: extrapolated toward shore from water about 1 km out (x9.9 test multiplier), never observed; not checkable");
    expect(classWords(est, 3)).toBe("Fixture: bay or estuary, borrowed background; turbid water");
    expect(classWords({ class_legend: null }, 2)).toBe("Extrapolated toward shore from water farther out; never seen by the satellite here; not checkable");
    expect(classWords(est, null)).toBe("Estimated; near shore it may be extrapolated");
    expect(Object.values(ESTIMATE_WORDS).join(" ")).not.toMatch(/×|x\d|clear pixels/);
    expect(est.class_validation?.[2]?.rmse_log10["0-0.6 km"]).toBe(0.58);
    expect(est.class_note).toMatch(/class 0/);
  });
});

describe("science wording", () => {
  it("says NOAA publishes no probability, by variable, never that it does not model the water", () => {
    expect(outsideLabel("particulate_domoic")).toBe("NOAA publishes no toxin probability here");
    expect(outsideLabel("pseudo_nitzschia")).toBe("NOAA publishes no Pseudo-nitzschia probability here");
    expect(outsideWords()).not.toMatch(/does not (model|reach)/);
  });
});

describe("search: species and harvest sites", () => {
  const all: SearchHit[] = [{ kind: "harbor", id: "220", label: "Eureka", sub: "Harbour · Humboldt" }, ...siteHits(), ...speciesHits()];
  it("finds Clam Beach and a species by its own words", () => {
    expect(rankHits(all, "clam beach")[0]).toMatchObject({ kind: "site", label: "Clam Beach, Humboldt" });
    expect(rankHits(all, "dungeness")[0]).toMatchObject({ kind: "species", id: "dungeness" });
    expect(rankHits(all, "albacore")[0]).toMatchObject({ kind: "species", id: "tunas" });
    expect(rankHits(all, "lingcod")[0]).toMatchObject({ kind: "species", id: "groundfish" });
  });
});

describe("cross-links (Explore → My Coast, Research)", () => {
  const stations = [
    { station_id: "HABs-SantaCruzWharf", name: "Santa Cruz Wharf", lat: 36.958, lon: -122.017 },
    { station_id: "HABs-MontereyWharf", name: "Monterey Wharf", lat: 36.605, lon: -121.889 },
  ] as never;
  it("keeps the trip from the briefing that opened the map, else the reader's own", () => {
    expect(tripOf("/coast?place=port-550&for=commercial&species=dungeness&date=2026-10-14", null)).toEqual({ for: "commercial", species: "dungeness", date: "2026-10-14" });
    expect(tripOf(null, { for: "recreational", species: "salmon" })).toEqual({ for: "recreational", species: "salmon" });
    expect(tripOf(null, null)).toEqual({});
  });
  it("Brief this keeps for/species/date; Research this carries the nearest station, the variable, the day and the way back", () => {
    const h = crossHrefs({ place: "port-592", lat: 36.806, lon: -121.786, stations, variable: "pDA", day: "2026-10-14", from: "/coast?place=port-592&for=commercial&species=dungeness&date=2026-10-14", here: "/?layer=forecast&sel=harbor:592" });
    expect(h.brief).toBe("/coast?place=port-592&for=commercial&species=dungeness&date=2026-10-14");
    const r = new URL(h.research, "http://x");
    expect(r.pathname).toBe("/research");
    expect(r.searchParams.get("station")).toBe("HABs-MontereyWharf");
    expect(r.searchParams.get("place")).toBe("port-592");
    expect(r.searchParams.get("variable")).toBe("pDA");
    expect(r.searchParams.get("day")).toBe("2026-10-14");
    expect(r.searchParams.get("from")).toBe("/?layer=forecast&sel=harbor:592");
    expect(h.stationName).toBe("Monterey Wharf");
    // a station's own record: no place, no "nearest"
    expect(new URL(crossHrefs({ place: "point-36.958,-122.017", lat: 36.958, lon: -122.017, station: "HABs-SantaCruzWharf", stations, variable: "pDA" }).research, "http://x").searchParams.has("place")).toBe(false);
    expect(researchVariable("forecast", "pseudo_nitzschia")).toBe("pn_seriata");
    expect(researchVariable("sst", "particulate_domoic")).toBe("temp");
  });
  it("Research picks the station nearest the place, never a silent default", () => {
    const n = nearestStation(stations as unknown as { station_id: string; lat: number; lon: number }[], 36.806, -121.786, "port-592");
    expect(n?.station.station_id).toBe("HABs-MontereyWharf");
    expect(researchVar("particulate_domoic")).toBe("pDA");
    expect(researchVar("pseudo_nitzschia")).toBe("pn_seriata");
    expect(researchVar("nonsense")).toBeNull();
  });
});

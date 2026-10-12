import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import type { PortsCollection, ValueGrid } from "@/generated/schema";
import { APPLICABILITY, APPLICABILITY_KM, HARVEST_SITES } from "@/content/coast";
import { associationsAt, cdfwRegionsAt, distanceToPolygonKm } from "@/lib/applicability";
import { availability, coastHref, parseAudience, pointKey, recency, resolvePlace, stationsNear } from "@/lib/coast";
import type { StationLite } from "@/lib/data";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const file = (prefix: string) => JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith(prefix))!), "utf8"));
const official = file("official-") as OfficialDataset;
const ports = file("ports-") as PortsCollection;
const port = (code: number) => {
  const f = ports.features.find((x) => x.properties.port_code === code)!;
  const [lon, lat] = f.geometry.coordinates as number[];
  return { lat, lon };
};
const ids = (a: ReturnType<typeof associationsAt>, tier?: string) => a.filter((x) => !tier || x.tier === tier).map((x) => x.record.id);

describe("notice applicability (from the notices' own geography only)", () => {
  it("Crescent City: the Del Norte razor clam notices and the North Coast rock crab advisory apply by geography", () => {
    const { lat, lon } = port(201);
    const a = associationsAt(official, "port-201", lat, lon);
    expect(ids(a, "geographic")).toEqual(expect.arrayContaining(["cdph-2026-annual-mussel-quarantine", "cdfw-2026-razor-clam-del-norte", "cdph-2026-sn26-020-razor-clam-del-norte", "cdph-rock-crab-viscera-north-coast-nci"]));
    expect(ids(a, "geographic")).not.toContain("cdfw-2024-razor-clam-humboldt");
    expect(ids(a)).not.toContain("cdph-2026-sn26-019-anchovy-central-coast");
  });

  it("Monterey: inside the anchovy notices' latitudes, with the south limit 8 km away called out", () => {
    const { lat, lon } = port(550);
    const a = associationsAt(official, "port-550", lat, lon);
    const anchovy = a.find((x) => x.record.id === "cdph-2026-sn26-019-anchovy-central-coast")!;
    expect(anchovy.tier).toBe("geographic");
    expect(anchovy.caveats.join(" ")).toMatch(/south limit .* is 8\.\d km south of here/);
    expect(anchovy.caveats.join(" ")).toMatch(/does not say how far offshore/);
  });

  it("Santa Cruz: Monterey County's bivalve advisory is a separate area nearby, never applied by name", () => {
    const { lat, lon } = port(593);
    const a = associationsAt(official, "port-593", lat, lon);
    const mb = a.find((x) => x.record.id === "cdph-2026-sn26-018-monterey-bivalves")!;
    expect(mb.tier).toBe("possible");
    expect(mb.basis).toMatch(/separate area nearby/);
    // the Northern Channel Islands advisory names "Santa Cruz" (island) in its area text: never matched to Santa Cruz port
    expect(ids(a)).not.toContain("cdph-nci-bivalve-special-advisory");
  });

  it("never matches by text: a notice whose description names the place but whose outline is elsewhere is not applied", () => {
    const ds = structuredClone(official);
    const r = ds.registry.records.find((x) => x.id === "cdph-2026-sn26-018-monterey-bivalves")!;
    r.area.description = "Santa Cruz County"; // text says Santa Cruz; the outline is still Monterey County
    const { lat, lon } = port(593);
    expect(associationsAt(ds, "port-593", lat, lon).find((x) => x.record.id === r.id)?.tier).not.toBe("geographic");
  });

  it("shore tolerance, unresolved offshore reach and nearby areas follow the stated distances", () => {
    const poly = { type: "Polygon", coordinates: [[[-122, 36], [-121, 36], [-121, 37], [-122, 37], [-122, 36]]] };
    const ds = structuredClone(official);
    const rec = ds.registry.records.find((x) => x.id === "cdph-2026-sn26-018-monterey-bivalves")!;
    (ds.geometry as { features: unknown[] }).features = [{ type: "Feature", geometry: poly, properties: { kind: "county", record_ids: [rec.id] } }];
    const at = (lonKm: number) => {
      const lon = -122 - lonKm / (111.32 * Math.cos((36.5 * Math.PI) / 180));
      return associationsAt(ds, "x", 36.5, lon).find((x) => x.record.id === rec.id);
    };
    expect(at(0.5)?.tier).toBe("geographic");
    expect(at(0.5)?.basis).toMatch(/On the shore of/);
    expect(at(3)?.tier).toBe("possible");
    expect(at(3)?.basis).toMatch(/does not say how far offshore/);
    expect(at(20)?.basis).toMatch(/separate area nearby/);
    expect(at(APPLICABILITY_KM.nearby + 5)).toBeUndefined();
    expect(distanceToPolygonKm(-121.5, 36.5, poly)).toBe(0);
  });

  it("a notice part borrowed from another outline is only ever 'possible', with its caveat", () => {
    const a = associationsAt(official, "port-605", ...(Object.values(port(605)) as [number, number]));
    const rc = a.find((x) => x.record.id === "cdph-rock-crab-viscera-north-coast-nci");
    expect(rc?.tier).toBe("possible");
    expect(rc?.caveats.join(" ")).toMatch(/borrowed from another CDPH notice/);
    expect(APPLICABILITY.noticeParts.every((p) => p.status === "candidate")).toBe(true);
  });

  it("is unreviewed unless a person's review is recorded, and records none by default", () => {
    expect(APPLICABILITY.reviews).toEqual([]);
    const { lat, lon } = port(201);
    for (const x of associationsAt(official, "port-201", lat, lon)) expect(x.review.state).toBe("not_reviewed");
    const model = { ...APPLICABILITY, reviews: [{ location: "port-201", noticeId: "cdfw-2026-razor-clam-del-norte", decision: "confirmed" as const, reviewer: "Reviewer", reviewedAt: "2026-10-20", note: "checked" }] };
    const r = associationsAt(official, "port-201", lat, lon, model).find((x) => x.record.id === "cdfw-2026-razor-clam-del-norte")!;
    expect(r.review).toMatchObject({ state: "confirmed", reviewer: "Reviewer" });
  });

  it("CDFW regulation regions follow CDFW's stated limits, with the neighbour near a boundary", () => {
    expect(cdfwRegionsAt(41.75).at.map((r) => r.id)).toEqual(["northern"]);
    expect(cdfwRegionsAt(36.97).at.map((r) => r.id)).toEqual(["central"]);
    const pp = cdfwRegionsAt(37 + 11 / 60 - 0.01);
    expect(pp.at.map((r) => r.id)).toEqual(["central"]);
    expect(pp.near.map((r) => r.id)).toEqual(["san-francisco"]);
  });
});

describe("forecast availability (nothing interpolated)", () => {
  const g: ValueGrid = { url: "", encoding: "uint16le+gzip", width: 5, height: 5, lat_first: 36, lat_step: 0.03, lon_first: -122.12, lon_step: 0.03, scale_factor: 0.01, add_offset: 0, nodata: 65535, max_quantization_error: 0, transform: "none", chunks: null } as unknown as ValueGrid;
  const N = 65535;
  // row 0 = 36.00 N; the only values are in the bottom-left corner
  const codes = new Uint16Array([10, 20, N, N, N, 30, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N]);

  it("a value in the location's own cell is 'here'", () => {
    const a = availability(g, codes, 36.0, -122.12);
    expect(a.here?.value).toBeCloseTo(0.1);
    expect(a.nearest).toBeNull();
  });

  it("an empty cell reports the nearest real cell with its distance, never a value here", () => {
    const a = availability(g, codes, 36.12, -122.0);
    expect(a.here).toBeNull();
    expect(a.nearest).not.toBeNull();
    expect(a.nearest!.km).toBeGreaterThan(5);
    expect([0.1, 0.2, 0.3]).toContain(Number(a.nearest!.value.toFixed(2)));
    expect(a.region?.n).toBeGreaterThan(0);
  });

  it("with no value within the search radius, there is no nearest and no regional summary", () => {
    const a = availability(g, new Uint16Array(25).fill(N), 36.06, -122.06);
    expect(a).toEqual({ here: null, nearest: null, region: null });
  });
});

describe("places and links", () => {
  it("resolves ports, CDPH sites and points; refuses malformed or far-away points", () => {
    expect(resolvePlace("port-593", ports)?.name).toBe("Santa Cruz");
    expect(resolvePlace("site-clam-beach", ports)?.lat).toBe(41.008);
    expect(resolvePlace(pointKey(36.95, -122.05), ports)).toMatchObject({ kind: "point", lat: 36.95, lon: -122.05 });
    expect(resolvePlace("point-10.000,-122.000", ports)).toBeNull();
    expect(resolvePlace("point-36.9,abc", ports)).toBeNull();
    expect(resolvePlace("port-9999", ports)).toBeNull();
    expect(resolvePlace("Santa Cruz", ports)).toBeNull();
  });

  it("every harvest site names its CDPH source and lies on the California coast", () => {
    for (const s of HARVEST_SITES) {
      expect(s.source.url).toMatch(/^https:\/\/cdphdata\.maps\.arcgis\.com\//);
      expect(s.lat).toBeGreaterThan(32.5);
      expect(s.lat).toBeLessThan(42.01);
    }
  });

  it("builds links with the waters, home port and audience", () => {
    expect(coastHref({ place: "port-593", home: "port-550", for: "commercial" })).toBe("/coast?place=port-593&home=port-550&for=commercial");
    expect(coastHref({ place: "port-593", for: "all" })).toBe("/coast?place=port-593");
    expect(parseAudience("shellfish")).toBe("shellfish");
    expect(parseAudience("x")).toBe("all");
  });

  it("labels laboratory results by how long ago they were sampled", () => {
    expect([recency(3), recency(30), recency(200), recency(400)]).toEqual(["recent", "older", "old", "none_this_year"]);
  });

  it("lists stations within 40 km, otherwise the nearest one anywhere", () => {
    const st = (name: string, lat: number, lon: number) => ({ station_id: name, name, lat, lon, status: "updated", source_url: "", latest: [] }) as StationLite;
    const list = [st("a", 36.96, -122.02), st("b", 38.0, -123.0)];
    expect(stationsNear(list, 36.97, -122.03).near.map((x) => x.station.name)).toEqual(["a"]);
    const far = stationsNear(list, 41.75, -124.2);
    expect(far.near).toEqual([]);
    expect(far.nearestBeyond?.station.name).toBe("b");
  });
});

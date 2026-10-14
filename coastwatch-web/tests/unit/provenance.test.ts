import { describe, expect, it } from "vitest";
import { nearestLatLimit, provenanceFeatures, STATION_FAR_KM } from "@/lib/provenance";

const limit = (lat: number, label: string): GeoJSON.Feature => ({ type: "Feature", properties: { kind: "lat_limit", label }, geometry: { type: "LineString", coordinates: [[-123, lat], [-121.9, lat]] } });
const official: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [limit(36.5243, "36°31.46′ N"), limit(37.1833, "37°11′ N")] };
const square: GeoJSON.Polygon = { type: "Polygon", coordinates: [[[-122.02, 36.79], [-121.98, 36.79], [-121.98, 36.81], [-122.02, 36.81], [-122.02, 36.79]]] };
const kinds = (fc: GeoJSON.FeatureCollection) => fc.features.map((f) => `${f.properties!.kind}${f.properties!.to ? ":" + f.properties!.to : ""}`);

describe("provenance threads", () => {
  it("the nearest official latitude limit, by latitude", () => {
    expect(nearestLatLimit(official, 36.6)?.label).toBe("36°31.46′ N");
    expect(nearestLatLimit(official, 37.1)?.label).toBe("37°11′ N");
    expect(nearestLatLimit(null, 37)).toBeNull();
  });

  it("each thread ends at the source's own location; the point never takes a source's position", () => {
    const fc = provenanceFeatures({ target: { lat: 36.6, lon: -122 }, cell: { polygon: square, lat: 36.8, lon: -122, own: false }, stations: [{ lat: 36.96, lon: -122.02, name: "Santa Cruz Wharf", km: 40 }], official });
    const ends = fc.features.filter((f) => f.properties!.kind === "end").map((f) => [f.properties!.to, (f.geometry as GeoJSON.Point).coordinates]);
    expect(ends).toEqual([
      ["cell", [-122, 36.8]],
      ["station", [-122.02, 36.96]],
      ["limit", [-122, 36.5243]],
    ]);
    const target = fc.features.find((f) => f.properties!.kind === "target")!;
    expect((target.geometry as GeoJSON.Point).coordinates).toEqual([-122, 36.6]);
    // a station beyond 25 km is still threaded, marked as too far to describe these waters
    expect(fc.features.find((f) => f.properties!.kind === "thread" && f.properties!.to === "station")!.properties!.far).toBe(40 > STATION_FAR_KM);
  });

  it("the point's own cell is outlined, not threaded; a far limit gets no ruler", () => {
    const fc = provenanceFeatures({ target: { lat: 36.8, lon: -122 }, cell: { polygon: square, lat: 36.8, lon: -122, own: true }, official });
    expect(kinds(fc)).toEqual(["target", "cell"]);
  });
});

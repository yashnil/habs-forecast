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
    // the distance label sits on the side of the cell away from the harbour (the harbour is south)
    expect(fc.features.find((f) => f.properties!.kind === "end" && f.properties!.to === "cell")!.properties!.anchor).toBe("bottom");
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

describe("the harbour view (carto T3)", async () => {
  const { GAP_NOTE, gapFeatures, lastHabSample, stationFeatures } = await import("@/lib/provenance");

  it("the gap note names the distance to the nearest modelled cell; the thread ends at that cell's centre", () => {
    const fc = gapFeatures({ lat: 36.605, lon: -121.89 }, { inDomain: true, here: false, nearestKm: 6.63, nearest: { lat: 36.63, lon: -121.95, value: 0.77, cell: { row: 1, col: 2, lat: 36.63, lon: -121.95 } } })!;
    // the end label is anchored on the side away from the harbour: the harbour is east, so the
    // text's right edge sits at the cell and the words run west
    expect(fc.features.find((f) => f.properties!.kind === "end")!.properties!.anchor).toBe("right");
    expect(gapFeatures({ lat: 36.6, lon: -121.9 }, { inDomain: true, here: false, nearestKm: 5, nearest: { lat: 36.55, lon: -121.9, value: 0.2, cell: { row: 0, col: 0, lat: 36.55, lon: -121.9 } } })!.features.find((f) => f.properties!.kind === "end")!.properties!.anchor).toBe("top");
    expect(kinds(fc)).toEqual(["note", "thread", "end"]);
    const note = fc.features[0];
    expect(note.properties!.text).toBe(`${GAP_NOTE} · nearest cell 6.6 km`);
    expect((note.geometry as GeoJSON.Point).coordinates).toEqual([-121.89, 36.605]);
    expect((fc.features[1].geometry as GeoJSON.LineString).coordinates).toEqual([[-121.89, 36.605], [-121.95, 36.63]]);
    expect(fc.features[1].properties!.label).toBe("6.6 km");
    expect(fc.features[2].properties!.label).toBe("6.6 km to the nearest modelled cell: 77 %");
    // beyond 10 km the distance is a whole number
    expect(gapFeatures({ lat: 36.6, lon: -121.9 }, { inDomain: true, here: false, nearestKm: 12.4, nearest: { lat: 36.7, lon: -122, value: 0.2, cell: { row: 0, col: 0, lat: 36.7, lon: -122 } } })!.features[0].properties!.text).toMatch(/nearest cell 12 km$/);
  });

  it("no note where the centre cell has a value, outside the domain, and no thread without a cell in reach", () => {
    expect(gapFeatures({ lat: 36.8, lon: -122 }, { inDomain: true, here: true, nearestKm: 0, nearest: { lat: 36.8, lon: -122, value: 0.5, cell: { row: 0, col: 0, lat: 36.8, lon: -122 } } })).toBeNull();
    expect(gapFeatures({ lat: 40, lon: -130 }, { inDomain: false, here: false, nearestKm: null, nearest: null })).toBeNull();
    const far = gapFeatures({ lat: 36.8, lon: -122 }, { inDomain: true, here: false, nearestKm: null, nearest: null })!;
    expect(kinds(far)).toEqual(["note"]);
    expect(far.features[0].properties!.text).toMatch(/no modelled cell within/);
  });

  it("stations carry their newest HAB sample date and whether it was in the last 30 days", () => {
    const now = new Date("2026-10-08T20:00:00Z");
    const latest = (v: string, date: string) => ({ variable: v, date });
    const stations = [
      { station_id: "HABs-SantaCruzWharf", name: "Santa Cruz Wharf", lat: 36.958, lon: -122.017, status: "ok", latest: [latest("pDA", "2026-09-30"), latest("pn_seriata", "2026-09-30"), latest("chl_extracted", "2026-10-07")] },
      { station_id: "HABs-MontereyWharf", name: "Monterey Wharf", lat: 36.6037, lon: -121.8893, status: "ok", latest: [latest("pDA", "2022-08-31"), latest("pn_seriata", "2025-08-27"), latest("pn_delicatissima", "2025-08-27")] },
      { station_id: "HABs-None", name: "Nowhere", lat: 36.5, lon: -121.9, status: "ok", latest: [] },
      { station_id: "HABs-Failed", name: "Failed", lat: 36.4, lon: -121.9, status: "failed", latest: [latest("pDA", "2026-10-01")] },
    ];
    expect(lastHabSample(stations[0])).toBe("2026-09-30"); // chlorophyll is not a HAB sample
    expect(lastHabSample(stations[1])).toBe("2025-08-27");
    expect(lastHabSample(stations[2])).toBeNull();
    const fc = stationFeatures(stations, now, "2026-10-08");
    expect(fc.features.map((f) => f.properties!.id)).toEqual(["HABs-SantaCruzWharf", "HABs-MontereyWharf", "HABs-None"]);
    expect(fc.features[0].properties).toMatchObject({ kind: "station", name: "Santa Cruz Wharf", last: "2026-09-30", recent: true, label: "Santa Cruz Wharf · last HAB sample Sep 30" });
    // an older sample says its year, and the station is drawn as a ring
    expect(fc.features[1].properties).toMatchObject({ recent: false, label: "Monterey Wharf · last HAB sample Aug 27, 2025" });
    expect(fc.features[2].properties).toMatchObject({ recent: false, last: null, label: "Nowhere · no HAB sample published" });
    expect((fc.features[1].geometry as GeoJSON.Point).coordinates).toEqual([-121.8893, 36.6037]);
    // without a clock nothing is called recent
    expect(stationFeatures(stations, null).features[0].properties!.recent).toBe(false);
    // exactly 30 days is recent; 31 is not
    expect(stationFeatures([stations[0]], new Date("2026-10-30T12:00:00Z")).features[0].properties!.recent).toBe(true);
    expect(stationFeatures([stations[0]], new Date("2026-10-31T13:00:00Z")).features[0].properties!.recent).toBe(false);
  });
});

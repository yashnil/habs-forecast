import { inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import { MIN_REGION_COVERAGE, openingLayer } from "@/lib/opening";
import { cellEdges } from "@/lib/grid";
import { EMPTY_TILE_PNG_BASE64 } from "@/lib/emptyTile";
import { nearbyStations, NEARBY_KM } from "@/lib/stations";
import type { StationLite } from "@/lib/data";
import Ajv2020 from "ajv/dist/2020";
import manifestJson from "../fixture-data/v1/manifest.json";
import failedJson from "../fixture-data-failed/v1/manifest.json";
import releasedSchema from "./compat/manifest.released.schema.json";

// Fixture: C-HARM issued 2026-10-08; latest clear views observed multi Oct 6, Sentinel-3 Oct 6,
// VIIRS Oct 4. Freshness policies are the published ones (C-HARM current 1 day, satellite 3).
const base = manifestJson as unknown as Manifest;
const clone = (): Manifest => JSON.parse(JSON.stringify(base));
const at = (iso: string) => new Date(iso);
const OPTS = { regionId: "monterey_bay", variable: "particulate_domoic", lead: 1 };
const OCT8 = at("2026-10-08T20:00:00Z"); // Oct 8 in California: C-HARM current
const OCT10 = at("2026-10-10T20:00:00Z"); // Oct 10: C-HARM 2 days old, stale

function sat(m: Manifest, id: string): LayerArtifact {
  return m.layers.find((l) => l.layer_id === id)!;
}
/** Set a latest-clear-view layer's newest observation date and its Monterey Bay coverage. */
function setSat(m: Manifest, id: string, observed: string, coverage: number) {
  const l = sat(m, id);
  l.time.observed_date = observed;
  l.time.valid_date = observed;
  l.coverage!.regions.find((r) => r.region_id === "monterey_bay")!.observed_fraction = coverage;
}
function dropCharm(m: Manifest) {
  m.layers = m.layers.filter((l) => l.group_id !== "charm");
  m.forecast_runs = m.forecast_runs.filter((r) => r.group_id !== "charm");
}

describe("opening layer policy (fresh, stale, missing, cloudy, partially covered)", () => {
  it("opens on C-HARM when its run is current", () => {
    const o = openingLayer(clone(), OCT8, OPTS);
    expect(o.group).toBe("forecast");
    expect(o.forecast.state).toBe("current");
  });

  it("stale C-HARM, recent multi-sensor view covering the bay: opens on satellite and says why", () => {
    const m = clone();
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.95);
    const o = openingLayer(m, OCT10, OPTS);
    expect(o.group).toBe("satellite");
    if (o.group !== "satellite") return;
    expect(o.product).toBe("multi");
    expect(o.reason).toBe("forecast-stale");
    expect(o.forecast.state).toBe("stale");
  });

  it("missing C-HARM: opens on satellite with reason 'unavailable'", () => {
    const m = clone();
    dropCharm(m);
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.8);
    const o = openingLayer(m, OCT10, OPTS);
    expect(o.group).toBe("satellite");
    if (o.group === "satellite") expect(o.reason).toBe("forecast-unavailable");
  });

  it("cloudy week (every product under half the bay): bathymetric map, no data layer", () => {
    const m = clone();
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.2);
    setSat(m, "olci300_chl_latest", "2026-10-09", 0.12);
    setSat(m, "viirs750_chl_latest", "2026-10-09", 0.3);
    const o = openingLayer(m, OCT10, OPTS);
    expect(o.group).toBeNull();
    if (o.group !== null) return;
    expect(o.reason).toBe("nothing-current");
    // the message offers the best partial view: the most covered current layer
    expect(o.satellite?.product).toBe("viirs750");
  });

  it("partially covered: falls through the products in order to one covering the region", () => {
    const m = clone();
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.45);
    setSat(m, "olci300_chl_latest", "2026-10-09", 0.62);
    const o = openingLayer(m, OCT10, OPTS);
    expect(o.group).toBe("satellite");
    if (o.group === "satellite") expect(o.product).toBe("olci300");
  });

  it("coverage threshold is inclusive at exactly half", () => {
    expect(MIN_REGION_COVERAGE).toBe(0.5);
    const m = clone();
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.5);
    expect(openingLayer(m, OCT10, OPTS).group).toBe("satellite");
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.4999);
    setSat(m, "olci300_chl_latest", "2026-10-09", 0.1);
    setSat(m, "viirs750_chl_latest", "2026-10-04", 0.94);
    expect(openingLayer(m, OCT10, OPTS).group).toBeNull();
  });

  it("stale satellite (newest pixel older than 3 days) is not used, however well covered", () => {
    // fixture views are Oct 6 / Oct 4: on Oct 10 they are 4 and 6 days old
    const o = openingLayer(clone(), OCT10, OPTS);
    expect(o.group).toBeNull();
  });

  it("nothing published at all: no data layer", () => {
    const m = clone();
    dropCharm(m);
    m.layers = m.layers.filter((l) => !l.layer_id.endsWith("_latest"));
    const o = openingLayer(m, OCT10, OPTS);
    expect(o.group).toBeNull();
    if (o.group === null) {
      expect(o.forecast.state).toBe("unavailable");
      expect(o.satellite).toBeNull();
    }
  });

  it("statewide uses the whole-domain coverage", () => {
    const m = clone();
    setSat(m, "multisensor_chl_latest", "2026-10-09", 0.95); // Monterey Bay well covered...
    sat(m, "multisensor_chl_latest").coverage!.domain_observed_fraction = 0.3; // ...but not the state
    setSat(m, "olci300_chl_latest", "2026-10-03", 0.95);
    setSat(m, "viirs750_chl_latest", "2026-10-03", 0.95);
    expect(openingLayer(m, OCT10, { ...OPTS, regionId: null }).group).toBeNull();
    expect(openingLayer(m, OCT10, OPTS).group).toBe("satellite");
  });

  it("a forecast that is current but lacks the requested day falls back like a missing one", () => {
    const m = clone();
    m.layers = m.layers.filter((l) => !(l.group_id === "charm" && l.time.lead_days === 1));
    setSat(m, "multisensor_chl_latest", "2026-10-08", 0.9);
    const o = openingLayer(m, OCT8, OPTS);
    expect(o.group).toBe("satellite");
    if (o.group === "satellite") expect(o.reason).toBe("forecast-unavailable");
  });
});

describe("C-HARM cell edges", () => {
  const g = { lat_first: 37, lat_step: -0.03, lon_first: -123, lon_step: 0.03, width: 3, height: 2, nodata: 65535, scale_factor: 0.001, add_offset: 0 } as unknown as ValueGrid;
  it("draws only edges that touch a valued cell, half a step from the centres", () => {
    // valid: (0,0) (0,1); row 1 all nodata, (0,2) nodata
    const codes = Uint16Array.from([10, 20, 65535, 65535, 65535, 65535]);
    const lines = cellEdges(g, codes).features[0].geometry.coordinates;
    // horizontal: top edge and bottom edge of the two-cell run; vertical: three edges
    expect(lines).toHaveLength(5);
    const top = lines.find((l) => l[0][1] === 37.015 && l[1][1] === 37.015)!;
    expect(top[0][0]).toBeCloseTo(-123.015, 5);
    expect(top[1][0]).toBeCloseTo(-122.955, 5);
    expect(lines.every((l) => l.every(([lon]) => lon <= -122.955 + 1e-9))).toBe(true);
  });
  it("no valued cell, no edges", () => {
    expect(cellEdges(g, new Uint16Array(6).fill(65535)).features[0].geometry.coordinates).toHaveLength(0);
  });
});

describe("tile index: skipped tiles are transparent", () => {
  it("the stand-in tile is a 1 x 1 RGBA PNG with every channel 0", () => {
    const png = Buffer.from(EMPTY_TILE_PNG_BASE64, "base64");
    expect(png.subarray(1, 4).toString()).toBe("PNG");
    expect(png.readUInt32BE(16)).toBe(1); // width
    expect(png.readUInt32BE(20)).toBe(1); // height
    expect(png[25]).toBe(6); // colour type RGBA
    const i = png.indexOf("IDAT");
    const len = png.readUInt32BE(i - 4);
    const raw = inflateSync(png.subarray(i + 4, i + 4 + len));
    expect([...raw]).toEqual([0, 0, 0, 0, 0]); // filter byte + R G B A
  });
});

describe("measured nearby", () => {
  const st = (id: string, lat: number, lon: number, status = "updated"): StationLite => ({ station_id: id, name: id, lat, lon, status, source_url: "", latest: [] });
  const list = [st("far", 38.5, -123.2), st("wharf", 36.958, -122.017), st("monterey", 36.605, -121.889), st("broken", 36.9, -122.0, "failed")];
  it(`lists stations within ${NEARBY_KM} km, nearest first, skipping failed ones`, () => {
    // mid-bay: Santa Cruz wharf about 19 km, Monterey wharf about 22 km
    expect(nearbyStations(list, 36.8, -121.95).map((x) => x.s.station_id)).toEqual(["wharf", "monterey"]);
    // at Santa Cruz the Monterey wharf (about 39 km) is out of reach
    expect(nearbyStations(list, 36.95, -122.02).map((x) => x.s.station_id)).toEqual(["wharf"]);
    expect(nearbyStations(list, 34.0, -120.0)).toEqual([]);
  });
});

describe("published data stays readable by the released web (BASES demo, main before M5)", () => {
  // The demo at coastwatch-demo.vercel.app reads the live production dataset and validates it
  // with this schema (identical in demo-release/2026-10-09-submitted, demo-release/2026-10-10
  // and main before M5). It rejects unknown fields, so new pipeline output must still pass.
  it.each([
    ["fixture", manifestJson],
    ["fixture (C-HARM failed)", failedJson],
  ])("%s manifest, with 512 px tiles, validates against the released schema", (_n, m) => {
    const ajv = new Ajv2020({ allErrors: true, strict: false });
    const validate = ajv.compile(releasedSchema);
    const ok = validate(m);
    expect(validate.errors ?? []).toEqual([]);
    expect(ok).toBe(true);
    const tiles = (m as unknown as Manifest).layers.flatMap((l) => (l.tiles?.relative ? [l.tiles] : []));
    expect(tiles.some((t) => t.tile_size === 512)).toBe(true);
  });
});

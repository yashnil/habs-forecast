import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { alignIn, fillZoom, fitBounds, keepInFrame, latY, lngX, unproject, worldSize, type Bounds } from "@/nx/chart/mercator";
import { noticeExtents, RAMP_BOUNDARIES, RAMP_ZONES, rampZoneAt } from "@/nx/chart/regulation";
import { extentLabel, limitLat } from "@/nx/chart/RegulationLayer";
import { eastmostShore, shoreLon } from "@/nx/chart/shore";
import { exploreBounds } from "@/nx/explore/camera";

const root = path.resolve(__dirname, "../../..");

describe("RAMP zones on the chart (E T43: latitude bands drawn as lines)", () => {
  it("draws the same boundaries the published fisheries-status dataset carries", () => {
    const ds = JSON.parse(readFileSync(path.join(root, "tests/fixture-data/v1/fisheries-status/index.json"), "utf8"));
    const published = ds.ramp.zones as { zone: string; lat_north: number; lat_south: number }[];
    for (const z of RAMP_ZONES) {
      const p = published.find((x) => x.zone === z.zone);
      expect(p, `zone ${z.zone}`).toBeTruthy();
      // the dataset rounds to 4 places (38°46.125′ = 38.76875 → 38.7688)
      expect(Math.abs(p!.lat_north - z.north)).toBeLessThan(1e-4);
      expect(Math.abs(p!.lat_south - z.south)).toBeLessThan(1e-4);
    }
  });
  it("and the curated source names them as the regulation does", () => {
    const yaml = readFileSync(path.join(root, "../data/curated/fisheries-status/status.yaml"), "utf8");
    for (const name of ["Cape Mendocino (40 10' N)", "Sonoma/Mendocino county line (38 46.125' N)", "Pigeon Point (37 11' N)", "Lopez Point (36 N)", "Point Conception (34 27' N)"]) expect(yaml).toContain(name);
    expect(RAMP_BOUNDARIES.map((b) => limitLat(b.lat))).toEqual(["42°00′ N", "40°10′ N", "38°46.125′ N", "37°11′ N", "36°00′ N", "34°27′ N"]);
  });
  it("finds a port's zone; a point on a boundary counts north of it, as the facts module reads it", () => {
    expect(rampZoneAt(36.8044)).toBe("4"); // Moss Landing
    expect(rampZoneAt(37.5)).toBe("3"); // Half Moon Bay
    expect(rampZoneAt(37 + 11 / 60)).toBe("3");
    expect(rampZoneAt(41.75)).toBe("1"); // Crescent City
    expect(rampZoneAt(33.7)).toBeNull(); // zone 6 is not described in ds3120, so not drawn
  });
});

describe("the shore (where a regulation line starts)", () => {
  it("is the mainland's, never an island's", () => {
    expect(shoreLon(37 + 11 / 60)!).toBeCloseTo(-122.39, 1); // Pigeon Point
    expect(shoreLon(36.0)!).toBeCloseTo(-121.5, 1); // Lopez Point
    expect(shoreLon(37.7)!).toBeGreaterThan(-122.6); // not the Farallones (-123.0)
    expect(shoreLon(34.0)!).toBeGreaterThan(-119.2); // not the Channel Islands
    expect(eastmostShore(36.4, 37.2)!).toBeCloseTo(-121.79, 1); // Moss Landing, the bay's east shore
    expect(shoreLon(50)).toBeNull();
  });
});

describe("notice extents (official latitude bands, never verified)", () => {
  const n = (id: string, agency: string, species: string, band: { south: number; north: number } | null, action = "Health advisory") => ({ id, agency, action, species: [species], band });
  it("merges notices with the same limits and leaves out those without a band", () => {
    const out = noticeExtents([
      n("cdph-anchovy", "CDPH", "northern anchovy", { south: 36.524333, north: 37.183333 }),
      n("cdfw-anchovy", "CDFW", "northern anchovy", { south: 36.52435, north: 37.183333 }, "Take restriction"),
      n("rock", "CDFW", "rock crab", { south: 40, north: 40.5 }, "Fishery closure"),
      n("county", "CDPH", "mussels", null),
    ]);
    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({ label: "2 anchovy notices", count: 2 });
    expect(out[1].label).toBe("CDFW rock crab fishery closure");
    expect(out.every((e) => !/verified|confirmed|lifted/i.test(e.label))).toBe(true);
  });
  it("labels every band 'not verified', long or short (T38; critic N1: the phone label is shorter, never bare)", () => {
    const [anch, rock] = noticeExtents([
      n("cdph-anchovy", "CDPH", "northern anchovy", { south: 36.524333, north: 37.183333 }),
      n("cdfw-anchovy", "CDFW", "northern anchovy", { south: 36.52435, north: 37.183333 }, "Take restriction"),
      n("rock", "CDFW", "rock crab", { south: 40, north: 40.5 }, "Fishery closure"),
    ]);
    expect(extentLabel(anch, true)).toBe("2 anchovy notices · not verified");
    expect(extentLabel(anch, false)).toBe("2 notices · not verified");
    expect(extentLabel(rock, false)).toBe("CDFW notice · not verified");
  });
});

describe("framing water, not land", () => {
  const vp = { width: 1000, height: 800 };
  const ins = { top: 44, right: 44, bottom: 44, left: 44 };
  const box: Bounds = [
    [-122.6, 36.4],
    [-121.7, 37.2],
  ];
  it("aligns a box against the free area's east edge, spare room to the west (the sea)", () => {
    const cam = alignIn(fitBounds(box, vp, ins), box, vp, ins, [1, 0.5]);
    const ws = worldSize(cam.zoom);
    const eastPx = vp.width / 2 + (lngX(-121.7) - lngX(cam.center[0])) * ws;
    expect(eastPx).toBeCloseTo(vp.width - ins.right, 0);
  });
  it("fills the free area with a box, cropping its longer side", () => {
    const z = fillZoom(box, vp, ins);
    expect(z).toBeGreaterThan(fitBounds(box, vp, ins).zoom);
    const ws = worldSize(z);
    expect(Math.max((lngX(-121.7) - lngX(-122.6)) * ws - (vp.width - 88), (latY(36.4) - latY(37.2)) * ws - (vp.height - 88))).toBeGreaterThan(-0.5);
  });
  it("keeps the whole viewport inside the chart's drawn extent", () => {
    const frame: Bounds = [
      [-131.35, 29.65],
      [-112.65, 44.65],
    ];
    const cam = keepInFrame({ center: [-124, 44], zoom: 3 }, vp, frame);
    const nw = unproject(cam, vp, 0, 0), se = unproject(cam, vp, vp.width, vp.height);
    expect(nw[0]).toBeGreaterThanOrEqual(-131.35 - 1e-6);
    expect(nw[1]).toBeLessThanOrEqual(44.65 + 1e-6);
    expect(se[0]).toBeLessThanOrEqual(-112.65 + 1e-6);
    expect(se[1]).toBeGreaterThanOrEqual(29.65 - 1e-6);
  });
  it("Explore without a port stops a little past the eastmost shore, not at the grid's inland edge", () => {
    const [[, ], [e]] = exploreBounds(null, [-122.6, 36.4, -121.15, 37.2]);
    expect(e).toBeLessThan(-121.6);
    expect(e).toBeGreaterThan(-121.8);
  });
});

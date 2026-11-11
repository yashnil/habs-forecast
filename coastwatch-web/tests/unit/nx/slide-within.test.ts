import { describe, expect, it } from "vitest";
import { fitBounds, slideWithin, visibleBounds, type Bounds } from "@/nx/chart/mercator";

// Explore at Monterey on a wide sheet: the port's box, clipped to the satellite grid, fitted into a 1000 × 800 frame
const grid: Bounds = [
  [-122.60125, 36.40125],
  [-121.75125, 37.20125],
];
// as Explore passes it (camera.ts extentBounds): open to the east, where the grid ends at the coast
const within: Bounds = [grid[0], [-114, grid[1][1]]];
const box: Bounds = [
  [-122.62, 36.38],
  [-121.8, 36.94],
];
const vp = { width: 1000, height: 800 };

describe("slideWithin", () => {
  it("moves a fitted view east so it stops at the grid's west edge instead of showing a sliver past it", () => {
    const cam = fitBounds(box, vp, { top: 44, right: 44, bottom: 44, left: 44 }, 12.5);
    expect(visibleBounds(cam, vp)[0][0]).toBeLessThan(grid[0][0] - 0.05);
    const slid = slideWithin(cam, vp, box, within);
    const v = visibleBounds(slid, vp);
    expect(slid.zoom).toBe(cam.zoom);
    expect(v[0][0]).toBeCloseTo(grid[0][0], 4);
    // the port's box stays in view (less the margin it kept past the grid)
    expect(v[1][0]).toBeGreaterThanOrEqual(box[1][0]);
    expect(v[0][1]).toBeLessThanOrEqual(box[0][1] + 0.03);
  });

  it("leaves a view alone when it is wider than the data on that axis", () => {
    const tiny: Bounds = [
      [-122.0, 36.6],
      [-121.9, 36.7],
    ];
    const cam = fitBounds(box, vp);
    expect(slideWithin(cam, vp, box, tiny).center).toEqual(cam.center);
  });
});

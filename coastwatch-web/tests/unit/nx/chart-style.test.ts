import { describe, expect, it } from "vitest";
import { chartStyle, DEPTH_LAYER_IDS, LITE_OMITS, type ChartPalette } from "@/nx/chart/style";

const P: ChartPalette = {
  depth: Array.from({ length: 9 }, () => "#335577"),
  land: "#eeeeee",
  reliefOpacity: 0.2,
  isobath: "#224466",
  contour: "#224466",
  coast: "#112233",
  graticule: "#cccccc",
  graticuleLand: "#cccccc",
  ink: "#111111",
  waterName: "#223344",
  halo: "#ffffff",
  paper: "#ffffff",
};

describe("chart style: the reduced tier's lite chart", () => {
  const full = chartStyle(P, "http://x");
  const lite = chartStyle(P, "http://x", { lite: true });
  const ids = (s: typeof full) => s.layers.map((l) => l.id);

  it("leaves out only the decorative lines and relief, keeping the depth bands, coast, contours and names", () => {
    const dropped = ids(full).filter((id) => !ids(lite).includes(id));
    expect(dropped.length).toBeGreaterThan(0);
    for (const id of dropped) expect(LITE_OMITS.has(id)).toBe(true);
    for (const id of DEPTH_LAYER_IDS) if (ids(full).includes(id)) expect(ids(lite)).toContain(id);
    expect(ids(lite).filter((id) => id.startsWith("depth-")).length).toBe(ids(full).filter((id) => id.startsWith("depth-")).length);
  });

  it("every lite layer's source is in the style, and no source is left without a layer it was kept for", () => {
    const sources = Object.keys(lite.sources);
    for (const l of lite.layers) if ("source" in l && typeof l.source === "string") expect(sources).toContain(l.source);
    for (const s of ["relief-land", "relief-land-monterey", "graticule"]) expect(sources).not.toContain(s);
  });

  it("the full chart is unchanged by default", () => {
    expect(ids(chartStyle(P, "http://x", { lite: false }))).toEqual(ids(full));
  });
});

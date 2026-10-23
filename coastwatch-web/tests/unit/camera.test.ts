import { describe, expect, it, vi } from "vitest";

// a reduced-motion device jumps; a full-motion one flies; neither may pass a present-but-undefined
// option (MapLibre reads the key's presence, and the camera became NaN)
function fakeMap() {
  const calls: { fn: string; opts: Record<string, unknown> }[] = [];
  const m = {
    cameraForBounds: vi.fn((_b: unknown, o: Record<string, unknown>) => {
      for (const [k, v] of Object.entries(o)) if (v === undefined) throw new Error(`undefined option ${k}`);
      return { center: { lng: -124.2, lat: 41 }, zoom: 7.5, bearing: 0 };
    }),
    getCenter: () => ({ lng: -122, lat: 36.8 }),
    getZoom: () => 9,
    once: vi.fn(),
    jumpTo: (opts: Record<string, unknown>) => calls.push({ fn: "jumpTo", opts }),
    easeTo: (opts: Record<string, unknown>) => calls.push({ fn: "easeTo", opts }),
    flyTo: (opts: Record<string, unknown>) => calls.push({ fn: "flyTo", opts }),
  };
  return { m, calls };
}

async function load(reduced: boolean) {
  vi.resetModules();
  vi.stubGlobal("window", { matchMedia: () => ({ matches: reduced }) });
  vi.stubGlobal("sessionStorage", { getItem: () => null, setItem: () => {} });
  vi.stubGlobal("document", { createElement: () => ({ getContext: () => null }) });
  vi.stubGlobal("requestAnimationFrame", () => 0);
  vi.stubGlobal("cancelAnimationFrame", () => {});
  return import("@/lib/camera");
}

describe("coastal journeys", () => {
  it("reduced motion jumps, with finite values and no undefined keys", async () => {
    const { journeyTo } = await load(true);
    const { m, calls } = fakeMap();
    journeyTo(m as never, { bounds: [[-124.72, 39.95], [-123.6, 42.02]] });
    expect(calls[0].fn).toBe("jumpTo");
    expect(Object.values(calls[0].opts).every((v) => v !== undefined)).toBe(true);
    expect(calls[0].opts.center).toEqual([-124.2, 41]);
  });

  it("software WebGL (no context here) eases briefly instead of flying", async () => {
    const { journeyTo, cameraTier } = await load(false);
    expect(cameraTier()).toBe("short");
    const { m, calls } = fakeMap();
    journeyTo(m as never, { bounds: [[-124.72, 39.95], [-123.6, 42.02]], padding: 48, maxZoom: 11 });
    expect(calls[0]).toMatchObject({ fn: "easeTo", opts: { duration: 300 } });
  });
});

// ---------------------------------------------------------------- v10 rework: the coastal frame
import { COAST_BOUNDS, coastCamera, seaward, waterSampler } from "@/lib/camera";
const mx = (lon: number) => (lon + 180) / 360;
const my = (lat: number) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - (0.25 * Math.log((1 + s) / (1 - s))) / Math.PI;
};
const screen = (cam: { center: [number, number]; zoom: number }, w: number, h: number, lon: number, lat: number) => {
  const wp = 512 * 2 ** cam.zoom;
  return { x: w / 2 + (mx(lon) - mx(cam.center[0])) * wp, y: h / 2 + (my(lat) - my(cam.center[1])) * wp };
};
describe("v10 rework: the opening frames the coast, not the state", () => {
  it("desktop: the coast's box fills the height between the insets, its east edge at the right inset", () => {
    const ins = { top: 64, bottom: 180, left: 300, right: 116 };
    const cam = coastCamera(1440, 900, ins);
    const nw = screen(cam, 1440, 900, COAST_BOUNDS[0][0], COAST_BOUNDS[1][1]);
    const se = screen(cam, 1440, 900, COAST_BOUNDS[1][0], COAST_BOUNDS[0][1]);
    expect(nw.y).toBeCloseTo(64, 0);
    expect(se.y).toBeCloseTo(900 - 180, 0);
    expect(se.x).toBeCloseTo(1440 - 116, 0);
    expect(nw.x).toBeGreaterThan(300);
  });
  it("a short phone gets the central coast at a legible zoom, the canyon in the strip", () => {
    const ins = { top: 108, bottom: 400, left: 12, right: 14 };
    const cam = coastCamera(320, 640, ins);
    expect(cam.zoom).toBe(4.5);
    const c = screen(cam, 320, 640, -121.95, 36.75);
    expect(c.y).toBeGreaterThan(108);
    expect(c.y).toBeLessThan(640 - 400);
    expect(c.x).toBeGreaterThan(160);
    expect(c.x).toBeLessThan(320);
  });
  it("seaward: the water's side of a shore point; none when all water or all land", () => {
    // water south of 36.95°N (a harbour on a north shore, like Santa Cruz)
    const w = { sample: (_lon: number, lat: number) => lat < 36.95 };
    const d = seaward(w, 36.97, -122.03)!;
    expect(d[1]).toBeGreaterThan(0.9); // screen y grows southward
    expect(seaward({ sample: () => true }, 36.9, -122)).toBeNull();
    expect(seaward({ sample: () => false }, 36.9, -122)).toBeNull();
    expect(seaward(null, 36.9, -122)).toBeNull();
    // the sampler: even-odd on Mercator rings
    const sq = Float64Array.from([mx(-123), my(37), mx(-122), my(37), mx(-122), my(36), mx(-123), my(36)]);
    const s = waterSampler({ rings: [sq], bbox: Float64Array.from([mx(-123), my(37), mx(-122), my(36)]) })!;
    expect(s.sample(-122.5, 36.5)).toBe(true);
    expect(s.sample(-121.5, 36.5)).toBe(false);
  });
});

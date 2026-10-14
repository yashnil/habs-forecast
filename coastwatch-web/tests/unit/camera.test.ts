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

import { afterEach, describe, expect, it, vi } from "vitest";
import { anchorShift, type CameraAnchor } from "@/nx/chart/anchor";
import { flightDelay, noteFlight, noteTicketLeaving, whenFlightNear } from "@/nx/chart/flight";
import { getMapLayers, resetMapLayers, setMapLayer, setMapLayers } from "@/nx/chart/mapLayers";
import { flightSeconds } from "@/nx/ds/motion";

/** v14 foundation (docs/coastwatch/v14/design-system/README.md §2–§5): the stage's sequencing, the ticket's anchor, the layer store. */
describe("the page change's sequence", () => {
  afterEach(() => {
    vi.useRealTimers();
    noteFlight(0, 0);
  });
  it("a flight lasts 0.5 to 1.15 s by distance", () => {
    expect(flightSeconds(0)).toBe(0.5);
    expect(flightSeconds(1600)).toBeCloseTo(1.0, 5);
    expect(flightSeconds(99999)).toBe(1.15);
  });
  it("the camera waits for a leaving ticket (170 ms), and a new ticket arrives at 62 % of the flight", async () => {
    vi.useFakeTimers();
    noteTicketLeaving();
    const delay = flightDelay();
    expect(delay).toBeGreaterThan(160);
    expect(delay).toBeLessThanOrEqual(170);
    noteFlight(delay, 1000);
    let arrived = false;
    whenFlightNear().then(() => (arrived = true));
    await vi.advanceTimersByTimeAsync(delay + 600);
    expect(arrived).toBe(false);
    await vi.advanceTimersByTimeAsync(40);
    expect(arrived).toBe(true);
  });
  it("with the camera still, a ticket arrives at once", async () => {
    noteFlight(0, 0);
    await expect(whenFlightNear()).resolves.toBeUndefined();
  });
});

describe("a ticket's anchor", () => {
  const a: CameraAnchor = { id: "t", lon: 0, lat: 0, side: "east", width: 404, height: 600, anchorY: 170, stub: 28, gutter: 24, top: 136 };
  const vp = { width: 1440, height: 900 };
  it("brings the place a stub's length west of an east ticket", () => {
    const { dx } = anchorShift(a, { x: 500, y: 400 }, vp);
    // the ticket's left edge is 1440 - 24 - 404 = 1012; the place stands 28 px west of it
    expect(500 - dx).toBe(1012 - 28);
  });
  it("keeps the place low enough for the ticket to stand under the bar, and high enough for it to fit", () => {
    expect(anchorShift(a, { x: 984, y: 100 }, vp).dy).toBe(100 - (136 + 170));
    expect(anchorShift(a, { x: 984, y: 400 }, vp).dy).toBe(0);
    expect(anchorShift(a, { x: 984, y: 880 }, vp).dy).toBe(880 - (900 - 24 - (600 - 170)));
  });
  it("mirrors for a west ticket", () => {
    const { dx } = anchorShift({ ...a, side: "west" }, { x: 900, y: 400 }, vp);
    expect(900 - dx).toBe(24 + 404 + 28);
  });
});

describe("the map's layers", () => {
  afterEach(() => resetMapLayers());
  it("a page's defaults, then the reader's choices for that page, kept for the visit", () => {
    setMapLayers("today", { regulation: true, depth: true, sea: true });
    expect(getMapLayers().on).toMatchObject({ regulation: true, depth: true, sea: true, toxin: false });
    setMapLayer("depth", false);
    setMapLayers("research", { toxin: true });
    expect(getMapLayers().on).toMatchObject({ regulation: false, depth: false, sea: false, toxin: true });
    setMapLayers("today", { regulation: true, depth: true, sea: true });
    expect(getMapLayers().on.depth).toBe(false);
  });
  it("a page's own data layers are listed after the four, off until asked", () => {
    setMapLayers("explore", { regulation: true }, [{ id: "chl", name: "Chlorophyll", question: "Q", key: "K", swatch: "data" }]);
    expect(getMapLayers().extra.map((e) => e.id)).toEqual(["chl"]);
    expect(getMapLayers().on.chl).toBe(false);
  });
});

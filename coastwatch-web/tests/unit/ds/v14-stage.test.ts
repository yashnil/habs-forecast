import { afterEach, describe, expect, it, vi } from "vitest";
import { anchorShift, anchorTarget, ticketMargin, type CameraAnchor } from "@/nx/chart/anchor";
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

describe("a ticket's anchor (the prototype's composition)", () => {
  const a: CameraAnchor = { id: "t", lon: 0, lat: 0, side: "east", width: 404, height: 530, anchorY: 170, stub: 28, gutter: 24, top: 94 };
  const vp = { width: 1440, height: 900 };
  it("stands the ticket a tenth of the screen in from the right (144 px at most, never under the gutter), the place a stub's length west of it", () => {
    expect(ticketMargin(1440, 24)).toBe(144);
    expect(ticketMargin(1100, 24)).toBe(110);
    expect(ticketMargin(200, 24)).toBe(24);
    expect(anchorTarget(a, vp).x).toBe(1440 - 144 - 404 - 28);
    const { dx } = anchorShift(a, { x: 500, y: 400 }, vp);
    expect(500 - dx).toBe(1440 - 144 - 404 - 28);
  });
  it("brings the place level with anchorY of a ticket at its top, lower only when the ticket would not fit", () => {
    expect(anchorTarget(a, vp).y).toBe(94 + 170);
    expect(anchorTarget({ ...a, height: 840 }, vp).y).toBe(94 + 170);
    expect(anchorTarget(a, { width: 1440, height: 500 }).y).toBe(94 + 170);
    expect(anchorShift(a, { x: 864, y: 400 }, vp)).toEqual({ dx: 0, dy: 400 - 264 });
  });
  it("mirrors for a west ticket", () => {
    expect(anchorTarget({ ...a, side: "west" }, vp).x).toBe(144 + 404 + 28);
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

import { describe, expect, it } from "vitest";
import { avoid, clampBox, clampSize, coverSideFor, fitBeside, fromPlacement, guidesFor, hitsAny, nudge, toPlacement, type Area } from "@/nx/ds/panels/geometry";
import { PANELS_NS, cleanPanel, layoutChanged, loadLayout, parseLayout, resetLayout, savePanel } from "@/nx/ds/panels/store";

const AREA: Area = { left: 24, top: 80, right: 1416, bottom: 876 }; // 1440 × 900, under the bar

function memory() {
  const m = new Map<string, string>();
  return {
    m,
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
  };
}

describe("panels: keeping a panel on screen", () => {
  it("leaves a box that is inside alone", () => {
    expect(clampBox({ x: 100, y: 200, w: 400, h: 300 }, AREA)).toEqual({ x: 100, y: 200, w: 400, h: 300 });
  });
  it("pulls a box dragged past any edge back inside", () => {
    expect(clampBox({ x: -300, y: 20, w: 400, h: 300 }, AREA)).toMatchObject({ x: 24, y: 80 });
    expect(clampBox({ x: 1300, y: 800, w: 400, h: 300 }, AREA)).toMatchObject({ x: 1016, y: 576 });
  });
  it("keeps the header of a box larger than the screen on screen", () => {
    expect(clampBox({ x: 500, y: 500, w: 2000, h: 1200 }, AREA)).toMatchObject({ x: 24, y: 80 });
  });
  it("keeps sizes within limits and the area", () => {
    const lim = { minW: 320, maxW: 600, minH: 160, maxH: 2000 };
    expect(clampSize({ w: 100, h: 50 }, lim, AREA)).toEqual({ w: 320, h: 160 });
    expect(clampSize({ w: 900, h: 5000 }, lim, AREA)).toEqual({ w: 600, h: 796 });
    // a screen narrower than the minimum: the screen wins
    expect(clampSize({ w: 400, h: 400 }, lim, { left: 0, top: 0, right: 300, bottom: 300 })).toEqual({ w: 300, h: 300 });
  });
  it("moves by arrow keys, clamped", () => {
    const b = { x: 30, y: 90, w: 300, h: 200 };
    expect(nudge(b, "ArrowLeft", 8, AREA)).toMatchObject({ x: 24, y: 90 });
    expect(nudge(b, "ArrowDown", 40, AREA)).toMatchObject({ x: 30, y: 130 });
    expect(nudge(b, "Enter", 8, AREA)).toBeNull();
  });
});

describe("panels: placements held against the nearest edges", () => {
  it("round-trips at the same size", () => {
    const b = { x: 900, y: 500, w: 400, h: 300 };
    const p = toPlacement(b, AREA);
    expect(p).toEqual({ ax: "r", dx: 116, ay: "b", dy: 76 });
    expect(fromPlacement(p, { w: 400, h: 300 }, AREA)).toEqual(b);
  });
  it("keeps its corner when the screen narrows (1440 to 1100) and stays on screen", () => {
    const p = toPlacement({ x: 900, y: 120, w: 400, h: 300 }, AREA);
    const small: Area = { left: 24, top: 80, right: 1076, bottom: 736 };
    const b = fromPlacement(p, { w: 400, h: 300 }, small);
    expect(b.x + b.w).toBe(1076 - 116);
    expect(b.y).toBe(120);
    // a placement far out (saved on a big screen) is clamped on load
    const far = fromPlacement({ ax: "l", dx: 3000, ay: "t", dy: 2000 }, { w: 400, h: 300 }, small);
    expect(far).toEqual({ x: 676, y: 436, w: 400, h: 300 });
  });
});

describe("panels: guides, covers, the default layout", () => {
  it("shows guides at the screen's edges and other panels' edges, within tolerance only", () => {
    const g = guidesFor({ x: 27, y: 300, w: 300, h: 200 }, [{ x: 900, y: 100, w: 400, h: 700 }], AREA);
    expect(g).toContainEqual(expect.objectContaining({ axis: "x", at: 24, kind: "area" }));
    expect(guidesFor({ x: 500, y: 300, w: 396, h: 200 }, [{ x: 900, y: 100, w: 400, h: 700 }], AREA)).toContainEqual(expect.objectContaining({ axis: "x", at: 900, kind: "panel" }));
    expect(guidesFor({ x: 400, y: 300, w: 300, h: 200 }, [{ x: 900, y: 100, w: 400, h: 700 }], AREA)).toEqual([]);
  });
  it("says which edge a panel covers, or none when it floats mid-chart", () => {
    const vp = { w: 1440, h: 900 };
    expect(coverSideFor({ x: 1000, y: 94, w: 404, h: 740 }, vp)).toBe("right");
    expect(coverSideFor({ x: 24, y: 94, w: 404, h: 740 }, vp)).toBe("left");
    expect(coverSideFor({ x: 24, y: 560, w: 760, h: 316 }, vp)).toBe("bottom");
    expect(coverSideFor({ x: 500, y: 300, w: 400, h: 200 }, vp)).toBe("none");
  });
  it("narrows a resting dock to stop short of the ticket beside it, never under its minimum", () => {
    const ticket = { x: 896, y: 94, w: 440, h: 700 };
    expect(fitBeside({ x: 24, y: 500, w: 1000, h: 300 }, [ticket], 24, 320)).toBe(848);
    // a ticket above the dock's band does not narrow it
    expect(fitBeside({ x: 24, y: 500, w: 1000, h: 300 }, [{ ...ticket, h: 300 }], 24, 320)).toBe(1000);
    expect(fitBeside({ x: 24, y: 500, w: 1000, h: 300 }, [{ ...ticket, x: 200 }], 24, 320)).toBe(320);
  });
});

describe("panels: never under the floating chrome", () => {
  const chips = { x: 24, y: 80, w: 442, h: 44 };
  const zoom = { x: 1376, y: 756, w: 40, h: 120 };
  it("moves a short panel dragged under the layer chips just below them", () => {
    const b = avoid({ x: 30, y: 82, w: 400, h: 200 }, [chips, zoom], AREA);
    expect(hitsAny(b, [chips, zoom])).toBe(false);
    expect(b).toMatchObject({ x: 30, y: 132, h: 200 });
  });
  it("lets a tall panel keep its column under the chips by giving up height", () => {
    const b = avoid({ x: 24, y: 80, w: 404, h: 790 }, [chips], AREA);
    expect(hitsAny(b, [chips])).toBe(false);
    expect(b.x).toBe(24);
    expect(b.y).toBe(132);
    expect(b.h).toBe(876 - 132);
  });
  it("steps a panel off the zoom buttons", () => {
    const b = avoid({ x: 1000, y: 600, w: 400, h: 270 }, [chips, zoom], AREA);
    expect(hitsAny(b, [zoom])).toBe(false);
  });
  it("finds the nearest free place beside another panel after the screen shrinks", () => {
    const sea = { x: 24, y: 520, w: 620, h: 216 };
    const small: Area = { left: 24, top: 80, right: 1076, bottom: 736 };
    const b = avoid({ x: 300, y: 240, w: 320, h: 400 }, [chips, sea], small);
    expect(hitsAny(b, [chips, sea])).toBe(false);
  });
  it("leaves a box with nothing in the way alone", () => {
    expect(avoid({ x: 600, y: 300, w: 300, h: 200 }, [chips, zoom], AREA)).toEqual({ x: 600, y: 300, w: 300, h: 200 });
  });
});

describe("panels: persistence", () => {
  it("saves, merges and resets one page's layout under a versioned key", () => {
    const st = memory();
    savePanel("coast", "ticket", { p: { ax: "r", dx: 40, ay: "t", dy: 10 } }, st);
    savePanel("coast", "sea-window", { c: true }, st);
    savePanel("coast", "ticket", { w: 480 }, st);
    const l = loadLayout("coast", st);
    expect(l.panels).toEqual({ ticket: { p: { ax: "r", dx: 40, ay: "t", dy: 10 }, w: 480 }, "sea-window": { c: true } });
    expect(st.m.has(`${PANELS_NS}coast`)).toBe(true);
    expect(PANELS_NS).toBe("nx-panels-v1:");
    expect(layoutChanged(l)).toBe(true);
    // pages are separate
    expect(layoutChanged(loadLayout("trip", st))).toBe(false);
    // un-collapsing the only change empties the panel's entry
    savePanel("coast", "sea-window", { c: false }, st);
    expect(loadLayout("coast", st).panels["sea-window"]).toBeUndefined();
    resetLayout("coast", st);
    expect(st.m.has(`${PANELS_NS}coast`)).toBe(false);
    expect(layoutChanged(loadLayout("coast", st))).toBe(false);
  });
  it("reads a malformed, foreign-version or tampered entry as nothing moved", () => {
    expect(parseLayout("{not json").panels).toEqual({});
    expect(parseLayout(JSON.stringify({ v: 2, panels: { ticket: { c: true } } })).panels).toEqual({});
    expect(parseLayout(JSON.stringify({ v: 1, panels: { ticket: { p: { ax: "x", dx: 1, ay: "t", dy: 1 }, w: -5, c: "yes" } } })).panels).toEqual({});
    expect(cleanPanel({ p: { ax: "l", dx: 10.4, ay: "b", dy: 3 }, h: 300, junk: 1 })).toEqual({ p: { ax: "l", dx: 10, ay: "b", dy: 3 }, h: 300 });
  });
  it("survives storage that throws", () => {
    const bad = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("full");
      },
      removeItem: () => {
        throw new Error("blocked");
      },
    };
    expect(loadLayout("coast", bad).panels).toEqual({});
    expect(() => savePanel("coast", "ticket", { c: true }, bad)).not.toThrow();
    expect(() => resetLayout("coast", bad)).not.toThrow();
  });
});

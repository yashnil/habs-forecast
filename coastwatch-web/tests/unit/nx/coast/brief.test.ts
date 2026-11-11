import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { REGULATED_BARS, whaleMeasures, windowWords } from "@/nx/coast/brief";
import { coastGates, coastSea } from "@/nx/coast/brief.server";

/** My Coast's gates and sea window on the committed fixture (tests/fixture-data) and recorded NOAA responses. */
beforeAll(() => {
  process.env.CW_DATA_BASE_URL = "/data/fixture/v1";
  process.env.CW_DATA_DIR = path.resolve(__dirname, "../../../fixture-data/v1");
  process.env.CW_CONDITIONS_FIXTURE = path.resolve(__dirname, "../../../conditions-fixtures.json");
  process.env.CW_NWS_API = "1";
  process.env.CW_FIXED_NOW = "2026-10-13T06:00:00Z";
});

describe("gates for one fishery at one port", () => {
  it("Dungeness at Moss Landing: the delay, the zone's measures without repeating it, and the latest CDPH test", async () => {
    const g = await coastGates("port-592", "dungeness_commercial");
    if (!g || !g.ok) throw new Error("no gates");
    expect(g.season?.text).toMatch(/delayed to at least Jan 1, 2027/);
    expect(g.season?.tone).toBe("regulation");
    expect(g.season?.source).toBe("CDFW");
    expect(g.season?.verified).toBe(false);
    expect(g.whale?.zone).toBe("4");
    expect(g.whale?.area).toBe("Pigeon Point to Lopez Point");
    // the delay is the season line's; recreational rules are not a commercial reader's
    expect(g.whale?.measures).not.toMatch(/delayed|recreational/i);
    expect(g.whale?.measures).toMatch(/Fleet Advisory\. No depth limit or gear reduction on record\./);
    expect(g.whale?.next).toBe("Dec 15–18, then Jan 13–15, subject to change");
    expect(g.whale?.verified).toBe(false);
    expect(g.test?.source).toBe("CDPH");
    expect(g.test?.site).toBe("Monterey Bay");
    expect(g.test?.max).toBe(9.8);
    expect(g.test?.action).toBe(30);
    expect(g.test?.over).toBe(0);
  });

  it("whale measures belong to Dungeness only; a fishery without a CDPH test has none", async () => {
    const g = await coastGates("port-592", "salmon_commercial");
    if (!g || !g.ok) throw new Error("no gates");
    expect(g.whale).toBeNull();
    expect(g.test).toBeNull();
    // a routine season end is drawn calmly, still plainly not open
    expect(g.season?.tone).toBe("ink");
  });

  it("rock crab keeps CDPH's censored values as printed", async () => {
    const g = await coastGates("port-592", "rock_crab");
    if (!g || !g.ok) throw new Error("no gates");
    expect(g.test?.raw).toContain("<2.5");
    expect(g.test?.raw).toContain("ND");
  });

  it("refuses an unknown fishery and an unknown port", async () => {
    expect(await coastGates("port-592", "bivalve_shellfish")).toEqual({ ok: false, error: "unknown fishery" });
    expect(await coastGates("port-9999", "dungeness_commercial")).toBeNull();
  });
});

describe("the sea window", () => {
  it("carries NWS's own sentence for every period, the measured buoys and the predicted tides", async () => {
    const s = await coastSea("port-592");
    if (!s || !s.ok) throw new Error("no sea");
    expect(s.sea.state).toBe("ok");
    expect(s.sea.zone).toBe("PZZ535");
    expect(s.sea.lanes.length).toBeGreaterThan(4);
    for (const p of s.sea.lanes) expect(p.text.length).toBeGreaterThan(10);
    expect(s.sea.lanes[1].wind).toBe("NE 5 kt, then W 10–15 kt");
    expect(s.sea.buoy?.id).toBe("46236");
    expect(s.sea.windBuoy?.id).toBe("46092");
    expect(s.sea.tides.length).toBeGreaterThan(4);
  });
});

describe("words", () => {
  it("writes CDFW's window in the house style", () => {
    expect(windowWords("December 15-18 (then January 13-15), subject to change")).toBe("Dec 15–18, then Jan 13–15, subject to change");
  });
  it("states depth limits and gear reductions when the zone has them", () => {
    expect(whaleMeasures("Fleet Advisory", 30, 50, false)).toBe("Fleet Advisory. Traps 30 fathoms or shallower. Trap numbers cut by 50 %.");
    expect(whaleMeasures(null, null, null, false)).toBe("No depth limit or gear reduction on record.");
  });
  it("lists the four regulated bars only", () => {
    expect(Object.keys(REGULATED_BARS).sort()).toEqual(["port-201", "port-220", "port-223", "port-606"]);
  });
});

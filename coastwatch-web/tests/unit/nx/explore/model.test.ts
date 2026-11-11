import { describe, expect, it } from "vitest";
import type { Manifest } from "@/generated/schema";
import { parseDerived } from "@/lib/derived";
import manifest from "../../../fixture-data/v1/manifest.json";
import derivedRaw from "../../../fixture-derived/v1/derived/index.json";
import { buildExploreModel } from "@/nx/explore/model";
import { legendCaption, currentsCaption } from "@/nx/explore/words";

const now = new Date("2026-10-13T06:00:00Z");
const model = buildExploreModel({ manifest: manifest as unknown as Manifest, derived: parseDerived(derivedRaw), stations: [], baseUrl: "/data/fixture/v1", now });
const product = (id: string) => model.products.find((p) => p.id === id)!;

describe("Explore model from the fixture (no manifest field added; frames are the published ones)", () => {
  it("offers chlorophyll, temperature, C-HARM and the outlook, chlorophyll first", () => {
    expect(model.products.map((p) => p.id)).toEqual(["chl", "sst", "charm", "outlook"]);
    expect(model.today).toBe("2026-10-12");
  });

  it("chlorophyll: the latest view first, then only days that have a grid, and VIIRS on its own track", () => {
    const chl = product("chl");
    expect(chl.defaultFrame).toBe("olci:latest");
    const olci = chl.frames.filter((f) => f.track === "olci").map((f) => f.id);
    expect(olci).toEqual(["olci:latest", "olci:2026-10-01", "olci:2026-10-02", "olci:2026-10-05", "olci:2026-10-06"]);
    const latest = chl.frames[0];
    expect([latest.start, latest.end]).toEqual(["2026-10-01", "2026-10-06"]);
    expect(latest.ageGrid).toBeTruthy();
    expect(latest.kind).toBe("measured");
    // availability includes the days nothing was seen, at zero
    const days = chl.tracks.find((t) => t.id === "olci")!.days;
    expect(days.find((d) => d.date === "2026-10-03")?.volume).toBe(0);
    expect(chl.frames.some((f) => f.track === "viirs")).toBe(true);
  });

  it("C-HARM: a run whose every lead is before today is a past run, opening on its last lead (T3)", () => {
    const c = product("charm");
    expect(c.pastRun).toBe(true);
    expect(c.frames.every((f) => f.past && f.kind === "model")).toBe(true);
    expect(c.defaultFrame).toBe("charm:3");
    expect(legendCaption(c, c.frames[3])).toMatch(/Past run: every day it covers is before today/);
    expect(legendCaption(c, c.frames[3])).toMatch(/Probability of conditions in the water, not toxin in seafood/);
  });

  it("the temperature is an analysis and says so; the outlook carries its error and no-signal words", () => {
    const sst = product("sst");
    expect(sst.frames[0].kind).toBe("analysis");
    expect(legendCaption(sst, sst.frames[0])).toMatch(/^Analysis · /);
    const o = product("outlook");
    expect(o.frames[0].kind).toBe("outlook");
    expect(o.frames[0].typicalGrid).toBeTruthy();
    expect(legendCaption(o, o.frames[0])).toMatch(/factor of 2/);
    expect(legendCaption(o, o.frames[0])).toMatch(/no signal/);
    expect(legendCaption(product("chl"), product("chl").frames[0])).toMatch(/Algae, not toxin/);
  });

  it("currents: every published hour plus the 24 h mean, hours in time order, the default a real hour", () => {
    const c = model.currents!;
    const hours = c.frames.filter((f) => !f.mean);
    expect(hours).toHaveLength(24);
    expect(hours.map((h) => h.time)).toEqual([...hours.map((h) => h.time)].sort());
    expect(c.frames.filter((f) => f.mean)).toHaveLength(1);
    expect(hours.some((h) => h.id === c.defaultFrame)).toBe(true);
    expect(currentsCaption(hours[0])).toMatch(/Relative speed/);
    expect(currentsCaption(hours[0])).toMatch(/No streak is no data, not calm/);
  });

  it("the rail spans every frame and today; the extent is the observed chlorophyll's grid", () => {
    const dates = model.products.flatMap((p) => p.frames.flatMap((f) => [f.start, f.end]));
    expect(dates.every((d) => d >= model.axis[0] && d <= model.axis[1])).toBe(true);
    expect(model.axis[1] > model.today).toBe(true);
    const [w, s, e, n] = model.extent!;
    expect(w).toBeCloseTo(-122.6025, 3);
    expect(e).toBeCloseTo(-121.75, 3);
    expect(s).toBeLessThan(n);
  });

  it("frame ids are unique and the model stays small enough to pass as props", () => {
    const ids = model.products.flatMap((p) => p.frames.map((f) => f.id));
    expect(new Set(ids).size).toBe(ids.length);
    expect(JSON.stringify(model).length).toBeLessThan(60_000);
  });
});

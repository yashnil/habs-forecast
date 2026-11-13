import { describe, expect, it } from "vitest";
import { forecastFromSea } from "@/nx/trips/forecast";
import { compareWindows, crossings, windowCells, type AdvisoryIn } from "@/nx/trips/windows";
import { NOW, recordedSea, seaOf } from "./seaFixture";

const none = { windKt: null, seasFt: null };
const TIDES = [
  { at: "2026-10-13 05:56", ft: 0.9, type: "L" as const },
  { at: "2026-10-13 12:03", ft: 5.4, type: "H" as const },
  { at: "2026-10-13 19:27", ft: 0.1, type: "L" as const },
];

describe("windowCells", () => {
  const f650 = forecastFromSea(recordedSea("PZZ650"));

  it("marks a crossing with the stated number and the reader's limit, never more", () => {
    const cells = windowCells(f650, { windKt: 15, seasFt: 6 }, [], [], NOW);
    const tue = cells.find((c) => c.key === "2026-10-13|day")!;
    expect(tue.state).toBe("reported");
    expect(tue.crossings.map((c) => c.words)).toEqual(["Wind stated up to 20 kt; your limit 15 kt", "Gusts stated to 25 kt; your wind limit 15 kt"]);
    const mon = cells.find((c) => c.key === "2026-10-12|night")!;
    expect(mon.crossings).toEqual([]);
  });

  it("equal to the limit is not over it; no limits set means no crossings", () => {
    expect(crossings({ windHi: 20, gustKt: null, seasHi: 5 }, { windKt: 20, seasFt: 5 })).toEqual([]);
    expect(windowCells(f650, none, [], [], NOW).every((c) => c.crossings.length === 0)).toBe(true);
  });

  it("says a limit has nothing to compare with when the forecast states no seas (bay zone)", () => {
    const cells = windowCells(forecastFromSea(recordedSea("PZZ530")), { windKt: 20, seasFt: 4 }, [], [], NOW);
    expect(cells[0].notStated).toEqual(["seas"]);
    expect(cells[0].crossings).toEqual([]);
  });

  it("a missing interval and a departure date beyond the forecast are unavailable, never favourable", () => {
    const sea = seaOf("PZZ535", "2026-10-12T20:10:00-07:00", [
      { name: "Tonight", text: "NW wind 5 to 10 kt. Seas 3 to 4 ft." },
      { name: "Tue", text: "NW wind 10 to 15 kt. Seas 3 to 4 ft." },
      { name: "Wed", text: "NW wind 25 to 30 kt. Seas 8 to 10 ft." },
    ]);
    const cells = windowCells(forecastFromSea(sea), { windKt: 20, seasFt: 6 }, [], [], NOW, { departDate: "2026-10-20" });
    expect(cells.map((c) => `${c.key}:${c.state}`)).toEqual([
      "2026-10-12|night:reported",
      "2026-10-13|day:reported",
      "2026-10-13|night:unavailable",
      "2026-10-14|day:reported",
      "2026-10-20|day:unavailable",
      "2026-10-20|night:unavailable",
    ]);
    const gap = cells[2];
    expect(gap.crossings).toEqual([]);
    expect(gap.period).toBeNull();
    expect(cells[4].departure).toBe(true);
  });

  it("a period already over is 'expired'", () => {
    const cells = windowCells(f650, { windKt: 5, seasFt: null }, [], [], "2026-10-13T14:00:00Z");
    expect(cells[0].state).toBe("expired");
    expect(cells[0].crossings).toEqual([]);
  });

  it("puts tide highs and lows inside the derived interval (CO-OPS wall time)", () => {
    const cells = windowCells(f650, none, TIDES, [], NOW);
    expect(cells.find((c) => c.key === "2026-10-12|night")!.tides.map((t) => t.at)).toEqual(["2026-10-13 05:56"]);
    expect(cells.find((c) => c.key === "2026-10-13|day")!.tides.map((t) => t.at)).toEqual(["2026-10-13 12:03"]);
    expect(cells.find((c) => c.key === "2026-10-13|night")!.tides.map((t) => t.at)).toEqual(["2026-10-13 19:27"]);
  });

  it("attaches an advisory to the periods its onset–ends overlaps (recorded SCA, until 3 AM)", () => {
    const sca: AdvisoryIn = { event: "Small Craft Advisory", headline: "Small Craft Advisory until 3 AM PDT", url: "https://api.weather.gov/alerts/x", onset: "2026-10-12T21:06:00-07:00", ends: "2026-10-13T03:00:00-07:00" };
    const cells = windowCells(f650, none, [], [sca], NOW);
    expect(cells.filter((c) => c.advisories.length).map((c) => c.key)).toEqual(["2026-10-12|night"]);
  });
});

describe("compareWindows", () => {
  const f = forecastFromSea(recordedSea("PZZ650"));
  const cells = windowCells(f, { windKt: 15, seasFt: null }, TIDES, [], NOW);
  const by = (k: string) => cells.find((c) => c.key === k)!;

  it("lists exactly the reported conditions that differ, including a direction change", () => {
    const d = compareWindows(by("2026-10-12|night"), by("2026-10-13|day"));
    const f2 = Object.fromEntries(d.map((x) => [x.field, [x.a, x.b]]));
    expect(f2.windDir).toEqual(["NW", "N → W"]);
    expect(f2.wind).toEqual(["5–10 kt", "5–20 kt"]);
    expect(f2.gust).toEqual(["not stated", "25 kt"]);
    expect(f2.limits[1]).toContain("Wind stated up to 20 kt; your limit 15 kt");
    expect(f2.seas).toBeUndefined(); // 3–5 ft both
    expect(f2.state).toBeUndefined();
  });

  it("an unavailable window differs in state, never reads as calm", () => {
    const sea = seaOf("PZZ535", "2026-10-12T20:10:00-07:00", [{ name: "Tonight", text: "NW wind 5 to 10 kt. Seas 3 to 4 ft." }]);
    const cs = windowCells(forecastFromSea(sea), none, [], [], NOW, { departDate: "2026-10-15" });
    const d = compareWindows(cs[0], cs[1]);
    expect(d[0]).toMatchObject({ field: "state", a: "NWS forecast", b: "no forecast for this period" });
    expect(d.find((x) => x.field === "wind")!.b).toBe("—");
  });

  it("identical windows have no differences", () => {
    expect(compareWindows(by("2026-10-13|day"), by("2026-10-13|day"))).toEqual([]);
  });
});

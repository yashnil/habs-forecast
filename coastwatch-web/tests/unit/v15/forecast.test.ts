import { describe, expect, it } from "vitest";
import { compassDeg, directionSequence, forecastFromSea, periodLabel, statedWinds, tideMs, waveComponents, windStatedWords, zonedIso } from "@/nx/trips/forecast";
import { recordedSea, seaOf } from "./seaFixture";

describe("forecastFromSea: the recorded NWS zone forecasts", () => {
  it("keys periods by derived valid time, the first starting at issuance (PZZ535, Monterey Bay)", () => {
    const f = forecastFromSea(recordedSea("PZZ535"));
    expect(f.zone).toBe("PZZ535");
    expect(f.issued).toBe("2026-10-12T20:10:00-07:00");
    expect(f.periods.map((p) => p.key).slice(0, 3)).toEqual(["2026-10-12|night", "2026-10-13|day", "2026-10-13|night"]);
    const [tonight, tue] = f.periods;
    expect(tonight.start).toBe("2026-10-13T03:10:00.000Z");
    expect(tonight.end).toBe("2026-10-13T13:00:00.000Z");
    expect(tue.start).toBe("2026-10-13T13:00:00.000Z");
    expect(tue.end).toBe("2026-10-14T01:00:00.000Z");
    expect(periodLabel(tue)).toBe("Tue 13 day, 6 AM–6 PM");
  });

  it("keeps direction changes within a period, in NWS's order", () => {
    const [tonight] = forecastFromSea(recordedSea("PZZ535")).periods;
    expect(directionSequence(tonight.text)).toEqual(["W", "NW", "N"]);
    expect(windStatedWords(tonight.text)).toBe("W 10–15 kt, then NW 5–10 kt late this evening, then N after midnight");
    expect(tonight.windDir).toBe("W");
    expect(tonight.windLo).toBe(5);
    expect(tonight.windHi).toBe(15);
  });

  it("reads gusts only when stated, and every wave component (PZZ650)", () => {
    const f = forecastFromSea(recordedSea("PZZ650"));
    const tue = f.periods.find((p) => p.key === "2026-10-13|day")!;
    expect(tue.gustKt).toBe(25);
    expect(tue.windHi).toBe(20);
    expect(tue.seasLo).toBe(3);
    expect(tue.seasHi).toBe(5);
    expect(tue.waves).toEqual([
      { dir: "NW", ft: 2, periodS: 6 },
      { dir: "W", ft: 3, periodS: 10 },
      { dir: "S", ft: 2, periodS: 17 },
    ]);
    expect(f.periods.find((p) => p.key === "2026-10-13|night")!.gustKt).toBeNull();
  });

  it("reads the '...becoming' and 'less than' wording without inventing a lower speed (PZZ740)", () => {
    const tue = forecastFromSea(recordedSea("PZZ740")).periods.find((p) => p.key === "2026-10-13|day")!;
    expect(tue.windDir).toBe("variable");
    expect(tue.windLo).toBeNull();
    expect(tue.windHi).toBe(10);
    expect(waveComponents(tue.text)[0]).toEqual({ dir: "E", ft: 1, periodS: 4 });
  });

  it("'light' winds state no number (PZZ645)", () => {
    const tue = forecastFromSea(recordedSea("PZZ645")).periods.find((p) => p.key === "2026-10-13|day")!;
    expect(tue.windDir).toBe("light");
    expect(tue.windLo).toBeNull();
    expect(tue.windHi).toBe(20);
  });

  it("a forecast without seas leaves seas null, never 0 (PZZ530, bay waters)", () => {
    const f = forecastFromSea(recordedSea("PZZ530"));
    expect(f.periods.every((p) => p.seasHi === null && p.seasLo === null && p.waves.length === 0)).toBe(true);
  });

  it("is empty and truthful when the sea could not be read", () => {
    const sea = recordedSea("PZZ535");
    const f = forecastFromSea({ ...sea, sea: { ...sea.sea, state: "error", lanes: [] } });
    expect(f.periods).toEqual([]);
    expect(forecastFromSea(null).periods).toEqual([]);
  });

  it("an 'Overnight' period after midnight belongs to the previous evening's night", () => {
    const f = forecastFromSea(seaOf("PZZ535", "2026-10-13T02:30:00-07:00", [
      { name: "Overnight", text: "NW wind 5 to 10 kt. Seas 3 to 4 ft." },
      { name: "Today", text: "NW wind 10 to 15 kt. Seas 3 to 4 ft." },
      { name: "Tonight", text: "NW wind 10 to 15 kt. Seas 3 to 4 ft." },
    ]));
    expect(f.periods.map((p) => p.key)).toEqual(["2026-10-12|night", "2026-10-13|day", "2026-10-13|night"]);
    expect(f.periods[0].start).toBe("2026-10-13T09:30:00.000Z");
    expect(f.periods[0].end).toBe("2026-10-13T13:00:00.000Z");
  });

  it("parses hand-written variants: gusts in their own sentence, seas building, easing", () => {
    const w = statedWinds("SW wind 15 to 20 kt, easing to 10 kt after midnight. Gusts up to 30 kt. Seas 6 to 8 ft, building to 9 to 11 ft.");
    expect(w.map((x) => [x.dir, x.lo, x.hi, x.verb])).toEqual([["SW", 15, 20, null], [null, 10, 10, "easing to"]]);
    expect(w.map((x) => x.gust)).toContain(30);
    const f = forecastFromSea(seaOf("PZZ535", "2026-10-12T20:10:00-07:00", [{ name: "Tonight", text: "SW wind 15 to 20 kt, easing to 10 kt after midnight. Gusts up to 30 kt. Seas 6 to 8 ft, building to 9 to 11 ft." }]));
    expect(f.periods[0]).toMatchObject({ windLo: 10, windHi: 20, gustKt: 30, seasLo: 6, seasHi: 11 });
  });
});

describe("time helpers", () => {
  it("converts Pacific wall time across DST", () => {
    expect(zonedIso("2026-10-13", 6, 0, "America/Los_Angeles")).toBe("2026-10-13T13:00:00.000Z");
    expect(zonedIso("2026-12-01", 6, 0, "America/Los_Angeles")).toBe("2026-12-01T14:00:00.000Z");
    expect(tideMs("2026-10-13 12:03")).toBe(Date.parse("2026-10-13T19:03:00Z"));
    expect(tideMs("2026-10-13T19:03:00Z")).toBe(Date.parse("2026-10-13T19:03:00Z"));
  });
  it("compass points", () => {
    expect(compassDeg("NW")).toBe(315);
    expect(compassDeg("light")).toBeNull();
  });
});

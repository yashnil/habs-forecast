import { describe, expect, it } from "vitest";
import { lanePeriods, periodLine, periodNumbers, windWords } from "@/pilot/today/marine";
import { parseCoastalText } from "@/lib/nwsText";
import { parseZoneForecast } from "@/lib/conditions";

describe("NWS periods as lanes (Today)", () => {
  it("reads wind, gusts, a later change, seas and the first wave detail from NWS's own words", () => {
    const p = periodNumbers("W wind 10 to 15 kt, becoming NW 5 to 10 kt late this evening. Seas 3 to 5 ft. Wave Detail: NW 5 ft at 10 seconds and SW 2 ft at 15 seconds.");
    expect(p).toMatchObject({ dir: "W", windLo: 10, windHi: 15, seasLo: 3, seasHi: 5, swell: { dir: "NW", ft: 5, s: 10 } });
    const q = periodNumbers("NE wind around 5 kt, backing to W 10 to 15 kt in the afternoon. Seas around 3 ft.");
    expect(q).toMatchObject({ dir: "NE", windLo: 5, windHi: 15, seasHi: 3 });
    expect(periodNumbers("NW winds 20 to 25 kt with gusts to 35 kt. Combined seas 8 to 10 ft.")).toMatchObject({ windHi: 25, gust: 35, seasLo: 8, seasHi: 10 });
    expect(periodLine(periodNumbers("Winds light. Seas 2 ft."))).toBe("light winds · seas 2 ft");
    // a number the text does not state stays null
    expect(periodNumbers("Patchy fog.")).toMatchObject({ dir: null, windHi: null, seasHi: null });
  });

  it("says a later wind change as a change, never joining the first direction to the top speed", () => {
    const p = periodNumbers("N wind 5 to 10 kt, becoming NW 10 to 15 kt in the afternoon. Seas 6 to 9 ft.");
    expect(periodLine(p)).toBe("N 5–10 kt, then NW 10–15 kt · seas 6–9 ft");
    expect(periodLine(p)).not.toContain("N 5–15");
    expect(windWords(periodNumbers("NW wind 5 to 10 kt, veering to NE after midnight."))).toBe("NW 5–10 kt, then NE");
    // cut short for the hero, an ellipsis says NWS stated more
    expect(windWords(periodNumbers("W wind 10 to 15 kt, becoming NW 5 to 10 kt late this evening, veering to N after midnight."), 2)).toBe("W 10–15 kt, then NW 5–10 kt …");
    expect(windWords(periodNumbers("NE wind around 5 kt, backing to W 10 to 15 kt with gusts to 20 kt."))).toBe("NE 5 kt, then W 10–15 kt, gusts 20");
  });

  it("dates each period by its own name and labels the column by its day", () => {
    const l = lanePeriods([{ name: "Tonight", text: "W wind 10 kt." }, { name: "Tue", text: "N wind 5 kt." }, { name: "Tue Night", text: "N wind 5 kt." }], "2026-10-13T03:10:00Z");
    expect(l.map((p) => [p.short, p.night, p.date])).toEqual([["Today", true, "2026-10-12"], ["Tue", false, "2026-10-13"], ["Tue", true, "2026-10-13"]]);
  });

  it("the parsers keep 4 periods unless asked for more (My Coast unchanged)", () => {
    const periods = Array.from({ length: 12 }, (_, i) => ({ name: `P${i}`, detailedForecast: "x" }));
    expect(parseZoneForecast("PZZ535", "Monterey Bay", { properties: { periods } })!.periods).toHaveLength(4);
    expect(parseZoneForecast("PZZ535", "Monterey Bay", { properties: { periods } }, 10)!.periods).toHaveLength(10);
    const text = ["PZZ535-131200-", "Monterey Bay-", "833 PM PDT Mon Oct 12 2026", "", ...["TONIGHT", "TUE", "TUE NIGHT", "WED", "WED NIGHT", "THU"].map((n) => `.${n}...W wind 10 kt.`), "$$"].join("\n");
    expect(parseCoastalText("PZZ535", text)!.forecast.periods).toHaveLength(4);
    expect(parseCoastalText("PZZ535", text, 10)!.forecast.periods).toHaveLength(6);
  });
});

import { describe, expect, it } from "vitest";
import { dayLabel, risingSpans, seaDays, seasWords, tideClock, tideWords } from "@/nx/coast/sea";
import type { SeaPeriod } from "@/nx/coast/brief";

const tides = [
  { at: "2026-10-13 01:19", ft: 3.878, type: "H" as const },
  { at: "2026-10-13 05:56", ft: 2.551, type: "L" as const },
  { at: "2026-10-13 12:03", ft: 5.438, type: "H" as const },
  { at: "2026-10-13 19:27", ft: 0.111, type: "L" as const },
  { at: "2026-10-14 02:21", ft: 3.679, type: "H" as const },
];
const p = (name: string, date: string, night: boolean, lo: number | null, hi: number | null): SeaPeriod => ({ name, date, night, text: "", wind: null, seasLo: lo, seasHi: hi });

describe("the sea window's days and tides", () => {
  it("reads CO-OPS wall-clock times as written (Pacific), never through the browser's zone", () => {
    expect(tideClock("2026-10-13 00:23")).toBe("12:23 AM");
    expect(tideClock("2026-10-13 12:03")).toBe("12:03 PM");
    expect(tideClock("2026-10-13 19:27")).toBe("7:27 PM");
  });
  it("lists a day's tides and the rising spans that touch it, across midnight", () => {
    expect(tideWords(tides, "2026-10-13")).toBe("High 3.9 ft 1:19 AM, low 2.6 ft 5:56 AM, high 5.4 ft 12:03 PM, low 0.1 ft 7:27 PM");
    expect(risingSpans(tides, "2026-10-13")).toEqual(["5:56 AM–12:03 PM", "7:27 PM–2:21 AM Wed"]);
  });
  it("groups NWS periods by their own dates, today first, and never picks one", () => {
    const days = seaDays([p("Tonight", "2026-10-12", true, 3, 5), p("Tue", "2026-10-13", false, 3, 4), p("Tue Night", "2026-10-13", true, 3, 3)], "2026-10-13T06:00:00Z");
    expect(days.map((d) => d.label)).toEqual(["Today", "Tue 13"]);
    expect(days[0].day).toBeNull();
    expect(seasWords(days[1].day)).toBe("3–4");
    expect(seasWords(days[1].night)).toBe("3");
    expect(dayLabel("2026-10-17", "2026-10-13T06:00:00Z")).toBe("Sat 17");
  });
});

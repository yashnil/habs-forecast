import { describe, expect, it } from "vitest";
import { windChangeNote, windSteps } from "@/lib/trip";

/**
 * v8 fix pass: an NWS period's wind is a sequence of stretches. "SE 5 kt … becoming W 10 to 15 kt late"
 * was drawn as "SE 5, to 15 kt": the 15 kt took the morning's direction. Every change keeps its own.
 */
describe("windSteps", () => {
  it("keeps each change of direction and speed in the forecaster's order (the Monterey Bay case, Oct 22)", () => {
    const s = windSteps("SE wind around 5 kt, veering to SW early this afternoon, then becoming W 10 to 15 kt late. Seas around 3 ft. Wave Detail: NW 3 ft at 10 seconds.");
    expect(s).toEqual([
      { dir: "SE", kt: "5", when: null },
      { dir: "SW", kt: null, when: "early this afternoon" },
      { dir: "W", kt: "10–15", when: "late" },
    ]);
    expect(windChangeNote(s)).toBe("then SW early this afternoon, then W 10–15 kt late");
  });

  it("a steady wind is one stretch, with no change note", () => {
    const s = windSteps("NW wind 15 to 20 kt. Seas 3 to 4 ft.");
    expect(s).toEqual([{ dir: "NW", kt: "15–20", when: null }]);
    expect(windChangeNote(s)).toBeNull();
  });

  it("gusts are not a stretch", () => {
    expect(windSteps("W winds 10 to 15 kt with gusts up to 25 kt. Seas 5 ft.")).toEqual([{ dir: "W", kt: "10–15", when: null }]);
    expect(windSteps("W winds 10 to 15 kt, gusts to 25 kt. Seas 5 ft.")).toEqual([{ dir: "W", kt: "10–15", when: null }]);
  });

  it("reads the NWS's upper-case text and a speed change without a direction", () => {
    const s = windSteps("NW WIND 10 TO 15 KT, BECOMING 15 TO 20 KT IN THE AFTERNOON. SEAS 5 TO 7 FT.");
    expect(s).toEqual([
      { dir: "NW", kt: "10–15", when: null },
      { dir: null, kt: "15–20", when: "IN THE AFTERNOON" },
    ]);
    expect(windChangeNote(s)).toBe("then 15–20 kt IN THE AFTERNOON");
  });

  it("backing and increasing are changes too; the wind sentence need not come first", () => {
    expect(windChangeNote(windSteps("S wind 10 kt, backing to SE after midnight."))).toBe("then SE after midnight");
    expect(windChangeNote(windSteps("Seas 3 ft. NW wind 10 kt, increasing to 15 to 20 kt in the afternoon."))).toBe("then 15–20 kt in the afternoon");
  });

  it("no wind stated: no stretches", () => {
    expect(windSteps("Seas 3 to 5 ft. Patchy fog.")).toEqual([]);
  });
});

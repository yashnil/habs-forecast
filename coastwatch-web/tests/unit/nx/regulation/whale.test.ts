import { describe, expect, it } from "vitest";
import { actionClauses, clauseSector, sectorOf, whaleMeasures } from "@/nx/regulation/whale";

/** The one whale-measures formatter Today and My Coast share (T41, T42: commercial and recreational apart). */
const ACTION = "Commercial season delayed to at least Jan 1, 2027; recreational crab traps prohibited; Fleet Advisory.";
const zone = (over: Partial<Parameters<typeof whaleMeasures>[0]> = {}) => ({ zone: "4", action: ACTION, depth_limit_fathoms: null, gear_reduction_pct: null, ...over });

describe("whale measures by sector", () => {
  it("splits CDFW's action into clauses and addresses each to a sector", () => {
    expect(actionClauses(ACTION)).toEqual(["Commercial season delayed to at least Jan 1, 2027", "recreational crab traps prohibited", "Fleet Advisory"]);
    expect(clauseSector("recreational crab traps prohibited")).toBe("recreational");
    expect(clauseSector("Commercial season delayed")).toBe("commercial");
    expect(clauseSector("Fleet Advisory")).toBe("commercial");
    expect(clauseSector("Zone closed to all crab trap gear")).toBe("both");
    expect(sectorOf("dungeness_recreational")).toBe("recreational");
    expect(sectorOf("dungeness_commercial")).toBe("commercial");
  });

  it("never tells a commercial crabber the recreational rule (the Today bug)", () => {
    const m = whaleMeasures(zone(), "commercial", { seasonSaysIt: true });
    expect(m.short).toBe("Fleet Advisory, no depth or gear limit on record");
    expect(JSON.stringify(m)).not.toMatch(/recreational|traps prohibited/i);
    expect(m.sentences).toEqual(["Fleet Advisory.", "No depth limit or gear reduction on record."]);
  });

  it("keeps the delay clause when the season line does not already say it, quoted in CDFW's words", () => {
    const m = whaleMeasures(zone(), "commercial");
    expect(m.quote).toBe("Commercial season delayed to at least Jan 1, 2027; Fleet Advisory.");
    expect(m.sentences[0]).toBe("Commercial season delayed to at least Jan 1, 2027.");
  });

  it("builds depth and gear from the structured fields, for the commercial fleet only", () => {
    const z = zone({ depth_limit_fathoms: 30, gear_reduction_pct: 50 });
    expect(whaleMeasures(z, "commercial", { seasonSaysIt: true }).short).toBe("Fleet Advisory, 30 fathom limit, traps cut 50 %");
    const rec = whaleMeasures(z, "recreational");
    expect(JSON.stringify(rec)).not.toMatch(/fathom|cut|Fleet Advisory|Commercial/);
    expect(rec.short).toBe("Recreational crab traps prohibited");
  });

  it("a recreational reader with no recreational clause is told so, not given the commercial list", () => {
    const m = whaleMeasures(zone({ action: "Fleet Advisory." }), "recreational");
    expect(m.short).toBe("No recreational measure on record");
    expect(m.quote).toBeNull();
  });
});

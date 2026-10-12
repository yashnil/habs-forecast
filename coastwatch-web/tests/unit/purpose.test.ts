import { describe, expect, it } from "vitest";
import { defaultAudience, purposeLabel, PURPOSES, sectionOrder } from "@/lib/purpose";

/** A purpose changes defaults and order only (docs/coastwatch/29 §10). */
describe("purpose", () => {
  it("offers five purposes, including just exploring", () => {
    expect(PURPOSES.map((p) => p.id)).toEqual(["commercial", "recreational", "research", "ports", "exploring"]);
  });
  it("sets My Coast's starting filter, never a narrower one than the purpose implies", () => {
    expect(defaultAudience(null)).toBe("all");
    expect(defaultAudience({ purpose: "commercial" })).toBe("commercial");
    expect(defaultAudience({ purpose: "recreational", harvest: "fishing" })).toBe("recreational");
    expect(defaultAudience({ purpose: "recreational", harvest: "shellfish" })).toBe("shellfish");
    for (const p of ["research", "ports", "exploring"] as const) expect(defaultAudience({ purpose: p })).toBe("all");
  });
  it("reorders sections for research only, and every order keeps all four sections", () => {
    const all = ["official", "measured", "model", "observed"];
    expect(sectionOrder({ purpose: "research" })).toEqual(["measured", "model", "observed", "official"]);
    for (const p of [null, { purpose: "commercial" as const }, { purpose: "ports" as const }, { purpose: "research" as const }]) expect([...sectionOrder(p)].sort()).toEqual([...all].sort());
    expect(sectionOrder(null)[0]).toBe("official");
  });
  it("labels the choice in plain words", () => {
    expect(purposeLabel(null)).toBe("No preference");
    expect(purposeLabel({ purpose: "recreational", harvest: "shellfish" })).toBe("Shellfish harvest");
  });
});

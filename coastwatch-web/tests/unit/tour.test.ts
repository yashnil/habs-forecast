import { describe, expect, it } from "vitest";
import { TOUR } from "@/content/copy";

/** The tour teaches the map; it must never read as a safety statement (same rule as the pages). */
describe("tour copy", () => {
  const steps = Object.values(TOUR.steps);
  it("has nine short steps, each with a title and at most two paragraphs", () => {
    expect(steps).toHaveLength(9);
    for (const s of steps) {
      expect(s.title.length).toBeLessThanOrEqual(40);
      expect(s.body.length).toBeGreaterThan(0);
      expect(s.body.length).toBeLessThanOrEqual(2);
    }
  });
  it("says 'safe' or 'open' only in negated sentences", () => {
    const sentences = steps.flatMap((s) => s.body.join(" ").split(/(?<=[.!?])\s+/));
    const bad = sentences.filter((t) => /\b(safe|open|all clear)\b/i.test(t) && !/\b(not|no|never|does not)\b/i.test(t));
    expect(bad).toEqual([]);
  });
  it("keeps the model, observation and official kinds distinct", () => {
    expect(TOUR.steps.forecast.body.join(" ")).toMatch(/probability, not a measurement/);
    expect(TOUR.steps.satellite.body.join(" ")).toMatch(/Chlorophyll is not toxin/);
    expect(TOUR.steps.official.body.join(" ")).toMatch(/has not verified them/);
    expect(TOUR.steps.what.body.join(" ")).toMatch(/not an official source/);
  });
});

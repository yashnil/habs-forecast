import { describe, expect, it } from "vitest";
import { askDeterministic } from "@/pilot/ask/engine";
import { ADVERSARIAL_ENGINE, GOLDEN, grade, GUARD_CASES, GUARD_CONTROLS, guardLines, loaderFor, nowFor, requestFor, type EngineCase } from "@/pilot/ask/evals/harness";
import { fixturePortToday, FIXTURE_NOW } from "@/pilot/ask/fixture";
import { check } from "@/pilot/ask/guard";

/**
 * Ask Neritic evals against the deterministic answerer (readiness "demo"). Ported from #49
 * (golden 22, adversarial 14) and extended: ≥ 40 golden, ≥ 25 adversarial engine cases, plus
 * scripted guard cases and controls.
 */

async function run(c: EngineCase) {
  const loader = loaderFor(c);
  const now = nowFor(c);
  const a = await askDeterministic(requestFor(c), { now, mode: "demo", loader });
  const facts = a.place ? ((await loader(a.place.place_id, now))?.today.facts ?? []) : [];
  return { a, failures: grade(c, a, facts, now) };
}

describe("golden questions", () => {
  it("has at least 40 cases", () => expect(GOLDEN.length).toBeGreaterThanOrEqual(40));
  for (const c of GOLDEN)
    it(c.id, async () => {
      const { failures } = await run(c);
      expect(failures, c.question).toEqual([]);
    });
});

describe("adversarial questions and data", () => {
  it("has at least 25 cases", () => expect(ADVERSARIAL_ENGINE.length).toBeGreaterThanOrEqual(25));
  for (const c of ADVERSARIAL_ENGINE)
    it(c.id, async () => {
      const { failures } = await run(c);
      expect(failures, c.question).toEqual([]);
    });
});

describe("guard: scripted bad answers are caught", () => {
  const facts = fixturePortToday("port-592")!.facts;
  const now = new Date(FIXTURE_NOW);
  for (const c of GUARD_CASES)
    it(c.id, () => {
      const r = check(guardLines(c), facts, now);
      for (const code of c.expect ?? []) expect(r.codes, JSON.stringify(r.violations)).toContain(code);
    });
  for (const c of GUARD_CONTROLS)
    it(`control: ${c.id}`, () => {
      const r = check(guardLines(c), c.provided ? facts.filter((f) => c.provided!.includes(f.id)) : facts, now);
      expect(r.violations).toEqual([]);
    });
});

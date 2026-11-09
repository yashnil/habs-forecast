import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { PortFact } from "@/pilot/contracts";
import { snapshotOf } from "@/pilot/ask/compose";
import { askDeterministic } from "@/pilot/ask/engine";
import { ADVERSARIAL_ENGINE, PUBLISHED, allText, grade, type EngineCase } from "@/pilot/ask/evals/harness";
import { loadPortToday } from "@/pilot/ask/source";
import type { AskRequest, SeenSnapshot } from "@/pilot/ask/types";
import { fixtureServerEnv } from "../fixtures/portToday";

/**
 * Ask Neritic on the REAL fact shape. The loader is the app's own (`loadPortToday` →
 * src/pilot/today/portToday.ts → buildPortToday), reading the committed web fixture data and recorded
 * NOAA answers exactly as the e2e servers do. No hand-written facts: what a fisherman is told here is
 * what the Today page shows, with the same ids.
 */
const env = fixtureServerEnv();
const saved: Record<string, string | undefined> = {};
const NOW = new Date(env.CW_FIXED_NOW);

beforeAll(() => {
  for (const [k, v] of Object.entries(env)) {
    saved[k] = process.env[k];
    process.env[k] = v;
  }
});
afterAll(() => {
  for (const k of Object.keys(env)) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

async function request(c: EngineCase): Promise<AskRequest> {
  let since: SeenSnapshot | null = null;
  if (c.since) {
    const base = await loadPortToday(c.place ?? "port-592", NOW);
    since = snapshotOf(base!.today);
    if (c.since.startsWith("changed:")) since[c.since.slice(8)] = "an older version";
  }
  return { question: c.question, place: c.place, first: false, since };
}

async function run(c: EngineCase) {
  const a = await askDeterministic(await request(c), { now: NOW, mode: "demo", loader: loadPortToday });
  const facts: PortFact[] = a.place ? ((await loadPortToday(a.place.place_id, NOW))?.today.facts ?? []) : [];
  return { a, facts, failures: grade(c, a, facts, NOW) };
}

describe("the app's own loader on the fixture data", () => {
  it("returns the Today facts for a pilot port, every fishery, with the sea, labelled as the test fixture", async () => {
    const r = await loadPortToday("port-592", NOW);
    expect(r?.dataset).toBe("fixture");
    const ids = r!.today.facts.map((f) => f.id);
    for (const id of ["status:dungeness_commercial", "status:spiny_lobster", "whale:ramp:zone-4", "test:bma-g-monterey-bay-dungeness-crab", "charm:pda", "outlook:chl8d", "weather:nws", "weather:nws:2026-10-13:day", "tide:next", "tide:day:2026-10-13", "sea:buoy", "age:data"]) expect(ids).toContain(id);
    expect(new Set(ids).size).toBe(ids.length);
    // official first: nothing measured or forecast is listed above an official record
    const firstOther = r!.today.facts.findIndex((f) => f.basis !== "official");
    expect(r!.today.facts.slice(firstOther).some((f) => f.basis === "official")).toBe(false);
  });
  it("a port that is not in the data is null, never invented", async () => {
    expect(await loadPortToday("port-99999", NOW)).toBeNull();
  });
});

describe("golden questions on the real fact shape", () => {
  it("has at least 40 cases", () => expect(PUBLISHED.length).toBeGreaterThanOrEqual(40));
  for (const c of PUBLISHED)
    it(c.id, async () => {
      const { a, failures } = await run(c);
      expect(failures, c.question).toEqual([]);
      // dates are in the reader's terms, as Today words them ("checked Oct 12", "tested Sep 28"): never an ISO date, never hours for a day-dated record
      expect(allText(a), c.question).not.toMatch(/\b\d{4}-\d{2}-\d{2}\b/);
      for (const x of a.citations) expect(x.as_of_words ?? "", x.fact_id).not.toMatch(/\d{4}-\d{2}/);
      // every citation opens its own Today card
      for (const x of a.citations) expect(x.today_href).toBe(`/today?place=${a.place!.place_id}#fact-${encodeURIComponent(x.fact_id)}`);
      if (a.citations.length) expect(a.dataset).toBe("fixture");
    });
});

describe("adversarial questions on the real fact shape", () => {
  // the cases that doctor a record or move the clock belong to the eval records (evals.test.ts)
  const portable = ADVERSARIAL_ENGINE.filter((c) => !c.fixture && !c.now);
  it("covers the safety bait, out-of-area and injection questions", () => expect(portable.length).toBeGreaterThanOrEqual(18));
  for (const c of portable)
    it(c.id, async () => {
      const { failures } = await run(c);
      expect(failures, c.question).toEqual([]);
    });
});

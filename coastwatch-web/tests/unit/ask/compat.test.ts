import { describe, expect, it } from "vitest";
import type { PortFact } from "@/pilot/contracts";
import { askDeterministic } from "@/pilot/ask/engine";
import { fixturePortToday, FIXTURE_NOW } from "@/pilot/ask/evals/sampleFacts";
import { check } from "@/pilot/ask/guard";

/**
 * The Today track's buildPortToday emits some forecasts without valid_for (the current NWS period,
 * the next tide) and different ids (weather:nws, tide:next, charm:pda, outlook:chl8d). Ask must
 * work on that shape too (checked against pilot/today eadccefa7 before integration).
 */
const now = new Date(FIXTURE_NOW);
function todayShaped() {
  const t = fixturePortToday("port-592")!;
  const facts: PortFact[] = t.facts
    .filter((f) => f.kind !== "sea" && f.kind !== "weather" && f.kind !== "tide")
    .concat([
      { id: "weather:nws", place_id: "port-592", kind: "weather", severity: "info", headline: "NWS PZZ535 forecast, Tonight: W 10–15 kt · seas 3–5 ft", basis: "forecast", verified: null, source: { name: "NOAA National Weather Service", url: "https://forecast.weather.gov/" }, as_of: "2026-11-08T11:00:00Z", valid_for: null },
      { id: "tide:next", place_id: "port-592", kind: "tide", severity: "info", headline: "Next tide: high 5.4 ft at 12:03 PM.", basis: "forecast", verified: null, source: { name: "NOAA CO-OPS", url: null }, as_of: "2026-11-08", valid_for: null },
    ]);
  return { ...t, facts };
}

describe("works on the Today track's fact shape", () => {
  it("quotes the current NWS period and says it may not reach tomorrow", async () => {
    const t = todayShaped();
    const a = await askDeterministic({ question: "Wind and tide tomorrow?", place: "port-592" }, { now, mode: "demo", loader: async () => ({ today: t, dataset: "published" }) });
    expect(a.outcome).toBe("answered");
    expect(a.citations.map((c) => c.fact_id)).toEqual(["weather:nws", "tide:next"]);
    expect(a.lines[0].text).toMatch(/may not reach tomorrow/);
    expect(check(a.lines, t.facts, now).passed).toBe(true);
    expect(a.dataset).toBe("published");
  });
});

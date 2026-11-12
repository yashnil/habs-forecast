import { describe, expect, it } from "vitest";
import { PILOT_PORTS } from "@/pilot/contracts";
import { PILOT_PORT_WORDS, askCoverage, savedCoverage, uncoveredWords } from "@/pilot/ask/ui/coverage";

/** Ask Neritic answers only for the pilot ports, and never opens on a different port from the one asked for. */
const names = { "port-473": "Bodega Bay", "port-592": "Moss Landing" };

describe("which port Ask opens on", () => {
  it("opens on a pilot port that was asked for", () => {
    for (const p of PILOT_PORTS) expect(askCoverage(p.place_id, names)).toEqual({ state: "covered", place: p.place_id });
  });

  it("a port outside the pilot is named as not covered: no pilot port stands in for it", () => {
    const c = askCoverage("port-473", names);
    expect(c).toEqual({ state: "uncovered", port: { key: "port-473", name: "Bodega Bay" } });
    expect(JSON.stringify(c)).not.toContain(PILOT_PORTS[0].place_id);
  });

  it("a port the list does not name is still not covered (and not named)", () => {
    expect(askCoverage("port-999", names)).toEqual({ state: "uncovered", port: { key: "port-999", name: null } });
  });

  it("no port asked for, or something that is not a port: the first pilot port, openly", () => {
    for (const raw of [null, undefined, "", "site-12", "port-", "<script>"]) expect(askCoverage(raw, names)).toEqual({ state: "open", place: PILOT_PORTS[0].place_id });
  });

  it("the reader's saved port follows the same rule when the address named none", () => {
    expect(savedCoverage("port-550", names)).toEqual({ state: "covered", place: "port-550" });
    expect(savedCoverage("port-473", names)).toEqual({ state: "uncovered", port: { key: "port-473", name: "Bodega Bay" } });
    expect(savedCoverage(null, names)).toBeNull();
    expect(savedCoverage("site-12", names)).toBeNull();
  });

  it("says which ports are covered and that nothing shown is about the reader's port", () => {
    expect(PILOT_PORT_WORDS).toBe("Moss Landing, Monterey, Santa Cruz and Princeton / Half Moon Bay");
    const w = uncoveredWords({ key: "port-473", name: "Bodega Bay" });
    expect(w.title).toBe("Ask Neritic does not cover Bodega Bay yet.");
    expect(w.body).toContain("Moss Landing, Monterey, Santa Cruz and Princeton / Half Moon Bay");
    expect(w.body).toContain("Nothing below is about Bodega Bay");
    expect(uncoveredWords({ key: "port-999", name: null }).title).toBe("Ask Neritic does not cover this port yet.");
  });
});

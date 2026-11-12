import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Which saved copy answers an address the pack does not hold (public/sw.js `nearestSaved`, read from the
 * worker's own source: it is not a module). A page is rendered for its whole address, so the worker serves a
 * copy only at the address it was saved for and sends every other address of that path on to the nearest one.
 * Serving the brief's HTML at the trip view's address was review finding F-08 (React error 418 offline).
 */
const src = readFileSync(path.join(__dirname, "..", "..", "..", "public", "sw.js"), "utf8");
const body = /function nearestSaved\(requestUrl, saved\) \{[\s\S]*?\n\}\n/.exec(src);
if (!body) throw new Error("public/sw.js no longer has nearestSaved");
const nearestSaved = new Function(`${body[0]}; return nearestSaved;`)() as (requestUrl: string, saved: string[]) => string | null;

const O = "https://neritic.example";
const BRIEF = `${O}/coast?place=port-592`;
const TRIP = `${O}/coast?place=port-592&view=trip`;
const near = (path: string, saved: string[]) => nearestSaved(`${O}${path}`, saved)?.slice(O.length) ?? null;

describe("the saved copy an address is sent on to", () => {
  it("the trip view and the brief are two documents: each address gets its own", () => {
    expect(near("/coast?place=port-592&view=trip", [BRIEF, TRIP])).toBe("/coast?place=port-592&view=trip");
    expect(near("/coast?place=port-592", [BRIEF, TRIP])).toBe("/coast?place=port-592");
    expect(near("/coast?place=port-592", [TRIP, BRIEF])).toBe("/coast?place=port-592");
  });

  it("a trip address with more on it (a fishery, parameters in another order, no port) goes to the saved trip view, never the brief", () => {
    for (const p of ["/coast?view=trip&fishery=dungeness_commercial", "/coast?view=trip&place=port-592", "/coast?place=port-550&view=trip", "/coast?place=port-592&view=trip&trip=abc"]) expect(near(p, [BRIEF, TRIP]), p).toBe("/coast?place=port-592&view=trip");
  });

  it("anything else on that path goes to the brief: a bare address, another port, a point and date carried from Explore", () => {
    for (const p of ["/coast", "/coast?place=port-550", "/coast?place=port-592&at=36.8,-121.9&date=2026-10-14", "/coast?utm_source=x"]) expect(near(p, [BRIEF, TRIP]), p).toBe("/coast?place=port-592");
  });

  it("a pack saved before the trip view was kept (the brief only) sends a trip address to the brief, where the page is what its HTML says", () => {
    expect(near("/coast?place=port-592&view=trip", [BRIEF])).toBe("/coast?place=port-592");
  });

  it("Research, Alerts and Today: another station, port or variable goes to the copy that was saved", () => {
    const research = `${O}/research?station=monterey-wharf&var=pda`;
    expect(near("/research?station=santa-cruz-wharf&var=pda", [research])).toBe("/research?station=monterey-wharf&var=pda");
    expect(near("/research?place=port-592", [research])).toBe("/research?station=monterey-wharf&var=pda");
    expect(near("/alerts?place=port-473", [`${O}/alerts?place=port-592`])).toBe("/alerts?place=port-592");
    expect(near("/alerts", [`${O}/alerts?place=port-592`])).toBe("/alerts?place=port-592");
    // two saved Todays: the port asked for
    expect(near("/today?place=port-550&utm_source=x", [`${O}/today?place=port-592`, `${O}/today?place=port-550`])).toBe("/today?place=port-550");
  });

  it("nothing saved for the path: nothing", () => {
    expect(nearestSaved(`${O}/coast?view=trip`, [])).toBeNull();
  });

  it("the worker never answers a navigation with a copy saved for another address", () => {
    // the only ignoreSearch read left on the navigation path collects candidates for the redirect
    const nav = /async function packNavigate\(request\) \{[\s\S]*?\n\}\n/.exec(src)![0];
    expect(nav).toContain("Response.redirect(to, 302)");
    expect(nav).not.toMatch(/c\.match\(key\)/);
  });
});

import { describe, expect, it } from "vitest";
import { EXPLORE_URL, exploreHref, frontDoorRedirect, namesExplore, pilotFrontDoor } from "@/pilot/frontDoor";

/** The pilot's front door (NERITIC_PILOT=1): a bare "/" opens Today; every deep link still opens Explore. */
describe("pilot front door", () => {
  const ON = { NERITIC_PILOT: "1" };

  it("is off unless NERITIC_PILOT is exactly 1: the public release is unchanged", () => {
    expect(pilotFrontDoor({})).toBe(false);
    expect(pilotFrontDoor({ NERITIC_PILOT: "0" })).toBe(false);
    expect(pilotFrontDoor({ NERITIC_PILOT: "true" })).toBe(false);
    expect(pilotFrontDoor(ON)).toBe(true);
    for (const sp of [{}, { region: "monterey_bay" }, { utm_source: "x" }]) expect(frontDoorRedirect(sp, {})).toBeNull();
    expect(exploreHref(false)).toBe("/explore");
  });

  it("with the flag, a bare visit opens Today", () => {
    expect(frontDoorRedirect({}, ON)).toBe("/today");
    // a campaign tag is not a request for the map
    expect(frontDoorRedirect({ utm_source: "newsletter", utm_medium: "email" }, ON)).toBe("/today");
    expect(frontDoorRedirect({ fbclid: "abc", ref: "x" }, ON)).toBe("/today");
  });

  it("with the flag, every existing deep link still opens Explore", () => {
    for (const sp of [
      { region: "monterey_bay" },
      { port: "593" },
      { region: "monterey_bay", port: "593" },
      { var: "particulate_domoic", lead: "2" },
      { layer: "olci300" },
      { layer: "olci300", age: "1" },
      { inspect: "36.8,-121.9" },
      { sel: "harbor:592" },
      { at: "36.8000,-121.9000,9.50" },
      { from: "/coast?place=port-592" },
      { fp: "charm:10:20" },
      // a parameter Explore gains later is a deep link too: only a bare address is redirected
      { something_new: "1" },
      { utm_source: "x", sel: "harbor:592" },
    ])
      expect(frontDoorRedirect(sp, ON), JSON.stringify(sp)).toBeNull();
    expect(namesExplore({ view: "map" })).toBe(true);
  });

  it("v12: Explore has its own address with or without the flag, and the nav uses it", () => {
    expect(EXPLORE_URL).toBe("/explore");
    expect(exploreHref(true)).toBe(EXPLORE_URL);
    expect(exploreHref(false)).toBe(EXPLORE_URL);
  });
});

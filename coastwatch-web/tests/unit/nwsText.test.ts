import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { headlineAlerts, parseCoastalText, parseIssued, zoneNear, zonePageUrl, zoneTextUrl, ZONE_ANCHORS } from "@/lib/nwsText";
import { marineHeadline } from "@/lib/conditions";

const FX = path.resolve(__dirname, "fixtures/conditions");
const text = (f: string) => readFileSync(path.join(FX, f), "utf8");

describe("NWS coastal waters forecast text (tgftp)", () => {
  it("reads one zone's name, issue time and periods, as published", () => {
    const r = parseCoastalText("PZZ535", text("pzz535-2026-10-13.txt"))!;
    expect(r.forecast.zone).toBe("PZZ535");
    expect(r.forecast.zoneName).toBe("Monterey Bay");
    expect(r.forecast.updated).toBe("2026-10-14T03:33:00.000Z"); // 833 PM PDT Tue Oct 13 2026
    expect(r.forecast.periods[0].name).toBe("Tonight");
    expect(r.forecast.periods[0].text).toMatch(/^NW wind 10 to 15 kt, becoming NE 5 to 10 kt after midnight\. Seas 3 to 5 ft\./);
    expect(r.forecast.periods.map((p) => p.name)).toEqual(["Tonight", "Wed", "Wed Night", "Thu"]);
    // the change of direction stays in the headline: "NW 10 to 15" alone would stand for the whole night
    expect(marineHeadline(r.forecast.periods[0])).toBe("NW wind 10 to 15 kt, becoming NE 5 to 10 kt after midnight · Seas 3 to 5 ft");
    expect(r.headlines).toEqual([]);
  });

  it("keeps a zone's advisory headline, quoted", () => {
    const r = parseCoastalText("PZZ650", text("pzz650-2026-10-13.txt"))!;
    expect(r.forecast.zoneName).toBe("East Santa Barbara Channel from Pt. Conception to Pt. Mugu CA including Santa Cruz Island");
    expect(r.headlines).toEqual(["SMALL CRAFT ADVISORY IN EFFECT UNTIL 9 PM PDT THIS EVENING"]);
    const a = headlineAlerts(r.headlines, zonePageUrl("PZZ650"));
    expect(a[0].event).toBe("Small craft advisory");
    expect(a[0].url).toBe("https://forecast.weather.gov/shmrn.php?mz=pzz650");
  });

  it("a product without the zone gives nothing, never another zone's forecast", () => {
    expect(parseCoastalText("PZZ545", text("pzz535-2026-10-13.txt"))).toBeNull();
  });

  it("issue times in PDT and PST", () => {
    expect(parseIssued("833 PM PDT Tue Oct 13 2026")).toBe("2026-10-14T03:33:00.000Z");
    expect(parseIssued("1015 AM PST Mon Dec 7 2026")).toBe("2026-12-07T18:15:00.000Z");
    expect(parseIssued("not a time")).toBeNull();
  });

  it("zones only for known places: each port and harvest site, and nothing far from them", () => {
    expect(zoneNear(41.74901, -124.20398)?.zone).toBe("PZZ450"); // Crescent City
    expect(zoneNear(41.008, -124.115)?.zone).toBe("PZZ450"); // Clam Beach
    expect(zoneNear(40.0255, -124.0664)?.zone).toBe("PZZ455"); // Shelter Cove
    expect(zoneNear(36.60024, -121.89469)?.zone).toBe("PZZ535"); // Monterey
    expect(zoneNear(37.5037, -122.48185)?.zone).toBe("PZZ545"); // Half Moon Bay
    expect(zoneNear(32.71512, -117.13414)?.zone).toBe("PZZ740"); // San Diego
    expect(zoneNear(36.7, -122.6)).toBeNull(); // 60 km off Monterey Bay: no guess
    // every anchor's zone is an inner (coastal or bay) zone with a published product
    for (const a of ZONE_ANCHORS) expect(["PZZ450", "PZZ455", "PZZ530", "PZZ535", "PZZ540", "PZZ545", "PZZ645", "PZZ650", "PZZ655", "PZZ740"]).toContain(a.zone);
    expect(zoneTextUrl("PZZ535")).toBe("https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/pz/pzz535.txt");
  });
});

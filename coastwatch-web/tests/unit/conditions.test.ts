import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { BUOYS, TIDE_STATIONS, WATER_TEMP_STATIONS } from "@/lib/noaaStations";
import { compass, marineHeadline, nearest, parseAlerts, parseNdbc, parseTides, parseWaterTemp, parseZoneForecast, HARBOR_TEMP_AT_KM, HARBOR_TEMP_MAX_KM } from "@/lib/conditions";

/** Recorded on 2026-10-13 from NWS (PZZ535 Monterey Bay), NDBC 46042 and CO-OPS 9413450 (Monterey). */
const F = path.resolve(__dirname, "fixtures/conditions");
const read = (f: string) => readFileSync(path.join(F, f), "utf8");

describe("NDBC observations", () => {
  const text = read("ndbc-46042.txt");
  const now = new Date("2026-10-13T06:00:00Z");

  it("takes each field from its latest valid row, with that row's time, converted to kt / ft", () => {
    const o = parseNdbc(text, now);
    // the newest row has no waves ("MM"); waves come from the next row, 10 minutes earlier
    expect(o.windKt?.at).toBe("2026-10-13T05:30:00.000Z");
    expect(o.windKt?.value).toBeCloseTo(3.0 * 1.943844, 3);
    expect(o.waveFt?.at).toBe("2026-10-13T05:20:00.000Z");
    expect(o.waveFt?.value).toBeCloseTo(2.9 * 3.28084, 3);
    expect(o.periodS).toEqual({ value: 17, at: "2026-10-13T05:20:00.000Z" });
    expect(o.waterC).toEqual({ value: 17, at: "2026-10-13T05:30:00.000Z" });
    expect(compass(o.windDirDeg!.value)).toBe("NNW");
  });

  it("never reports old observations as current", () => {
    expect(parseNdbc(text, new Date("2026-10-14T06:00:00Z"))).toEqual({});
    expect(parseNdbc("", now)).toEqual({});
  });
});

describe("NWS marine zone forecast and alerts", () => {
  it("keeps the forecasters' text and the issue time", () => {
    const f = parseZoneForecast("PZZ535", "Monterey Bay", JSON.parse(read("pzz535-forecast.json")))!;
    expect(f.updated).toBe("2026-10-12T20:10:00-07:00");
    expect(f.periods[0].name).toBe("Tonight");
    expect(f.periods[0].text).toMatch(/^W wind 10 to 15 kt/);
    expect(marineHeadline(f.periods[0])).toBe("W wind 10 to 15 kt · Seas 3 to 5 ft");
  });

  it("an empty forecast is not a forecast", () => {
    expect(parseZoneForecast("PZZ535", "Monterey Bay", { properties: { periods: [] } })).toBeNull();
  });

  it("lists active alerts with their times", () => {
    const a = parseAlerts(JSON.parse(read("alerts-pz.json")));
    expect(a.length).toBe(11);
    expect(a[0].event).toBe("Small Craft Advisory");
    expect(a[0].url).toMatch(/^https:\/\/api\.weather\.gov\/alerts\//);
  });
});

describe("tide predictions", () => {
  it("are high and low water times in local time, in feet above MLLW", () => {
    const t = parseTides(JSON.parse(read("tides-9413450.json")));
    expect(t[0]).toEqual({ at: "2026-10-12 00:22", ft: 4.197, type: "H" });
    expect(t.every((x) => x.type === "H" || x.type === "L")).toBe(true);
  });
});

describe("station lists", () => {
  it("find the nearest stations to Monterey within the limit", () => {
    expect(nearest(TIDE_STATIONS, 36.6, -121.89, 1, 40)[0].id).toBe("9413450");
    const b = nearest(BUOYS, 36.6, -121.89, 3, 60);
    expect(b.length).toBeGreaterThan(0);
    expect(b.every((x) => x.km <= 60)).toBe(true);
    expect(nearest(BUOYS, 40.0, -119.0, 3, 60)).toEqual([]); // inland Nevada: none
  });
});

describe("CO-OPS water temperature in the harbour", () => {
  const NOW = new Date("2026-10-13T06:00:00Z");
  const fx = JSON.parse(readFileSync(path.join(__dirname, "../conditions-fixtures.json"), "utf8")) as Record<string, string>;
  const recorded = (station: string) => {
    const url = `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_temperature&station=${station}&begin_date=20261013%2003%3A00&end_date=20261013%2006%3A00&units=metric&time_zone=gmt&format=json&application=neritic`;
    const key = `${createHash("sha1").update(url).digest("hex")}.json`;
    expect(fx[key], `recorded response for ${station}`).toBeDefined();
    return JSON.parse(fx[key]);
  };

  it("reads the newest flagged-clean 6-minute reading at Monterey's harbour station (recorded 2026-10-13)", () => {
    const t = parseWaterTemp(recorded("9413450"), NOW)!;
    expect(t).not.toBeNull();
    expect(t.waterC.value).toBeGreaterThan(5);
    expect(t.waterC.value).toBeLessThan(30);
    expect(t.waterC.at).toBe("2026-10-13T06:00:00.000Z");
    expect(t.flags).toBe("0,0,0");
  });

  it("a station whose sensor is down ('No data') is null, not a value", () => {
    expect(parseWaterTemp(recorded("9414290"), NOW)).toBeNull();
    expect(parseWaterTemp({ error: { message: "No data was found." } }, NOW)).toBeNull();
  });

  it("skips readings that fail NOAA's flags, readings older than three hours, and readings from the future", () => {
    const row = (t: string, v: string, f = "0,0,0") => ({ t, v, f });
    expect(parseWaterTemp({ data: [row("2026-10-13 05:48", "17.1"), row("2026-10-13 05:54", "99.0", "1,0,0")] }, NOW)!.waterC.value).toBe(17.1);
    expect(parseWaterTemp({ data: [row("2026-10-13 02:54", "17.1")] }, NOW)).toBeNull();
    expect(parseWaterTemp({ data: [row("2026-10-13 07:30", "17.1")] }, NOW)).toBeNull();
    expect(parseWaterTemp({ data: [] }, NOW)).toBeNull();
  });

  it("the bundled station list puts a thermistor inside Monterey, Crescent City and San Diego harbours and none near Santa Cruz", () => {
    const at = (lat: number, lon: number) => nearest(WATER_TEMP_STATIONS, lat, lon, 1, HARBOR_TEMP_MAX_KM)[0];
    expect(at(36.60024, -121.89469)).toMatchObject({ id: "9413450" });
    expect(at(36.60024, -121.89469).km).toBeLessThan(HARBOR_TEMP_AT_KM);
    expect(at(41.74901, -124.20398)).toMatchObject({ id: "9419750" });
    expect(at(32.71482, -117.1335)).toMatchObject({ id: "9410170" });
    expect(at(36.9741, -122.0308)).toBeUndefined(); // Santa Cruz: Monterey's is 40 km away across the bay
  });
});

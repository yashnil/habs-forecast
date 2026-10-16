import { readFileSync } from "node:fs";
import path from "node:path";
import { mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { BUOYS, TIDE_STATIONS, WATER_TEMP_STATIONS } from "@/lib/noaaStations";
import {
  compass,
  harborCandidate,
  marineHeadline,
  nearest,
  parseAlerts,
  parseNdbc,
  parseTides,
  parseWaterTemp,
  parseWaterTempSeries,
  parseZoneForecast,
  waterTempFlagNote,
  waterTempFlagsPass,
  HARBOR_TEMP_AT_KM,
  HARBOR_TEMP_MAX_KM,
  HARBOR_TEMP_STALE_H,
  WATER_TEMP_FLAGS,
} from "@/lib/conditions";

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
  const FIXTURES = path.join(__dirname, "../conditions-fixtures.json");
  const fx = JSON.parse(readFileSync(FIXTURES, "utf8")) as Record<string, string>;
  /** the route's request for the fixed clock: the three hours up to 2026-10-13 06:00 GMT */
  const waterTempUrl = (station: string) =>
    `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_temperature&station=${station}&begin_date=20261013%2003%3A00&end_date=20261013%2006%3A00&units=metric&time_zone=gmt&format=json&application=neritic`;
  const recorded = (station: string) => {
    const key = `${createHash("sha1").update(waterTempUrl(station)).digest("hex")}.json`;
    expect(fx[key], `recorded response for ${station}`).toBeDefined();
    return JSON.parse(fx[key]);
  };
  const row = (t: string, v: string, f = "0,0,0") => ({ t, v, f });

  it("reads the newest flagged-clean 6-minute reading at Monterey's harbour station (recorded 2026-10-13), with its flags in words", () => {
    const t = parseWaterTemp(recorded("9413450"), NOW)!;
    expect(t).not.toBeNull();
    expect(t.waterC.value).toBeGreaterThan(5);
    expect(t.waterC.value).toBeLessThan(30);
    expect(t.waterC.at).toBe("2026-10-13T06:00:00.000Z");
    expect(t.flags).toBe("0,0,0");
    expect(t.flagNote).toBe("passed NOAA's checks: maximum, minimum, rate of change");
    expect(t.stale).toBe(false);
  });

  it("every one of the 16 bundled stations has a recording for the same window; 12 report, 4 answer 'No data'", () => {
    const reporting = WATER_TEMP_STATIONS.filter((s) => parseWaterTemp(recorded(s.id), NOW) !== null).map((s) => s.id);
    expect(reporting.length).toBe(12);
    expect(WATER_TEMP_STATIONS.map((s) => s.id).filter((id) => !reporting.includes(id))).toEqual(["9410660", "9414290", "9414523", "9415102"]);
  });

  it("a station whose sensor is down ('No data') is null, not a value, and the series says why", () => {
    expect(parseWaterTemp(recorded("9414290"), NOW)).toBeNull();
    const s = parseWaterTempSeries({ error: { message: "No data was found." } }, NOW);
    expect(s).toEqual({ reading: null, rows: 0, flagged: 0, outOfWindow: 0, error: "No data was found." });
    expect(parseWaterTempSeries(null, NOW).reading).toBeNull();
    expect(parseWaterTempSeries({ data: "nonsense" }, NOW).rows).toBe(0);
  });

  describe("NOAA's three flags (X,N,R: maximum, minimum, rate of change), each rejecting a reading", () => {
    it("are documented in order", () => {
      expect(WATER_TEMP_FLAGS.map((f) => f.code)).toEqual(["X", "N", "R"]);
    });
    it.each([
      ["1,0,0", "above the maximum expected water temperature"],
      ["0,1,0", "below the minimum expected water temperature"],
      ["0,0,1", "changed faster than the rate-of-change tolerance"],
    ])("flags %s: the reading is set aside and the note says %s", (flags, meaning) => {
      expect(waterTempFlagsPass(flags)).toBe(false);
      expect(waterTempFlagNote(flags)).toBe(`set aside: ${meaning}`);
      // the newest row carries the flag: the clean row before it is used instead
      const s = parseWaterTempSeries({ data: [row("2026-10-13 05:48", "17.1"), row("2026-10-13 05:54", "99.0", flags)] }, NOW);
      expect(s.reading!.waterC.value).toBe(17.1);
      expect(s.reading!.waterC.at).toBe("2026-10-13T05:48:00.000Z");
      expect(s.flagged).toBe(1);
    });
    it("two failures are both named; a count above 1, an unknown position and absent flags are handled conservatively", () => {
      expect(waterTempFlagNote("1,0,1")).toBe("set aside: above the maximum expected water temperature; changed faster than the rate-of-change tolerance");
      expect(waterTempFlagsPass("2,0,0")).toBe(false);
      expect(waterTempFlagsPass("0,0,0,1")).toBe(false);
      expect(waterTempFlagNote("0,0,0,1")).toBe("set aside: failed NOAA check 4");
      expect(waterTempFlagsPass("0,0,0")).toBe(true);
      expect(waterTempFlagsPass("")).toBe(true);
      expect(waterTempFlagNote("")).toBe("NOAA reported no quality flags for this reading");
      expect(parseWaterTemp({ data: [{ t: "2026-10-13 05:54", v: "17.1" }] }, NOW)!.flagNote).toBe("NOAA reported no quality flags for this reading");
    });
  });

  describe("freshness", () => {
    it("no rows, all rows flagged, and rows older than three hours or more than an hour in the future give no value", () => {
      expect(parseWaterTempSeries({ data: [] }, NOW)).toMatchObject({ reading: null, rows: 0 });
      const allFlagged = parseWaterTempSeries({ data: [row("2026-10-13 05:48", "17.1", "0,0,1"), row("2026-10-13 05:54", "17.2", "1,0,0")] }, NOW);
      expect(allFlagged).toMatchObject({ reading: null, rows: 2, flagged: 2, outOfWindow: 0 });
      expect(parseWaterTempSeries({ data: [row("2026-10-13 02:54", "17.1")] }, NOW)).toMatchObject({ reading: null, outOfWindow: 1 });
      expect(parseWaterTemp({ data: [row("2026-10-13 03:00", "17.1")] }, NOW)).not.toBeNull(); // exactly three hours: still in the window
      expect(parseWaterTempSeries({ data: [row("2026-10-13 07:30", "17.1")] }, NOW)).toMatchObject({ reading: null, outOfWindow: 1 });
      expect(parseWaterTempSeries({ data: [row("garbage", "17.1"), row("2026-10-13 05:54", "n/a")] }, NOW)).toMatchObject({ reading: null, outOfWindow: 2 });
    });

    it("a reading older than an hour is returned but marked stale, so the UI can say when it was taken", () => {
      expect(HARBOR_TEMP_STALE_H).toBe(1);
      expect(parseWaterTemp({ data: [row("2026-10-13 04:54", "17.1")] }, NOW)).toMatchObject({ waterC: { at: "2026-10-13T04:54:00.000Z" }, stale: true });
      expect(parseWaterTemp({ data: [row("2026-10-13 05:06", "17.1")] }, NOW)).toMatchObject({ stale: false });
      expect(parseWaterTemp({ data: [row("2026-10-13 05:00", "17.1")] }, NOW)).toMatchObject({ stale: false }); // exactly an hour is not yet stale
      // the newest clean row decides, even when a flagged newer row exists
      expect(parseWaterTemp({ data: [row("2026-10-13 04:30", "17.1"), row("2026-10-13 05:54", "17.1", "0,1,0")] }, NOW)).toMatchObject({ stale: true });
    });
  });

  describe("which station may speak for a harbour (nearest only, within 6 km, no fallback)", () => {
    const PORTS = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/data/v1/ports-adfcb13636.geojson"), "utf8")) as {
      features: { properties: { name: string }; geometry: { coordinates: number[] } }[];
    };
    const port = (name: string) => {
      const f = PORTS.features.find((x) => x.properties.name === name)!;
      expect(f, name).toBeDefined();
      return { lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0] };
    };
    const candidate = (name: string) => {
      const p = port(name);
      return harborCandidate(WATER_TEMP_STATIONS, p.lat, p.lon);
    };

    it("the limits are harbour scale", () => {
      expect(HARBOR_TEMP_AT_KM).toBe(2);
      expect(HARBOR_TEMP_MAX_KM).toBe(6);
    });

    it.each([
      ["MONTEREY", "9413450", "in-harbour"],
      ["CRESCENT CITY", "9419750", "in-harbour"],
      ["AVILA/PORT SAN LUIS", "9412110", "in-harbour"],
      ["POINT ARENA", "9416841", "in-harbour"],
      ["EUREKA", "9418767", "nearby-station"],
      ["SAN DIEGO", "9410170", "nearby-station"],
      ["SAN FRANCISCO", "9414290", "nearby-station"],
      ["SAN PEDRO", "9410660", "nearby-station"],
    ])("%s → station %s (%s)", (name, id, basis) => {
      const c = candidate(name)!;
      expect(c).toMatchObject({ id, basis });
      expect(c.km).toBeLessThanOrEqual(HARBOR_TEMP_MAX_KM);
      if (basis === "in-harbour") expect(c.km).toBeLessThanOrEqual(HARBOR_TEMP_AT_KM);
      else expect(c.km).toBeGreaterThan(HARBOR_TEMP_AT_KM);
    });

    it.each(["TRINIDAD", "SHELTER COVE", "FORT BRAGG", "BODEGA BAY", "PRINCETON-HALF MOON", "SANTA CRUZ", "MOSS LANDING", "MORRO BAY", "SANTA BARBARA HARBOR", "VENTURA", "PORT HUENEME", "NEWPORT BEACH", "DANA POINT", "OCEANSIDE"])(
      "%s → no station within 6 km",
      (name) => {
        expect(candidate(name)).toBeNull();
      },
    );

    it("San Francisco's own station (4.4 km) is the only candidate; Alameda (10.9 km, across the bay) is never a fallback", () => {
      const p = port("SAN FRANCISCO");
      const byDistance = nearest(WATER_TEMP_STATIONS, p.lat, p.lon, 3, 20).map((s) => [s.id, Math.round(s.km * 10) / 10]);
      expect(byDistance).toEqual([
        ["9414290", 4.4],
        ["9414750", 10.9],
        ["9414863", 13.4],
      ]);
      expect(harborCandidate(WATER_TEMP_STATIONS, p.lat, p.lon)).toMatchObject({ id: "9414290" });
    });

    it("Eureka is 5.7 km from North Spit: within the limit, named with its distance", () => {
      expect(candidate("EUREKA")!.km).toBeCloseTo(5.7, 1);
    });

    it("every bundled station states what NOAA's metadata says about its thermistor, and the four that do not report say so", () => {
      expect(WATER_TEMP_STATIONS.length).toBe(16);
      for (const s of WATER_TEMP_STATIONS) expect(s.sensorNote, s.id).toMatch(/mdapi sensors\.json, read 2026-10-15/);
      const note = (id: string) => WATER_TEMP_STATIONS.find((s) => s.id === id)!.sensorNote!;
      expect(note("9413450")).toMatch(/^Thermistor E1 at 6\.8 ft \(2\.1 m\) below MLLW .*; sensor active\.$/);
      expect(note("9410660")).toMatch(/^No water-temperature sensor/);
      for (const id of ["9414290", "9414523", "9415102"]) expect(note(id)).toMatch(/inactive since 20(25-12|26-03)-\d\d: Suspect Data/);
    });
  });

  describe("the conditions route, for the recorded 2026-10-13 window", () => {
    afterEach(() => {
      vi.unstubAllGlobals();
      vi.unstubAllEnvs();
      vi.resetModules();
    });
    const harbor = async (name: string, fixture = FIXTURES) => {
      vi.stubEnv("CW_NWS_API", "");
      vi.stubEnv("CW_NWS_TEXT", "0");
      vi.stubEnv("CW_FIXED_NOW", NOW.toISOString());
      vi.stubEnv("CW_CONDITIONS_FIXTURE", fixture);
      const f = (JSON.parse(readFileSync(path.resolve(__dirname, "../../public/data/v1/ports-adfcb13636.geojson"), "utf8")) as { features: { properties: { name: string }; geometry: { coordinates: number[] } }[] }).features.find((x) => x.properties.name === name)!;
      const { GET } = await import("@/app/api/conditions/route");
      const body = await (await GET(new Request(`http://x/api/conditions?lat=${f.geometry.coordinates[1]}&lon=${f.geometry.coordinates[0]}`))).json();
      return body.harbor;
    };

    it("Monterey: in the harbour, with the sensor's placement", async () => {
      const h = await harbor("MONTEREY");
      expect(h).toMatchObject({ status: "ok", basis: "in-harbour", station: { id: "9413450", name: "Monterey" }, data: { flags: "0,0,0", stale: false }, url: "https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450" });
      expect(h.station.km).toBeLessThan(HARBOR_TEMP_AT_KM);
      expect(h.station.sensorNote).toMatch(/2\.1 m\) below MLLW/);
      expect(h.data.waterC).toEqual({ value: 17.6, at: "2026-10-13T06:00:00.000Z" });
    });

    it("Crescent City, Eureka and San Diego: the recorded readings, with the basis stated", async () => {
      expect(await harbor("CRESCENT CITY")).toMatchObject({ status: "ok", basis: "in-harbour", station: { id: "9419750" }, data: { waterC: { value: 11.0 } } });
      const e = await harbor("EUREKA");
      expect(e).toMatchObject({ status: "ok", basis: "nearby-station", station: { id: "9418767", name: "North Spit" }, data: { waterC: { value: 13.1 } } });
      expect(e.station.km).toBeCloseTo(5.7, 1);
      expect(await harbor("SAN DIEGO")).toMatchObject({ status: "ok", basis: "nearby-station", station: { id: "9410170" }, data: { waterC: { value: 24.2 } } });
    });

    it("San Francisco: its own station is down, so the part is unavailable; Alameda is never substituted", async () => {
      const h = await harbor("SAN FRANCISCO");
      expect(h.status).toBe("unavailable");
      expect(h.reason).toBe("NOAA tide station 9414290 (San Francisco, 4.4 km): sensor not reporting.");
      expect(h.url).toBe("https://tidesandcurrents.noaa.gov/stationhome.html?id=9414290");
      expect(JSON.stringify(h)).not.toMatch(/9414750|Alameda/);
    });

    it("San Pedro: Los Angeles 9410660 is 2.6 km away but has no reporting sensor", async () => {
      const h = await harbor("SAN PEDRO");
      expect(h).toMatchObject({ status: "unavailable", reason: "NOAA tide station 9410660 (Los Angeles, 2.6 km): sensor not reporting." });
    });

    it("Santa Cruz: no station within 6 km (Monterey's is 42 km away across the bay)", async () => {
      expect(await harbor("SANTA CRUZ")).toEqual({ status: "none", reason: "No NOAA water-temperature station within 6 km.", source: "NOAA CO-OPS water temperature" });
    });

    it("a station that is listed but whose response is missing (recorded as an outage) is unavailable, not none", async () => {
      const dir = mkdtempSync(path.join(os.tmpdir(), "cw-fx-"));
      const empty = path.join(dir, "none.json");
      writeFileSync(empty, "{}");
      const h = await harbor("MONTEREY", empty);
      expect(h).toMatchObject({ status: "unavailable", reason: "NOAA tide station 9413450 (Monterey, 1.0 km): the CO-OPS service did not answer (HTTP 503 (not recorded))." });
    });

    it("an HTTP 500 from CO-OPS is an outage of the service, named as such", async () => {
      vi.stubEnv("CW_NWS_API", "");
      vi.stubEnv("CW_NWS_TEXT", "0");
      vi.stubEnv("CW_FIXED_NOW", NOW.toISOString());
      vi.stubEnv("CW_CONDITIONS_FIXTURE", "");
      vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
      const { GET } = await import("@/app/api/conditions/route");
      const body = await (await GET(new Request("http://x/api/conditions?lat=36.60024&lon=-121.89469"))).json();
      expect(body.harbor).toMatchObject({ status: "unavailable", reason: /^NOAA tide station 9413450 \(Monterey, 1\.0 km\): the CO-OPS service did not answer \(HTTP 500\)\.$/ });
    });

    it("a window with only flagged readings is unavailable and says so (not a value, not an outage)", async () => {
      vi.stubEnv("CW_NWS_API", "");
      vi.stubEnv("CW_NWS_TEXT", "0");
      vi.stubEnv("CW_FIXED_NOW", NOW.toISOString());
      vi.stubEnv("CW_CONDITIONS_FIXTURE", "");
      const flagged = JSON.stringify({ data: [row("2026-10-13 05:48", "17.1", "0,0,1"), row("2026-10-13 05:54", "40.0", "1,0,0")] });
      vi.stubGlobal("fetch", vi.fn(async (u: string) => (String(u).includes("product=water_temperature") ? new Response(flagged, { status: 200 }) : new Response("", { status: 404 }))));
      const { GET } = await import("@/app/api/conditions/route");
      const body = await (await GET(new Request("http://x/api/conditions?lat=36.60024&lon=-121.89469"))).json();
      expect(body.harbor).toMatchObject({ status: "unavailable", reason: "NOAA tide station 9413450 (Monterey, 1.0 km): every reading in the last 3 h failed NOAA's quality checks." });
    });
  });
});
